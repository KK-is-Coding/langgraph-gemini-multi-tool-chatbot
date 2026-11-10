import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage


def content_text(content) -> str:
    """Gemini 3.x returns content as a list of blocks; keep only the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def friendly_error(exc: Exception) -> str:
    """Turn common Gemini/API errors into a short, readable message."""
    text = str(exc)
    low = text.lower()
    if "503" in text or "unavailable" in low or "high demand" in low:
        return "⚠️ Gemini is busy right now (503). Please wait a few seconds and ask again."
    if "429" in text or "resource_exhausted" in low or "quota" in low:
        return "⚠️ Gemini quota/rate limit reached (429). Wait a bit, switch GEMINI_MODEL, or enable billing."
    if "404" in text or "not found" in low:
        return "⚠️ Model not found (404). Check GEMINI_MODEL in your .env."
    if "api key" in low or "api_key" in low or "permission" in low or "403" in text:
        return "⚠️ Gemini rejected the API key (check GOOGLE_API_KEY in .env)."
    return f"⚠️ Something went wrong: {text[:300]}"


def safe_stream(chunks):
    """Stream text; if the model call fails, show a friendly message instead of a traceback."""
    try:
        yield from chunks
    except Exception as exc:
        yield "\n\n" + friendly_error(exc)

# st.session_state -> dict -> 
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

#{'role': 'user', 'content': 'Hi'}
#{'role': 'assistant', 'content': 'Hi=ello'}

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # first add the message to message_history
    with st.chat_message('assistant'):

        ai_message = st.write_stream(safe_stream(
            content_text(message_chunk.content) for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= {'configurable': {'thread_id': 'thread-1'}},
                stream_mode= 'messages'
            )
        ))

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})