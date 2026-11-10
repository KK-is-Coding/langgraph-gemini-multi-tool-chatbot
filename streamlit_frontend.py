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

    try:
        response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
        ai_message = content_text(response['messages'][-1].content)
    except Exception as exc:
        text = str(exc)
        if '503' in text or 'UNAVAILABLE' in text:
            ai_message = '⚠️ Gemini is busy right now (503). Please wait a few seconds and ask again.'
        elif '429' in text or 'RESOURCE_EXHAUSTED' in text:
            ai_message = '⚠️ Gemini quota/rate limit reached (429). Wait a bit or switch GEMINI_MODEL.'
        else:
            ai_message = f'⚠️ Something went wrong: {text[:300]}'
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)