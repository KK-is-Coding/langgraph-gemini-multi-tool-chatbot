from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Optional, TypedDict

import requests
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Add it to your .env file."
    )


# ============================================================
# 2. LLM AND EMBEDDINGS
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    temperature=0,
    timeout=60,
    max_retries=4,
    google_api_key=GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# 3. PDF RETRIEVER STORAGE
# ============================================================

_THREAD_RETRIEVERS: dict[str, Any] = {}
_THREAD_METADATA: dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Return the PDF retriever associated with a chat thread."""

    if not thread_id:
        return None

    return _THREAD_RETRIEVERS.get(str(thread_id))


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Load a PDF, split it into chunks, create a FAISS vector store,
    and associate the retriever with a chat thread.
    """

    if not file_bytes:
        raise ValueError("No bytes received for PDF ingestion.")

    if not thread_id:
        raise ValueError("A valid thread_id is required.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            raise ValueError("The uploaded PDF contains no readable pages.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No text could be extracted from this PDF. It is probably a scanned "
                "or image-only PDF (no selectable text). Try a PDF where you can "
                "select and copy the text, or run OCR on it first."
            )

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        )

        thread_id = str(thread_id)

        _THREAD_RETRIEVERS[thread_id] = retriever

        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(documents),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[thread_id].copy()

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# 4. TOOLS
# ============================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform basic arithmetic.

    Supported operations: add, sub, mul, div.
    """

    if operation == "add":
        result = first_num + second_num

    elif operation == "sub":
        result = first_num - second_num

    elif operation == "mul":
        result = first_num * second_num

    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero is not allowed."}

        result = first_num / second_num

    else:
        return {
            "error": f"Unsupported operation: {operation}"
        }

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest available stock quote using Alpha Vantage.

    Examples: AAPL, IBM, TSLA.
    """

    if not ALPHA_VANTAGE_API_KEY:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is missing from the .env file."
        }

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol.strip().upper(),
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}

        if "Note" in data:
            return {"error": data["Note"]}

        if "Information" in data:
            return {"error": data["Information"]}

        quote = data.get("Global Quote")

        if not quote:
            return {
                "error": "No stock quote found. Check the symbol or API response."
            }

        return quote

    except requests.RequestException as exc:
        return {"error": f"Stock API request failed: {str(exc)}"}

    except ValueError:
        return {"error": "The stock API returned invalid JSON."}


@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Search the PDF uploaded in the current chat thread.

    The application must supply the correct thread_id.
    """

    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No PDF is indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    try:
        documents = retriever.invoke(query)

        if not documents:
            return {
                "query": query,
                "context": [],
                "message": "No relevant information was found in the PDF.",
            }

        return {
            "query": query,
            "context": [
                document.page_content
                for document in documents
            ],
            "metadata": [
                document.metadata
                for document in documents
            ],
            "source_file": _THREAD_METADATA.get(
                str(thread_id), {}
            ).get("filename"),
        }

    except Exception as exc:
        return {
            "error": f"PDF retrieval failed: {str(exc)}",
            "query": query,
        }


tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]


# ============================================================
# 5. BIND TOOLS TO GEMINI
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 6. CHAT STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 7. CHAT NODE
# ============================================================

def chat_node(state: ChatState, config=None):
    """
    Process the conversation and allow Gemini to call tools.
    """

    thread_id = None

    if config:
        thread_id = (
            config.get("configurable", {}).get("thread_id")
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful AI assistant.\n\n"

            "You can use these tools:\n"
            "1. DuckDuckGo search for web searches.\n"
            "2. get_stock_price for stock quotes.\n"
            "3. calculator for arithmetic.\n"
            "4. rag_tool for uploaded PDF questions.\n\n"

            "IMPORTANT PDF INSTRUCTIONS:\n"
            "- For questions about an uploaded PDF, use rag_tool.\n"
            "- Always pass the current chat thread ID to rag_tool.\n"
            f"- The current chat thread ID is: {thread_id}\n"
            "- Never invent information from a PDF.\n"
            "- If no PDF is available, ask the user to upload one.\n\n"

            "Use tools whenever they are relevant. "
            "Explain results clearly and accurately."
        )
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {
        "messages": [response]
    }


# ============================================================
# 8. TOOL EXECUTION NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 9. SQLITE CHECKPOINTER
# ============================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn=conn)


# ============================================================
# 10. LANGGRAPH WORKFLOW
# ============================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge(
    "tools",
    "chat_node",
)

chatbot = graph.compile(
    checkpointer=checkpointer,
)


# ============================================================
# 11. HELPER FUNCTIONS
# ============================================================

def retrieve_all_threads():
    try:
        rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
    except sqlite3.OperationalError:
        # Fresh database: the table is created on the first saved chat.
        return []
    return [r[0] for r in rows]


def thread_has_document(thread_id: str) -> bool:
    """Check whether a PDF is indexed for a thread."""

    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """Return PDF metadata associated with a thread."""

    return _THREAD_METADATA.get(
        str(thread_id), {}
    ).copy()