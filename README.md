# 🤖 LangGraph Multi-Tool Chatbot

An agentic chatbot built with LangGraph and Streamlit, powered by Google Gemini. It can decide when to search the web, fetch stock prices, do calculations, call MCP tool servers, or answer from a PDF you upload, with streaming replies and chat history that persists across sessions.

## 🌟 Features

- **Tool Calling**: The model chooses between web search, stock quotes and a calculator on its own
- **PDF Question Answering (RAG)**: Upload a PDF per chat; answers come from FAISS similarity search over its text
- **MCP Integration**: Load tools from Model Context Protocol servers (stdio or HTTP)
- **Persistent Memory**: Conversations are stored in SQLite and restored from the sidebar
- **Multi-Thread Chats**: Start new chats and switch between old ones
- **Streaming UI**: Token-by-token replies with a live "Using tool…" status
- **Resilient by Design**: Timeouts, retries, loop limits and friendly error messages (busy model, quota, bad model name)
- **Gemini Model Switching**: Pick any Gemini model your key supports with one environment variable (`GEMINI_MODEL`)

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **Google Gemini API key**: Required (free tier works, with daily limits). The project currently supports Gemini models only
- **Alpha Vantage API key**: Optional, only for the stock price tool
- **MCP servers**: Optional, only for the MCP version

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/KK-is-Coding/langgraph-gemini-multi-tool-chatbot.git
cd langgraph-gemini-multi-tool-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

**Main packages:**
- `langgraph` - Agent graph orchestration
- `langchain` / `langchain-core` - LLM building blocks
- `langchain-google-genai` - Gemini models and embeddings
- `langchain-mcp-adapters` - MCP tool integration
- `streamlit` - Web interface
- `faiss-cpu` / `pypdf` - PDF indexing and retrieval
- `ddgs` - Web search
- `langgraph-checkpoint-sqlite` - Saved conversations

### 4. Set Up Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# .env

# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional: stock price tool
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Optional: choose the Gemini model (check which ones your key can use)
GEMINI_MODEL=gemini-3.5-flash-lite

# Optional: path to a local MCP server (MCP version only)
MCP_MATH_SERVER_PATH=/path/to/mcp-math-server/main.py
```

**Where to get API keys:**
- **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey)
- **Alpha Vantage**: [alphavantage.co](https://www.alphavantage.co/support/#api-key)

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 5. Run the App

Start the full version with PDF support:

```bash
streamlit run streamlit_rag_frontend.py
```

## 🧭 Versions of the App

The project is built in stages. Each frontend pairs with a backend:

| Frontend | Backend | What it adds |
|---|---|---|
| `streamlit_frontend.py` | `langgraph_backend.py` | Basic chat, in-memory history |
| `streamlit_frontend_streaming.py` | `langgraph_backend.py` | Streaming replies |
| `streamlit_frontend_threading.py` | `langgraph_tool_backend.py` | Multiple chats |
| `streamlit_frontend_database.py` | `langgraph_database_backend.py` | SQLite-saved history |
| `streamlit_frontend_tool.py` | `langgraph_tool_backend.py` | Web search, stocks, calculator |
| `streamlit_frontend_mcp.py` | `langgraph_mcp_backend.py` | MCP tool servers |
| `streamlit_rag_frontend.py` | `langraph_rag_backend.py` | PDF question answering (all features) |

## 💡 Usage

### Basic Chat

```
User: Hello!
Assistant: Hi! How can I help you today?
```

### Using Tools

```
User: What is 123 times 45?
→ 🔧 Using `calculator`
Assistant: 5535

User: What is the price of NVDA?
→ 🔧 Using `get_stock_price`
```

### Chatting With a PDF

1. Upload a PDF in the sidebar and wait for **"PDF indexed"**
2. Ask a question about it
3. The assistant retrieves the most relevant passages and answers from them

> The PDF must contain real, selectable text. Scanned or image-only PDFs need OCR first.

### Past Conversations

Use **New Chat** to start a fresh thread, and click any past thread in the sidebar to reopen it. Chats are stored in `chatbot.db`.

## 📁 Project Structure

```
langgraph-gemini-multi-tool-chatbot/
├── langgraph_backend.py            # Basic chatbot, in-memory
├── langgraph_database_backend.py   # Chatbot with SQLite memory
├── langgraph_tool_backend.py       # Tools: search, stocks, calculator
├── langgraph_mcp_backend.py        # Adds MCP tool servers
├── langraph_rag_backend.py         # Adds PDF RAG (FAISS)
├── streamlit_frontend.py           # Basic UI
├── streamlit_frontend_streaming.py # Streaming UI
├── streamlit_frontend_threading.py # Multi-thread UI
├── streamlit_frontend_database.py  # Persistent history UI
├── streamlit_frontend_tool.py      # Tool-calling UI
├── streamlit_frontend_mcp.py       # MCP UI
├── streamlit_rag_frontend.py       # Full UI with PDF upload
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
└── .gitignore
```

## 🔧 Configuration

### Change the Gemini Model

Set `GEMINI_MODEL` in `.env`. No code changes are needed to switch between Gemini models:

```bash
GEMINI_MODEL=gemini-3.5-flash-lite
```

Model quotas differ. Check the [rate limits page](https://aistudio.google.com/rate-limits) for what your key can use.

### Using a Different AI Provider

The chat model (`ChatGoogleGenerativeAI`) and the PDF embeddings (`GoogleGenerativeAIEmbeddings`) are created directly in the backend files, so OpenAI, Groq or other keys will not work as-is. To add another provider, replace those two objects in the backends with the matching LangChain classes (for example `ChatOpenAI` and `OpenAIEmbeddings` from `langchain-openai`) and add the new API key to `.env`.

### Adjust Behavior

- **System prompt and tool instructions**: `langraph_rag_backend.py` (`chat_node`)
- **Retries and timeouts**: the `ChatGoogleGenerativeAI(...)` block in each backend
- **Tool loop limit**: `recursion_limit` in each frontend's `CONFIG`
- **PDF chunking and retrieval**: `chunk_size`, `chunk_overlap` and `k` in `ingest_pdf`

## 🐛 Troubleshooting

### `429 RESOURCE_EXHAUSTED` / quota exceeded
- The free tier has a small daily request limit per model
- Wait for the reset, set a different `GEMINI_MODEL`, or enable billing

### `404 model not found`
- The model name is wrong or unavailable to your key
- Check the exact name in Google AI Studio and update `GEMINI_MODEL`

### `503 UNAVAILABLE` / high demand
- Temporary problem on Google's side; ask again after a few seconds
- Retries are already built in

### `Function call is missing a thought_signature`
- Your `langchain-google-genai` is too old for current Gemini models
- Run `python -m pip install --upgrade -r requirements.txt`

### "No text could be extracted from the PDF"
- The PDF is probably scanned or image-only
- Use a PDF with selectable text, or run OCR on it first

### Stock tool returns an error or nothing
- Check `ALPHA_VANTAGE_API_KEY` in `.env`
- The free key allows about 25 requests per day
- Coverage is best for US tickers; Indian stocks need an exchange suffix such as `.BSE`

### Web search fails
- It needs internet access; a VPN or firewall can block it
- A failed search is reported to the model instead of crashing the chat

### MCP tools do not load
- Check `MCP_MATH_SERVER_PATH` and that the server runs on its own
- Failed servers are logged as `[MCP] Could not load tools…` and skipped

### `pip install` fails on Windows with "file in use"
- Stop any running `streamlit` process, then run `python -m pip install -r requirements.txt` again

## 📊 System Architecture

```
User Message
    ↓
Streamlit UI ──→ LangGraph (thread_id)
    ↓
chat_node (Gemini with tools)
    ├─→ No tool needed: stream the answer
    └─→ Tool requested
            ↓
        tools node
         ├─ web_search
         ├─ get_stock_price
         ├─ calculator
         ├─ rag_tool ──→ FAISS (per-thread PDF index)
         └─ MCP tools
            ↓
        back to chat_node → final answer
    ↓
SQLite checkpointer (saves the conversation)
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 🔗 Links

- **Repository**: [https://github.com/KK-is-Coding/langgraph-gemini-multi-tool-chatbot](https://github.com/KK-is-Coding/langgraph-gemini-multi-tool-chatbot)

## 📧 Support

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ using LangGraph, Gemini and Streamlit**
