"""
Streamlit chat UI for the RAG chatbot.

Features:
  - File upload (PDF / TXT) → sends to FastAPI /upload
  - Chat interface with full session history
  - Source citations expandable below each answer
  - Sidebar shows indexed documents and allows clearing the index
  - Works with any FastAPI backend URL (set BACKEND_URL in .env)

Run:
    streamlit run frontend/app.py
"""

import os
import httpx
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120  # seconds — LLM inference can be slow

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialisation ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str, "sources": []}]

if "documents" not in st.session_state:
    st.session_state.documents = []


# ── Helper Functions ──────────────────────────────────────────────────────────

def fetch_documents() -> list[str]:
    """Fetch list of indexed documents from the backend."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/documents", timeout=10)
        resp.raise_for_status()
        return [d["filename"] for d in resp.json()]
    except Exception:
        return []


def upload_file(file_bytes: bytes, filename: str) -> dict:
    """Upload a file to the backend and return the response."""
    resp = httpx.post(
        f"{BACKEND_URL}/upload",
        files={"file": (filename, file_bytes, "application/octet-stream")},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def ask_question(question: str) -> dict:
    """Send a question to the backend and return answer + sources."""
    resp = httpx.post(
        f"{BACKEND_URL}/query",
        json={"question": question},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def clear_index() -> None:
    """Clear all documents and the FAISS index."""
    httpx.delete(f"{BACKEND_URL}/documents", timeout=30)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document Q&A")
    st.caption("Powered by LangChain + FAISS + HuggingFace")
    st.divider()

    # ── File Upload ──────────────────────────────────────────────────────────
    st.subheader("📤 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        help="Upload a document to add it to the knowledge base.",
    )

    if uploaded_file is not None:
        with st.spinner(f"Ingesting **{uploaded_file.name}**…"):
            try:
                result = upload_file(uploaded_file.getvalue(), uploaded_file.name)
                st.success(
                    f"✅ **{result['filename']}** indexed!\n\n"
                    f"{result['chunks_indexed']} chunks created."
                )
                st.session_state.documents = fetch_documents()
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e))
                st.error(f"Upload failed: {detail}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()

    # ── Indexed Documents ────────────────────────────────────────────────────
    st.subheader("📚 Indexed Documents")
    docs = fetch_documents()
    if docs:
        for doc in docs:
            st.markdown(f"• {doc}")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # ── Controls ─────────────────────────────────────────────────────────────
    st.subheader("⚙️ Controls")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("💣 Clear All Documents & Index", type="secondary"):
        with st.spinner("Clearing index…"):
            try:
                clear_index()
                st.session_state.messages = []
                st.success("Index cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()
    st.caption("ℹ️ Backend: `" + BACKEND_URL + "`")


# ── Main Chat Area ────────────────────────────────────────────────────────────
st.title("💬 Chat with Your Documents")

# Check backend health on first load
try:
    health = httpx.get(f"{BACKEND_URL}/health", timeout=5).json()
    if not health.get("index_loaded"):
        st.info("👈 Upload a document in the sidebar to get started.")
except Exception:
    st.error(
        f"⚠️ Cannot connect to backend at `{BACKEND_URL}`. "
        "Make sure the FastAPI server is running:\n\n"
        "```bash\ncd backend && uvicorn main:app --reload\n```"
    )

# Render existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Source citations (only on assistant messages)
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])} chunks)", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src['source']}** — Page {src['page']}\n\n"
                        f"> {src['content']}"
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = ask_question(prompt)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                # Show sources inline
                if sources:
                    with st.expander(f"📄 Sources ({len(sources)} chunks)", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**[{i}] {src['source']}** — Page {src['page']}\n\n"
                                f"> {src['content']}"
                            )

                # Persist to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e))
                error_msg = f"❌ Error: {detail}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })
            except Exception as e:
                error_msg = f"❌ Connection error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })
