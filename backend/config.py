"""
Central configuration for the RAG chatbot.

Copy .env.example to .env and fill in your values.
All settings can be overridden via environment variables.
"""

from pydantic_settings import BaseSettings
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    # Supported: "openai" | "huggingface" | "groq"
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    hf_api_token: str = ""
    hf_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # HuggingFace sentence-transformers — completely free, no API key needed
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── FAISS Vector Store ────────────────────────────────────────────────────
    faiss_index_path: str = str(BASE_DIR / "faiss_index")
    allow_dangerous_deserialization: bool = True  # Required for local FAISS load

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_k: int = 4  # Number of chunks to retrieve per query

    # ── Document Storage ──────────────────────────────────────────────────────
    upload_dir: str = str(BASE_DIR / "uploaded_docs")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Frontend ──────────────────────────────────────────────────────────────
    # URL the Streamlit app uses to call the backend
    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
