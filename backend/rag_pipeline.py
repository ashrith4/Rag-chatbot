"""
RAG pipeline: embeddings, FAISS vector store, and QA chain.

Architecture:
    Document chunks → HuggingFace Embeddings → FAISS index (persisted to disk)
                         ↓
    User query → embed → FAISS similarity search → top-k chunks
                         ↓
    Prompt template (context + question) → LLM → answer + source citations

Uses modern LangChain LCEL (LangChain Expression Language) — no deprecated chains.

LLM selection (via config):
    "openai"       → OpenAI ChatGPT (requires OPENAI_API_KEY)
    "huggingface"  → Mistral via HF Inference API (requires HF_API_TOKEN)
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


# ── Prompt Template ───────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based strictly on the provided document context.

Rules:
- Answer ONLY from the context provided below.
- If the answer is not contained in the context, respond with: "I don't know based on the provided documents."
- Be concise and precise.
- Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns HuggingFace sentence-transformers embeddings.
    Model: all-MiniLM-L6-v2 (384-dim, fast, great for semantic search, FREE).
    Downloaded once and cached in ~/.cache/huggingface/
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # enables cosine similarity
    )


# ── LLM Selection ─────────────────────────────────────────────────────────────

def get_llm():
    """
    Returns the configured LLM.
    Set LLM_PROVIDER in .env to switch between providers.
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Either add it to .env or switch LLM_PROVIDER=huggingface"
            )
        return ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0,  # deterministic answers for Q&A
        )

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        if not settings.hf_api_token:
            raise ValueError("HF_API_TOKEN is not set.")
        endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_model,
            huggingfacehub_api_token=settings.hf_api_token,
            task="conversational",
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
        return ChatHuggingFace(llm=endpoint)

    elif provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com"
            )
        return ChatGroq(
            model=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=0,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. Choose 'openai' or 'huggingface'."
        )


# ── FAISS Vector Store ────────────────────────────────────────────────────────

def build_vectorstore(docs: List[Document]) -> FAISS:
    """
    Build a new FAISS index from document chunks and save to disk.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(settings.faiss_index_path)
    print(f"✅ FAISS index saved to {settings.faiss_index_path}")
    return vectorstore


def load_vectorstore() -> Optional[FAISS]:
    """
    Load the persisted FAISS index from disk.
    Returns None if no index exists yet.
    """
    index_path = Path(settings.faiss_index_path)
    if not (index_path / "index.faiss").exists():
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        settings.faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=settings.allow_dangerous_deserialization,
    )
    return vectorstore


def add_documents_to_store(new_docs: List[Document]) -> FAISS:
    """
    Add new document chunks to an existing FAISS index (or create one if none exists).
    Merges the new vectors into the existing index and re-saves.
    """
    embeddings = get_embeddings()
    existing = load_vectorstore()

    if existing is None:
        print("ℹ️  No existing index found — creating a new one.")
        vectorstore = FAISS.from_documents(new_docs, embeddings)
    else:
        new_store = FAISS.from_documents(new_docs, embeddings)
        existing.merge_from(new_store)
        vectorstore = existing

    vectorstore.save_local(settings.faiss_index_path)
    print(f"✅ Index updated and saved ({len(new_docs)} new chunks added).")
    return vectorstore


# ── RAG Chain (LCEL) ──────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """Concatenate document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def query_documents(question: str, vectorstore: FAISS) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for a user question using LCEL.

    Flow:
        question → retriever (top-k FAISS chunks) → prompt → LLM → answer
        question → retriever → source metadata (for citations)

    Returns:
        {
            "answer": str,
            "sources": [{"source": filename, "page": page_num, "content": snippet}]
        }
    """
    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
    )

    # Retrieve source docs once and reuse for both answer and citations
    source_docs = retriever.invoke(question)

    # Build answer via LCEL chain: context + question → prompt → LLM → string
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": _format_docs(source_docs),
        "question": question,
    })

    # Format source citations (deduplicated by filename + page)
    sources = []
    seen = set()
    for doc in source_docs:
        meta = doc.metadata
        key = (meta.get("source", ""), meta.get("page", 0))
        if key not in seen:
            seen.add(key)
            content = doc.page_content
            sources.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 0) + 1,  # 1-indexed for humans
                "content": content[:300] + "..." if len(content) > 300 else content,
            })

    return {
        "answer": answer,
        "sources": sources,
    }
