"""
FastAPI backend for the RAG chatbot.

Endpoints:
    POST /upload        Upload a PDF or TXT file → ingest + embed
    POST /query         Ask a question → RAG answer + source citations
    GET  /documents     List all indexed documents
    DELETE /documents   Clear the entire FAISS index
    GET  /health        Health check

The vectorstore is loaded once at startup and kept in module-level state.
New uploads call add_documents_to_store() which merges into the live index.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add backend dir to path so relative imports work
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from ingestion import ingest_file
from rag_pipeline import (
    load_vectorstore,
    add_documents_to_store,
    query_documents,
)

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description="Document Q&A powered by LangChain + FAISS + HuggingFace",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure storage directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.faiss_index_path, exist_ok=True)

# Module-level vectorstore (loaded once at startup, updated on upload)
_vectorstore = None


@app.on_event("startup")
async def startup():
    """Load persisted FAISS index on startup if it exists."""
    global _vectorstore
    _vectorstore = load_vectorstore()
    if _vectorstore:
        print("✅ FAISS index loaded from disk.")
    else:
        print("ℹ️  No FAISS index found — upload documents to get started.")


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "What are the main findings of the report?"}
        }


class SourceCitation(BaseModel):
    source: str
    page: int
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class DocumentInfo(BaseModel):
    filename: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check — returns index status."""
    return {
        "status": "ok",
        "index_loaded": _vectorstore is not None,
        "upload_dir": settings.upload_dir,
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
    }


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file.
    The file is saved to disk, chunked, embedded, and merged into the FAISS index.
    """
    global _vectorstore

    # Validate file type
    allowed_extensions = {".pdf", ".txt"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    # Save file to disk
    save_path = Path(settings.upload_dir) / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        )

    # Ingest → chunk
    try:
        chunks = ingest_file(str(save_path))
    except Exception as e:
        save_path.unlink(missing_ok=True)  # Clean up on failure
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process document: {e}",
        )

    # Embed → add to vectorstore
    try:
        _vectorstore = add_documents_to_store(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to embed document: {e}",
        )

    return UploadResponse(
        filename=file.filename,
        chunks_indexed=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{file.filename}'.",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question. Returns an answer and the source document chunks used.
    Requires at least one document to have been uploaded first.
    """
    if _vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents indexed yet. Upload a PDF or TXT file first via POST /upload.",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    try:
        result = query_documents(request.question, _vectorstore)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline error: {e}",
        )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
    )


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    upload_path = Path(settings.upload_dir)
    files = [
        DocumentInfo(filename=f.name)
        for f in upload_path.iterdir()
        if f.is_file() and f.suffix.lower() in {".pdf", ".txt"}
    ]
    return files


@app.delete("/documents", status_code=status.HTTP_200_OK)
async def clear_index():
    """
    Clear all uploaded documents and the FAISS index.
    WARNING: This is irreversible.
    """
    global _vectorstore

    # Remove uploaded files
    upload_path = Path(settings.upload_dir)
    for f in upload_path.iterdir():
        if f.is_file():
            f.unlink()

    # Remove FAISS index files
    index_path = Path(settings.faiss_index_path)
    for f in index_path.iterdir():
        if f.is_file():
            f.unlink()

    _vectorstore = None
    return {"message": "All documents and index cleared successfully."}
