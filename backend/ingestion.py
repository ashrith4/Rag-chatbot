"""
PDF / document ingestion and chunking pipeline.

Supported formats: PDF, plain text (.txt)
Key design: chunks carry metadata (source filename + page number) so the
Streamlit UI can show source citations alongside answers.

Interview talking point:
    chunk_size=500, chunk_overlap=50 is the industry sweet spot.
    - Too small → each chunk loses context, LLM gets confused fragments.
    - Too large → retriever returns fewer, less precise chunks; you waste tokens.
    - Overlap=50 prevents answers from being cut in half at chunk boundaries.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def load_document(file_path: str) -> List[Document]:
    """
    Load a single document from disk and return raw LangChain Documents.

    Each Document contains:
        - page_content: the raw text of that page
        - metadata: {"source": filename, "page": page_number}
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported: .pdf, .txt"
        )

    docs = loader.load()

    # Normalise metadata: always have "source" as just the filename
    for doc in docs:
        doc.metadata["source"] = path.name

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on:
        paragraph → sentence → word → character
    This preserves semantic coherence better than naive fixed-size splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,  # adds "start_index" to metadata — useful for debugging
    )

    chunks = splitter.split_documents(docs)

    # Add chunk index to metadata for cleaner citation display
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def ingest_file(file_path: str) -> List[Document]:
    """
    Full ingestion pipeline: load → chunk.
    Returns a list of Document chunks ready for embedding.
    """
    raw_docs = load_document(file_path)
    chunks = chunk_documents(raw_docs)
    return chunks


def load_all_from_dir(directory: str = None) -> List[Document]:
    """
    Load and chunk every supported document in a directory.
    Useful for initial bulk ingestion.
    """
    directory = directory or settings.upload_dir
    all_chunks: List[Document] = []

    for filename in os.listdir(directory):
        ext = Path(filename).suffix.lower()
        if ext in {".pdf", ".txt"}:
            file_path = os.path.join(directory, filename)
            try:
                chunks = ingest_file(file_path)
                all_chunks.extend(chunks)
                print(f"✅ Ingested {filename}: {len(chunks)} chunks")
            except Exception as e:
                print(f"❌ Failed to ingest {filename}: {e}")

    return all_chunks
