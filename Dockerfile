# ── Backend Dockerfile ────────────────────────────────────────────────────────
# FastAPI + RAG pipeline on port 8000

FROM python:3.11-slim

# Install system dependencies for pypdf, sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy .env if it exists (ignored in production — use env vars instead)
COPY .env* ./

# Create storage directories
RUN mkdir -p /app/faiss_index /app/uploaded_docs

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
