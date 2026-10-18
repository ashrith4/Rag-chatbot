# 📄 Document Q&A RAG Chatbot

> A production-style **Retrieval-Augmented Generation (RAG)** pipeline that lets you chat with any PDF or document set.  
> Built with LangChain · FAISS · HuggingFace Embeddings · Groq LLM · FastAPI · Streamlit · Docker.

🔗 **Live Demo:** _[Add your deployed URL here]_

---

## 🏗️ Architecture

```
PDF / TXT files
      │
      ▼
┌─────────────────────┐
│   PyPDFLoader       │   Load raw text from document
│   TextLoader        │   (preserves page numbers for citations)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ RecursiveCharacter  │   Split into 500-token chunks
│ TextSplitter        │   with 50-token overlap
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ HuggingFace         │   all-MiniLM-L6-v2
│ Embeddings          │   384-dim dense vectors (FREE, runs locally)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ FAISS Vector Store  │   Persisted to disk
│ (local)             │   Instant similarity search
└─────────┬───────────┘
          │   (on query)
          ▼
┌─────────────────────┐
│ LCEL RAG Chain      │   Top-4 chunks → prompt → LLM
│ (LangChain)         │   Returns answer + source citations
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────────┐
│ FastAPI Backend     │◄────│ Streamlit Chat UI     │
│ POST /upload        │     │ File upload sidebar   │
│ POST /query         │────►│ Chat with history     │
│ GET  /documents     │     │ 📄 Source citations   │
│ GET  /health        │     └──────────────────────┘
└─────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| RAG Orchestration | LangChain (LCEL) | Chains, retrievers, prompt templates |
| Vector Store | FAISS (local) | Fast similarity search, persisted to disk |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free, runs locally, no API key needed |
| LLM | Groq (Qwen / Llama) | Fast free LLM inference |
| Backend API | FastAPI + Uvicorn | REST API — decouples UI from RAG logic |
| Frontend | Streamlit | Chat UI with session history + citations |
| Containerisation | Docker + Docker Compose | Reproducible, portable deployment |
| Cloud Deploy | Hugging Face Spaces | Free public URL for CV/portfolio |

---

## 🚀 Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ First install downloads `all-MiniLM-L6-v2` (~90MB) and CPU PyTorch (~250MB). Cached after first run.

### 2. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and choose a provider:

**Option A — Groq (recommended, free):**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=qwen/qwen3.8-27b
```
Get a free key at [console.groq.com](https://console.groq.com)

**Option B — OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo
```

**Option C — HuggingFace:**
```env
LLM_PROVIDER=huggingface
HF_API_TOKEN=hf_...
```
> Requires "Inference Providers" enabled on your HF account.

### 3. Run locally (no Docker)

**Terminal 1 — Backend:**
```bash
cd backend
source ../venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
source venv/bin/activate
streamlit run frontend/app.py
```

Open **[http://localhost:8501](http://localhost:8501)** → upload a PDF → start chatting!

Also visit **[http://localhost:8000/docs](http://localhost:8000/docs)** for the full Swagger API UI.

### 4. Run with Docker Compose

```bash
docker compose up --build
```

- Frontend: [http://localhost:8501](http://localhost:8501)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

The FAISS index and uploaded docs are volume-mounted — they survive container restarts.

---

## 📡 API Reference

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `GET` | `/health` | — | Health check + index status |
| `POST` | `/upload` | `multipart/form-data` | Upload PDF/TXT, chunk, embed, index |
| `POST` | `/query` | `{"question": "..."}` | RAG query → answer + source citations |
| `GET` | `/documents` | — | List all indexed documents |
| `DELETE` | `/documents` | — | Clear index and all uploads |

Example query with `curl`:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'
```

---

## 📁 Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py           # FastAPI app — all REST endpoints
│   ├── rag_pipeline.py   # Embeddings, FAISS, LLM, LCEL chain, citations
│   ├── ingestion.py      # PDF/TXT loading + chunking
│   └── config.py         # Settings via Pydantic BaseSettings (.env-driven)
├── frontend/
│   └── app.py            # Streamlit chat UI
├── faiss_index/          # Persisted FAISS index (gitignored)
├── uploaded_docs/        # Uploaded PDFs (gitignored)
├── Dockerfile            # Backend container image
├── Dockerfile.frontend   # Frontend container image
├── docker-compose.yml    # Orchestrate both services
├── requirements.txt      # All dependencies (CPU-only torch)
├── .env.example          # API key template
└── README.md
```

---

## 🐳 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push this repo: `git push` to the Space remote
4. Go to Space Settings → **Variables and Secrets** → add:
   - `LLM_PROVIDER=groq`
   - `GROQ_API_KEY=gsk_...`
   - `GROQ_MODEL=qwen/qwen3.8-27b`
5. Live at: `https://huggingface.co/spaces/YOUR_USERNAME/rag-chatbot`

---

## 🧠 What I Learned

**Chunking strategy is a core interview topic:**  
`chunk_size=500, chunk_overlap=50` is the industry default. Too small → fragments lose context. Too large → retriever is imprecise and wastes tokens. Overlap prevents answers being split at chunk boundaries.

**Why FAISS over a cloud vector DB:**  
For a portfolio project, FAISS local is fast, free, and zero-dependency. In production you'd replace it with Pinecone, Weaviate, or pgvector for multi-tenant scale.

**Source citations are a key differentiator:**  
Returning which document chunks (with page numbers) were used makes the system trustworthy and auditable — a hard requirement in enterprise RAG deployments.

**RAG prompt design prevents hallucinations:**  
The "answer ONLY from the context" instruction stops the LLM from making things up. The "I don't know" fallback builds user trust — critical for production use.

**FastAPI decouples concerns:**  
Putting RAG logic behind a REST API means any frontend (Streamlit, React, mobile app) can use it. This is the real-world production pattern that recruiters recognise.

**Modern LangChain LCEL over deprecated chains:**  
This project uses LangChain Expression Language (LCEL) instead of the deprecated `RetrievalQA` chain — showing awareness of current best practices.

---

## 📄 License

MIT — use freely for your portfolio, learning, and job applications.
# Rag-chatbot
