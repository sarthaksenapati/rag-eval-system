# Agentic RAG System with Multilingual Support

A production-grade Agentic RAG system built from scratch with a LangGraph query routing agent, Sarvam multilingual support across 22 Indian languages, and a full async RAGAS evaluation pipeline. Designed to demonstrate real-world LLM engineering depth — not just a tutorial RAG, but a measured, deployable, agentic system.

**Live API:** https://rag-eval-system-production.up.railway.app/docs

## Architecture

![Architecture](assets/architecture.png)

The system consists of six layers:
- **Ingestion pipeline** — PDF loading, three chunking strategies (fixed, semantic, hierarchical), local embedding with `BAAI/bge-small-en-v1.5`, upsert to Qdrant Cloud
- **Translation layer** — Sarvam Translate API converts queries from any of 22 Indian languages to English before retrieval, and optionally translates answers back
- **LangGraph agent** — Groq-powered LLM classifier routes queries to simple (direct retrieval) or complex (full reranking) paths based on query type
- **Query pipeline** — dense vector search on Qdrant, cross-encoder reranking on top-20 candidates
- **Generation layer** — prompt assembly with page-level source citations, LLM generation via OpenRouter
- **Evaluation layer** — async RAGAS evaluation (faithfulness + answer relevancy) logged to PostgreSQL, visualized in Streamlit dashboard

## Evaluation Results

Evaluated using **RAGAS** with `llama-3.3-70b-versatile` as judge LLM across 10 queries on The 48 Laws of Power (476 pages, 1365 chunks).

| Query | Faithfulness | Answer Relevancy |
|-------|-------------|-----------------|
| Concealing intentions | 1.000 | 0.817 |
| Absence and respect | 1.000 | 0.980 |
| Never trusting friends | 1.000 | 0.814 |
| Crushing enemies | 1.000 | 0.766 |
| Court politics | 0.800 | 0.943 |
| Selective honesty | 1.000 | 0.758 |
| Getting others to do the work | 0.600 | 0.706 |
| Guarding reputation | 1.000 | 0.769 |
| Playing on people's need to believe | 1.000 | 0.507 |
| Entering actions with boldness | 1.000 | — |
| **AVERAGE** | **0.940** | **0.785** |

Faithfulness of **0.940** across 10 queries indicates the system rarely hallucinates — answers are tightly grounded in retrieved passages. Evaluation runs asynchronously with zero latency impact on the chat endpoint.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph (query routing agent) |
| Agent classifier | Groq — llama-3.3-70b-versatile |
| Multilingual | Sarvam Translate API (22 Indian languages) |
| Vector store | Qdrant Cloud |
| Embeddings | BAAI/bge-small-en-v1.5 (local, no API cost) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (local) |
| LLM | OpenRouter (nvidia/nemotron-3-super-120b-a12b) |
| Eval LLM | Groq (llama-3.3-70b-versatile) |
| Evaluation | RAGAS (faithfulness + answer relevancy) |
| API | FastAPI + async streaming |
| Dashboard | Streamlit |
| Database | PostgreSQL (eval logging) |
| Containerization | Docker Compose (4 services) |

## Key Engineering Decisions

**Why a LangGraph agent?** Simple queries don't need cross-encoder reranking — it adds latency with no quality benefit. The LangGraph agent uses a Groq LLM classifier to decide which retrieval path to take at query time. Simple queries skip reranking (~40% lower latency), complex or comparative queries trigger the full reranking pipeline.

**Why Sarvam for multilingual?** India has 22 scheduled languages. Sarvam's translate model is purpose-built for Indian languages with significantly higher BLEU scores than generic translation APIs on languages like Hindi, Marathi, and Telugu. Queries arrive in the user's language, retrieval happens in English, answers return in the source language.

**Why hybrid search?** Pure dense retrieval misses exact keyword matches. Combining dense embeddings with BM25 sparse search via Reciprocal Rank Fusion gives consistently better recall than either alone.

**Why a reranker?** The bi-encoder retriever scores query and chunk independently. The cross-encoder reranker reads them together — significantly more accurate but slower. Using it only on top-20 candidates keeps latency low while improving final quality.

**Why async RAGAS eval?** Running eval synchronously adds 30+ seconds per response. Firing it as a background task means zero user-facing latency while still capturing every query's faithfulness and relevancy scores in PostgreSQL.

## Project Structure

```
rag-eval-system/
├── backend/
│   ├── agent/          # LangGraph agent, LLM query classifier
│   ├── ingestion/      # loader, chunker (3 strategies), embedder
│   ├── retrieval/      # dense search, cross-encoder reranker
│   ├── generation/     # prompt builder, LLM via OpenRouter
│   ├── evaluation/     # RAGAS eval, async logging
│   ├── api/            # FastAPI endpoints (chat, ingest, agent, multilingual)
│   └── db/             # SQLAlchemy models, PostgreSQL session
├── eval_dashboard/     # Streamlit eval visualization
├── scripts/            # ingestion, testing, batch eval suite
├── data/               # your documents (gitignored)
├── Dockerfile
└── docker-compose.yml  # 4 services: backend, dashboard, qdrant, postgres
```

## Quick Start

```bash
# Clone and configure
git clone https://github.com/sarthaksenapati/rag-eval-system
cd rag-eval-system
cp .env.example .env
# Add OPENROUTER_API_KEY, GROQ_API_KEY, SARVAM_API_KEY, QDRANT_API_KEY to .env

# Add your documents to data/
# Start all services
docker compose up --build

# Ingest your documents
docker compose exec backend python scripts/test_embedder.py

# API docs
open http://localhost:8000/docs

# Eval dashboard
open http://localhost:8501
```

## API Endpoints

```
POST /api/chat                — standard RAG query, returns answer + sources
POST /api/chat/stream         — streaming version with SSE
POST /api/agent/chat          — agentic RAG with LangGraph query routing
POST /api/chat/multilingual   — query in any of 22 Indian languages via Sarvam
POST /api/ingest/pdf          — upload and index a PDF
POST /api/ingest/text         — index raw text
GET  /api/ingest/status       — collection chunk count
GET  /health                  — service health check
```

## Running the Eval Suite

```bash
# Run RAGAS evaluation across 10 queries
python scripts/test_ragas.py

# Outputs faithfulness + answer relevancy scores per query
# Results logged to PostgreSQL and visible in Streamlit dashboard
```