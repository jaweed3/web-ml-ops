# LibraryRAG — Real-Time AI Librarian

RAG pipeline with real-time document streaming, vector search, and LLM serving.
Use case: AI chatbot for campus library catalog, collections, and policies.

---

## Architecture

```
                         ┌──────────────────────┐
                         │   Data Sources       │
                         │  (Library DB, PDF,   │
                         │   Web Crawl, Webhook)│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Redpanda        │
                         │  (streaming queue)   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         │                      │
                         ▼                      ▼
                 ┌──────────────┐     ┌──────────────────┐
                 │   Chunker    │     │   Chunker B      │
                 │ (semantic)   │     │ (recursive)      │
                 └──────┬───────┘     └────────┬─────────┘
                        │                      │
                        └──────────┬───────────┘
                                   ▼
                         ┌──────────────────────┐
                         │      Embedder        │
                         │ (ONNX MiniLM-L6-v2)  │
                         │    (local, free)     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐    ┌─────────────────┐
                         │       Qdrant         │◄───│  PostgreSQL     │
                         │    (Vector DB)       │    │ (metadata store)│
                         └──────────┬───────────┘    └─────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Query API (FastAPI) │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         │                      │
                         ▼                      ▼
                 ┌──────────────┐     ┌──────────────────┐
                 │  Retriever   │     │   Reranker       │
                 │  (Qdrant)    │     │ (optional)       │
                 └──────┬───────┘     └────────┬─────────┘
                        │                      │
                        └──────────┬───────────┘
                                   ▼
                         ┌──────────────────────┐
                         │   LLM (Groq API)     │
                         │  Llama 3 70B (free)  │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  Response + Citations │
                         └──────────────────────┘

                         ┌──────────────────────────────────────┐
                         │       Observability Layer           │
                         │  Prometheus + Grafana               │
                         │  Metrics: retrieval/embedding latency│
                         │  Eval: LLM-as-judge context relevance│
                         └──────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Data Ingestion (Redpanda / Kafka)

| Source | Format | Method |
|---|---|---|
| Library catalog (DB) | JSON rows | CDC via Debezium → Redpanda |
| E-books / PDFs | PDF, EPUB | Upload webhook → Redpanda |
| Library website | HTML | Periodic crawl → Redpanda |
| Log peminjaman | CSV/JSON | Batch export → Redpanda |

### 2. Chunker

Dua strategi buat A/B comparison nanti:

- **RecursiveCharacterSplitter** — LangChain-style, split by `\n\n` → `\n` → `.`
- **SemanticChunker** — Split berdasarkan embedding similarity threshold

Output: `{chunk_id, doc_id, text, metadata, strategy}`

### 3. Embedder (ONNX MiniLM-L6-v2)

- Model: `all-MiniLM-L6-v2` via ONNX Runtime
- Dimension: 384
- Speed: ~250 doc/sec on CPU
- Gratis, no API call, no rate limit

### 4. Vector DB (Qdrant)

- Self-hosted via Docker
- Native `/metrics` endpoint buat Prometheus
- Collection structure:

```json
{
  "collection": "library_chunks",
  "vectors": {
    "size": 384,
    "distance": "Cosine"
  },
  "payload": ["doc_id", "source", "title", "author", "chunk_idx"]
}
```

### 5. Metadata Store (PostgreSQL)

| Table | Description |
|---|---|
| `documents` | doc_id, source, title, author, created_at, total_chunks |
| `chunks` | chunk_id, doc_id, chunk_idx, text, strategy |
| `queries` | query_id, query_text, retrieved_chunks, response |
| `feedback` | query_id, rating, corrected_response |

### 6. LLM (Groq API — gratis)

- Model: `llama3-70b-8192` (free tier: 30 req/min, 6000 req/day)
- Fallback: Ollama local (Llama 3.2 8B) kalo Groq rate-limited
- Prompt template:

```
You are a librarian assistant for {campus_name} library.
Answer based ONLY on the retrieved context below.

Context:
{retrieved_chunks}

Question: {query}

If the context doesn't contain the answer, say "I don't have that information."
Cite the document title for each claim.
```

### 7. Reranker (Opsional)

- Cohere Rerank API ($5 free credits cukup buat testing)
- Or skip entirely buat MVP — Qdrant + good embedding udah cukup

---

## Free APIs & Services

| Service | Component | Free Tier | Notes |
|---|---|---|---|
| **Groq** | LLM | 30 req/min, 6000 req/day | Llama 3 70B, Mixtral — register di groq.com |
| **Ollama** | LLM (local) | Unlimited | Dev fallback, no registration needed |
| **OpenRouter** | LLM fallback | Varies per model | Register, some models free |
| **Cohere** | Reranker | $5 trial credits | Optional, activate di cohere.com |
| **ONNX (local)** | Embedding | Unlimited | `all-MiniLM-L6-v2` — 80MB, no API call |
| **Qdrant (self-hosted)** | Vector DB | Unlimited | Docker container |
| **Redpanda** | Streaming | Unlimited | Docker container |
| **PostgreSQL** | Metadata | Unlimited | Docker container |
| **Prometheus + Grafana** | Monitoring | Unlimited | Docker containers |

### API keys yang perlu didaftarin:

1. **Groq** → `GROQ_API_KEY`
2. **Cohere** → `COHERE_API_KEY` (optional)
3. **OpenRouter** → `OPENROUTER_API_KEY` (optional, fallback)

---

## AWS Strategy

| Layer | AWS Service | Kapan |
|---|---|---|
| **Container orchestration** | ECS Fargate | Saat deploy |
| **Streaming** | MSK (managed Kafka) | Scale > Redpanda self-hosted |
| **Vector DB** | Qdrant Cloud (multi-cloud) or self-hosted di ECS | Kapan aja |
| **LLM** | Bedrock (Claude) or tetap Groq | Kalo mau managed |
| **Metadata** | RDS PostgreSQL | Scale / prod |
| **Object storage** | S3 | Dokumen mentah |
| **CI/CD** | Keep GitHub Actions | Udah jalan |

Migration path: semua pake Docker → deploy ke ECS Fargate. Same images, same configs.

---

## Folder Structure

```
library-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entry point
│   ├── dependencies.py          # DI, rate limiter, auth
│   ├── components/
│   │   ├── chunker.py           # Recursive + semantic chunkers
│   │   ├── embedder.py          # ONNX MiniLM wrapper
│   │   ├── retriever.py         # Qdrant search
│   │   ├── generator.py         # Groq / Ollama LLM call
│   │   ├── reranker.py          # Cohere rerank wrapper
│   │   └── metrics.py           # Prometheus metrics
│   ├── pipeline/
│   │   ├── ingestion.py         # Kafka consumer → chunk → embed → store
│   │   └── query.py             # Query → retrieve → rerank → generate
│   ├── router/
│   │   ├── query.py             # POST /query
│   │   ├── ingest.py            # POST /ingest (webhook)
│   │   ├── documents.py         # GET /documents, DELETE /documents/{id}
│   │   └── health.py            # GET /health, /ready
│   ├── schema/
│   │   ├── query.py             # QueryRequest, QueryResponse
│   │   ├── ingest.py            # IngestRequest, IngestResponse
│   │   └── feedback.py          # FeedbackRequest
│   └── monitoring/
│       ├── drift.py             # Query drift detection
│       └── evaluator.py         # LLM-as-judge quality eval
├── configs/
│   ├── serve_config.yaml        # App config
│   └── schema.sql               # PostgreSQL DDL
├── scripts/
│   ├── seed_data.py             # Seed library catalog sample data
│   ├── eval_dataset.json        # Test queries for CI eval gate
│   └── compute_baseline.py      # Query distribution baseline
├── tests/
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   └── test_query_pipeline.py
├── docker-compose.yml           # Redpanda, Qdrant, PostgreSQL, Prometheus, Grafana
├── Dockerfile                   # Multi-stage uv build (copy from rescuevision)
├── pyproject.toml
├── Makefile
└── monitoring/
    ├── prometheus.yml
    └── alerts.yml
```

---

## Docker Compose Services

```yaml
services:
  redpanda:
    image: redpandadata/redpanda:v24.x

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]

  postgres:
    image: postgres:16-alpine

  app:
    build: .
    ports: ["8080:8080"]
    depends_on: [redpanda, qdrant, postgres]

  prometheus:
    image: prom/prometheus:v2.51.0
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:10.4.0
    ports: ["3000:3000"]
```

---

## API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/query` | POST | API Key | Ask a question, get answer + citations |
| `/ingest` | POST | API Key | Upload document (PDF/txt) → stream ke Redpanda |
| `/documents` | GET | API Key | List ingested documents |
| `/documents/{id}` | DELETE | API Key | Remove document + its chunks |
| `/feedback` | POST | API Key | Rate a query response |
| `/health` | GET | — | Liveness |
| `/ready` | GET | — | Readiness |
| `/metrics` | GET | — | Prometheus scrape |

---

## Implementation Phases

### Fase 1 — Foundation (2-3 hari)

- [ ] Init project: `pyproject.toml`, `Makefile`, `docker-compose.yml`, `Dockerfile`
- [ ] Chunker service (recursive split, semantic split)
- [ ] Embedder service (ONNX MiniLM-L6-v2)
- [ ] Qdrant integration (upsert, search)
- [ ] Ingestion pipeline: POST /ingest → Redpanda → chunk → embed → Qdrant
- [ ] Seed data: library catalog sample (buku, skripsi, jurnal)
- [ ] Test: `make test`

### Fase 2 — Serving (1-2 hari)

- [ ] FastAPI app with router structure
- [ ] POST /query — retrieve → generate
- [ ] Groq API integration (free tier)
- [ ] Rate limiting + API key auth (copy pattern dari rescuevision)
- [ ] Response with citations + document titles
- [ ] Test: smoke test dengan sample queries

### Fase 3 — Quality (1-2 hari)

- [ ] LLM-as-judge evaluation (via Groq juga)
- [ ] Reranker integration (Cohere or skip)
- [ ] A/B chunking strategy comparison
- [ ] Prometheus metrics for all components
- [ ] Grafana dashboard
- [ ] Eval CI gate in GitHub Actions

### Fase 4 — Production Polish (opsional)

- [ ] Query drift monitoring
- [ ] Hallucination detection
- [ ] Feedback collection → continuous improvement
- [ ] AWS deployment (ECS Fargate + RDS + MSK)
- [ ] RAGAS benchmark suite

---

## Quick Start

```bash
# Prerequisites: Python 3.10+, Docker, uv
# Register: groq.com → can't api key

git clone <repo-url>
cd library-rag
cp .env.example .env  # isi GROQ_API_KEY

# Install + run
make install-dev
docker compose up -d         # infra: redpanda, qdrant, postgres, prometheus, grafana
uvicorn app.main:app --reload

# Seed library data
uv run python scripts/seed_data.py

# Try it
curl -X POST http://localhost:8080/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "buku pemrograman python apa aja yang ada?"}'
```

---

## Key Decisions (Ponytail Rationale)

| Decision | Why |
|---|---|
| Redpanda over Kafka | 1 Docker service, no ZK/JVM, Kafka-compatible, built-in metrics |
| Qdrant over Milvus/Weaviate | Rust, native `/metrics`, 1 container, managed cloud same API |
| ONNX MiniLM over API embedding | Gratis, no rate limit, 250 doc/s on CPU, 80MB model |
| Groq over OpenAI | Gratis, Llama 3 70B, fast inference |
| Skip reranker for MVP | Qdrant HNSW + good embedding already decent, add later if needed |
| PostgreSQL separate from Qdrant | Vector DB only stores vectors, relational data stays in RDBMS |

---

## Prerequisites

- [ ] Python >= 3.10
- [ ] Docker + Docker Compose
- [ ] [uv](https://docs.astral.sh/uv/)
- [ ] Groq API key ([groq.com](https://groq.com))
- [ ] Cohere API key (optional, [cohere.com](https://cohere.com))

---

Selamat membangun! 🚀
