# RAG Backend Tool — Project 2

Full-split microservices RAG system. LLM answers questions grounded in a
document knowledge base, currently the PostgreSQL official docs.

## Architecture

- **gateway** — auth, public REST + WebSocket entrypoint, the only service
  exposed to clients. Streams answers token-by-token over WebSockets.
- **ai-service** — embeddings, retrieval, re-ranking, LLM orchestration.
  Only this service calls the Gemini API.
- **data-service** — owns Postgres: documents, chunks, embeddings (pgvector),
  conversation history, feedback. Nothing else touches the DB directly.
- **db** — Postgres with the `pgvector` extension.

Each service is independently deployable and only talks to the others over
HTTP (internal network via Docker Compose).

## Getting started

1. Copy `.env.example` -> `.env` in each service folder (already done for
   local dev — just fill in `ai-service/.env` with a real `GEMINI_API_KEY`
   from https://aistudio.google.com/).
2. `docker compose up --build`
3. Health checks:
   - http://localhost:8000/health (gateway)
   - http://localhost:8001/health (data-service, direct — not exposed in prod)
   - http://localhost:8002/health (ai-service, direct — not exposed in prod)

## Ingesting the knowledge base

```
python scripts/fetch_postgres_docs.py   # pulls + cleans a starter set of PG docs into docs_raw/
python scripts/ingest_docs.py           # chunks + embeds + stores each file via data-service
```

Then try a search directly against data-service:
```
curl "http://localhost:8001/chunks/search?query=how+does+an+index+speed+up+a+query"
```

## Running tests

```
docker compose exec data-service pytest   # needs the DB (health test) — chunking tests run anywhere
docker compose exec ai-service pytest
docker compose exec gateway pytest
```

## Roadmap for this project (see /docs or main roadmap for full detail)

- [x] Ingestion pipeline (chunking + metadata tagging)
- [x] Embedding + storage via data-service
- [x] Basic vector search endpoint (`GET /chunks/search`)
- [ ] Hybrid retrieval (vector + BM25)
- [ ] Re-ranking step
- [ ] Source citations in answers
- [ ] WebSocket streaming end-to-end (gateway -> ai-service -> client)
- [ ] Conversation memory
- [ ] Feedback loop (thumbs up/down)
- [ ] Access control per document set
- [ ] Caching for repeated queries
