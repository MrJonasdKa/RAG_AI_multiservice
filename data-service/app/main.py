"""
data-service
Owns: documents, chunks, embeddings, conversation history, feedback.
Nothing outside this service touches Postgres directly.

Embedding calls are delegated to ai-service (POST /embed) — this service
never calls Gemini directly, keeping that boundary in one place.
"""

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.chunking import chunk_text
from app.db import close_pool, get_pool, init_pool

AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://ai-service:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="rag-data-service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "service": "data-service"}


# ---------- ingestion ----------

class IngestRequest(BaseModel):
    title: str
    source: str | None = None
    content: str


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int


async def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{AI_SERVICE_URL}/embed",
            json={"texts": texts, "task_type": task_type},
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"ai-service /embed failed: {resp.text}")
    return resp.json()["embeddings"]


@app.post("/documents", response_model=IngestResponse)
async def ingest_document(req: IngestRequest):
    chunks = chunk_text(req.content)
    if not chunks:
        raise HTTPException(400, "content produced no chunks")

    embeddings = await _embed(chunks, task_type="RETRIEVAL_DOCUMENT")
    if len(embeddings) != len(chunks):
        raise HTTPException(502, "ai-service returned a mismatched embedding count")

    pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        doc_id = await conn.fetchval(
            "INSERT INTO documents (title, source) VALUES ($1, $2) RETURNING id",
            req.title,
            req.source,
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await conn.execute(
                """
                INSERT INTO chunks (document_id, content, embedding, chunk_index)
                VALUES ($1, $2, $3, $4)
                """,
                doc_id,
                chunk,
                embedding,
                idx,
            )

    return IngestResponse(document_id=str(doc_id), chunk_count=len(chunks))


# ---------- retrieval ----------

class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    content: str
    distance: float


@app.get("/chunks/search", response_model=list[SearchResult])
async def search_chunks(query: str, top_k: int = 5):
    query_embedding = (await _embed([query], task_type="RETRIEVAL_QUERY"))[0]

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id AS chunk_id, c.document_id, d.title AS document_title,
                   c.content, c.embedding <=> $1 AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> $1
            LIMIT $2
            """,
            query_embedding,
            top_k,
        )

    return [
        SearchResult(
            chunk_id=str(r["chunk_id"]),
            document_id=str(r["document_id"]),
            document_title=r["document_title"],
            content=r["content"],
            distance=r["distance"],
        )
        for r in rows
    ]
