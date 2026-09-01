"""
data-service
Owns: documents, chunks, embeddings, conversation history, feedback.
Nothing outside this service touches Postgres directly.

Embedding calls are delegated to ai-service (POST /embed) — this service
never calls Gemini directly, keeping that boundary in one place.
"""

import os
import uuid
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
    score: float  # Reciprocal Rank Fusion score — higher is more relevant


RRF_K = 60  # standard RRF damping constant; de-emphasizes low ranks without needing score normalization


@app.get("/chunks/search", response_model=list[SearchResult])
async def search_chunks(query: str, top_k: int = 5, candidates: int = 20):
    """
    Hybrid retrieval: runs vector similarity search and PostgreSQL full-text
    (keyword) search independently, then merges the two ranked lists with
    Reciprocal Rank Fusion. RRF works on rank position rather than raw
    scores, which sidesteps the problem that cosine distance and text-rank
    scores live on completely different, non-comparable scales.
    """
    query_embedding = (await _embed([query], task_type="RETRIEVAL_QUERY"))[0]

    pool = get_pool()
    async with pool.acquire() as conn:
        vector_rows = await conn.fetch(
            """
            SELECT c.id AS chunk_id, c.document_id, d.title AS document_title, c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> $1
            LIMIT $2
            """,
            query_embedding,
            candidates,
        )

        keyword_rows = await conn.fetch(
            """
            SELECT c.id AS chunk_id, c.document_id, d.title AS document_title, c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.search_vector @@ plainto_tsquery('english', $1)
            ORDER BY ts_rank_cd(c.search_vector, plainto_tsquery('english', $1)) DESC
            LIMIT $2
            """,
            query,
            candidates,
        )

    rrf_scores: dict[str, float] = {}
    chunk_info: dict[str, object] = {}

    for rank, row in enumerate(vector_rows, start=1):
        cid = str(row["chunk_id"])
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunk_info[cid] = row

    for rank, row in enumerate(keyword_rows, start=1):
        cid = str(row["chunk_id"])
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunk_info.setdefault(cid, row)

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    return [
        SearchResult(
            chunk_id=cid,
            document_id=str(chunk_info[cid]["document_id"]),
            document_title=chunk_info[cid]["document_title"],
            content=chunk_info[cid]["content"],
            score=rrf_scores[cid],
        )
        for cid in ranked_ids
    ]


# ---------- conversations & messages ----------

class ConversationResponse(BaseModel):
    conversation_id: str


@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation():
    pool = get_pool()
    async with pool.acquire() as conn:
        conv_id = await conn.fetchval(
            "INSERT INTO conversations DEFAULT VALUES RETURNING id"
        )
    return ConversationResponse(conversation_id=str(conv_id))


class MessageRequest(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str


@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(conversation_id: str, req: MessageRequest):
    if req.role not in ("user", "assistant"):
        raise HTTPException(400, "role must be 'user' or 'assistant'")

    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(400, "invalid conversation_id")

    pool = get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT 1 FROM conversations WHERE id = $1", conv_uuid
        )
        if not exists:
            raise HTTPException(404, "conversation not found")

        msg_id = await conn.fetchval(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES ($1, $2, $3) RETURNING id
            """,
            conv_uuid,
            req.role,
            req.content,
        )
    return MessageResponse(message_id=str(msg_id), role=req.role, content=req.content)


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(conversation_id: str):
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(400, "invalid conversation_id")

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, role, content FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            """,
            conv_uuid,
        )
    return [
        MessageResponse(message_id=str(r["id"]), role=r["role"], content=r["content"])
        for r in rows
    ]


# ---------- feedback ----------

class FeedbackRequest(BaseModel):
    rating: int  # 1 (thumbs up) or -1 (thumbs down)


class FeedbackResponse(BaseModel):
    feedback_id: str
    message_id: str
    rating: int


@app.post("/messages/{message_id}/feedback", response_model=FeedbackResponse)
async def add_feedback(message_id: str, req: FeedbackRequest):
    if req.rating not in (-1, 1):
        raise HTTPException(400, "rating must be 1 or -1")

    try:
        msg_uuid = uuid.UUID(message_id)
    except ValueError:
        raise HTTPException(400, "invalid message_id")

    pool = get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM messages WHERE id = $1", msg_uuid)
        if not exists:
            raise HTTPException(404, "message not found")

        feedback_id = await conn.fetchval(
            "INSERT INTO feedback (message_id, rating) VALUES ($1, $2) RETURNING id",
            msg_uuid,
            req.rating,
        )

    return FeedbackResponse(feedback_id=str(feedback_id), message_id=message_id, rating=req.rating)


class FeedbackStats(BaseModel):
    message_id: str
    up: int
    down: int


@app.get("/messages/{message_id}/feedback", response_model=FeedbackStats)
async def get_feedback(message_id: str):
    try:
        msg_uuid = uuid.UUID(message_id)
    except ValueError:
        raise HTTPException(400, "invalid message_id")

    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE rating = 1) AS up,
                COUNT(*) FILTER (WHERE rating = -1) AS down
            FROM feedback
            WHERE message_id = $1
            """,
            msg_uuid,
        )

    return FeedbackStats(message_id=message_id, up=row["up"], down=row["down"])
