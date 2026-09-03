"""
ai-service
Owns: embeddings, retrieval orchestration, re-ranking, LLM prompting/generation.
Nothing outside this service calls the Gemini API directly.
"""

import os
import re
import time

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import EmbedContentConfig, GenerateContentConfig, ThinkingConfig

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_DIM = int(os.environ.get("GEMINI_EMBEDDING_DIM", "768"))
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
GEMINI_THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "LOW")
DATA_SERVICE_URL = os.environ.get("DATA_SERVICE_URL", "http://data-service:8000")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
CACHE_MAX_ENTRIES = 200

app = FastAPI(title="rag-ai-service")

# In-memory answer cache, keyed by (normalized question, top_k). Only used
# for fresh questions (no conversation_id) — a cached answer is never reused
# for a follow-up, since the same text can mean something different mid-
# conversation. Single-process only: fine for this project's scale, but a
# multi-worker or multi-instance deployment would need Redis instead.
_answer_cache: dict[tuple[str, int, str | None], dict] = {}


def _cache_get(key: tuple[str, int, str | None]):
    entry = _answer_cache.get(key)
    if entry is None:
        return None
    if time.monotonic() > entry["expires_at"]:
        _answer_cache.pop(key, None)
        return None
    return entry


def _cache_set(key: tuple[str, int, str | None], chunks: list[dict], answer_text: str) -> None:
    if len(_answer_cache) >= CACHE_MAX_ENTRIES:
        oldest_key = min(_answer_cache, key=lambda k: _answer_cache[k]["expires_at"])
        _answer_cache.pop(oldest_key, None)
    _answer_cache[key] = {
        "chunks": chunks,
        "answer_text": answer_text,
        "expires_at": time.monotonic() + CACHE_TTL_SECONDS,
    }

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-key-here":
            raise HTTPException(500, "GEMINI_API_KEY not configured")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-service"}


# ---------- embeddings ----------

class EmbedRequest(BaseModel):
    texts: list[str]
    task_type: str = "RETRIEVAL_DOCUMENT"  # or "RETRIEVAL_QUERY"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


GEMINI_MAX_BATCH = 100  # Gemini's embed_content batch limit
MAX_RETRIES = 5
RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def _embed_batch_with_retry(client: genai.Client, batch: list[str], task_type: str):
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch,
                config=EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=GEMINI_EMBEDDING_DIM,
                ),
            )
        except genai_errors.ClientError as e:
            is_rate_limit = getattr(e, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            if not is_rate_limit or attempt == MAX_RETRIES - 1:
                raise
            match = RETRY_DELAY_RE.search(str(e))
            delay = float(match.group(1)) + 1 if match else (2 ** attempt) * 5
            time.sleep(delay)

    raise RuntimeError("unreachable")  # loop always returns or raises


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(400, "texts must not be empty")

    client = get_client()
    vectors: list[list[float]] = []

    for i in range(0, len(req.texts), GEMINI_MAX_BATCH):
        batch = req.texts[i : i + GEMINI_MAX_BATCH]
        response = _embed_batch_with_retry(client, batch, req.task_type)
        vectors.extend(e.values for e in response.embeddings)

    return EmbedResponse(embeddings=vectors)


# ---------- answer (retrieve + generate) ----------

class AnswerRequest(BaseModel):
    question: str
    top_k: int = 5
    conversation_id: str | None = None
    access_group: str | None = None  # scopes retrieval to public + this group's documents


class SourceChunk(BaseModel):
    document_title: str
    content: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    conversation_id: str
    cached: bool = False


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the "
    "provided documentation excerpts below. If the excerpts don't contain "
    "the answer, say so honestly instead of guessing or using outside "
    "knowledge. Answer clearly and concisely. Use the prior conversation "
    "(if any) to understand follow-up questions in context."
)


def _retrieve_chunks(question: str, top_k: int, access_group: str | None = None) -> list[dict]:
    params = {"query": question, "top_k": top_k}
    if access_group:
        params["access_group"] = access_group
    with httpx.Client(timeout=30) as http_client:
        resp = http_client.get(f"{DATA_SERVICE_URL}/chunks/search", params=params)
    if resp.status_code != 200:
        raise HTTPException(502, f"data-service /chunks/search failed: {resp.text}")
    return resp.json()


def _create_conversation() -> str:
    with httpx.Client(timeout=15) as http_client:
        resp = http_client.post(f"{DATA_SERVICE_URL}/conversations")
    if resp.status_code != 200:
        raise HTTPException(502, f"data-service /conversations failed: {resp.text}")
    return resp.json()["conversation_id"]


def _get_history(conversation_id: str) -> list[dict]:
    with httpx.Client(timeout=15) as http_client:
        resp = http_client.get(f"{DATA_SERVICE_URL}/conversations/{conversation_id}/messages")
    if resp.status_code != 200:
        raise HTTPException(502, f"data-service get messages failed: {resp.text}")
    return resp.json()


def _save_message(conversation_id: str, role: str, content: str) -> None:
    with httpx.Client(timeout=15) as http_client:
        resp = http_client.post(
            f"{DATA_SERVICE_URL}/conversations/{conversation_id}/messages",
            json={"role": role, "content": content},
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"data-service save message failed: {resp.text}")


def _build_prompt(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    context = "\n\n".join(
        f"[Source: {c['document_title']}]\n{c['content']}" for c in chunks
    )
    history_block = ""
    if history:
        turns = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
        history_block = f"--- Prior conversation ---\n{turns}\n\n"
    return (
        f"{history_block}"
        f"--- Documentation excerpts ---\n{context}\n\n"
        f"--- Question ---\n{question}"
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    cache_key = (req.question.strip().lower(), req.top_k, req.access_group)
    cached = _cache_get(cache_key) if not req.conversation_id else None

    if cached:
        chunks = cached["chunks"]
        answer_text = cached["answer_text"]
    else:
        chunks = _retrieve_chunks(req.question, req.top_k, req.access_group)
        if not chunks:
            raise HTTPException(404, "no relevant chunks found in the knowledge base")

        history = _get_history(req.conversation_id) if req.conversation_id else []

        client = get_client()
        response = client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=_build_prompt(req.question, chunks, history),
            config=GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                thinking_config=ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
            ),
        )
        answer_text = response.text

        if not req.conversation_id:
            _cache_set(cache_key, chunks, answer_text)

    conversation_id = req.conversation_id or _create_conversation()
    _save_message(conversation_id, "user", req.question)
    _save_message(conversation_id, "assistant", answer_text)

    sources = [
        SourceChunk(
            document_title=c["document_title"],
            content=c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
            score=c["score"],
        )
        for c in chunks
    ]

    return AnswerResponse(
        answer=answer_text,
        sources=sources,
        conversation_id=conversation_id,
        cached=cached is not None,
    )


# ---------- answer streaming (WebSocket) ----------

async def _retrieve_chunks_async(question: str, top_k: int, access_group: str | None = None) -> list[dict]:
    params = {"query": question, "top_k": top_k}
    if access_group:
        params["access_group"] = access_group
    async with httpx.AsyncClient(timeout=30) as http_client:
        resp = await http_client.get(f"{DATA_SERVICE_URL}/chunks/search", params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"data-service /chunks/search failed: {resp.text}")
    return resp.json()


async def _create_conversation_async() -> str:
    async with httpx.AsyncClient(timeout=15) as http_client:
        resp = await http_client.post(f"{DATA_SERVICE_URL}/conversations")
    if resp.status_code != 200:
        raise RuntimeError(f"data-service /conversations failed: {resp.text}")
    return resp.json()["conversation_id"]


async def _get_history_async(conversation_id: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as http_client:
        resp = await http_client.get(f"{DATA_SERVICE_URL}/conversations/{conversation_id}/messages")
    if resp.status_code != 200:
        raise RuntimeError(f"data-service get messages failed: {resp.text}")
    return resp.json()


async def _save_message_async(conversation_id: str, role: str, content: str) -> None:
    async with httpx.AsyncClient(timeout=15) as http_client:
        resp = await http_client.post(
            f"{DATA_SERVICE_URL}/conversations/{conversation_id}/messages",
            json={"role": role, "content": content},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"data-service save message failed: {resp.text}")


def _sources_payload(chunks: list[dict]) -> list[dict]:
    return [
        {
            "document_title": c["document_title"],
            "content": c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
            "score": c["score"],
        }
        for c in chunks
    ]


@app.websocket("/answer/stream")
async def answer_stream(websocket: WebSocket):
    """
    Protocol: client sends {"question": "...", "top_k": 5, "conversation_id": "..."}
    as JSON. conversation_id is optional — omit it to start a new conversation.
    Server sends a sequence of JSON messages back:
      {"type": "conversation", "conversation_id": "..."} - sent once, first
      {"type": "token", "text": "..."}   - zero or more, as the answer streams in
      {"type": "done", "sources": [...], "cached": false} - once generation finishes
      {"type": "error", "message": "..."} - on any failure for that question
    The connection stays open for follow-up questions until the client disconnects.
    Pass the same conversation_id back on later questions to keep context.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            question = (data.get("question") or "").strip()
            top_k = data.get("top_k", 5)
            requested_conversation_id = data.get("conversation_id")
            access_group = data.get("access_group")

            if not question:
                await websocket.send_json({"type": "error", "message": "question must not be empty"})
                continue

            try:
                chunks = await _retrieve_chunks_async(question, top_k, access_group)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            if not chunks:
                await websocket.send_json(
                    {"type": "error", "message": "no relevant chunks found in the knowledge base"}
                )
                continue

            try:
                if requested_conversation_id:
                    conversation_id = requested_conversation_id
                    history = await _get_history_async(conversation_id)
                else:
                    conversation_id = await _create_conversation_async()
                    history = []
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            await websocket.send_json({"type": "conversation", "conversation_id": conversation_id})

            cache_key = (question.strip().lower(), top_k, access_group)
            cached = _cache_get(cache_key) if not requested_conversation_id else None

            if cached:
                answer_text = cached["answer_text"]
                await websocket.send_json({"type": "token", "text": answer_text})
            else:
                client = get_client()
                prompt = _build_prompt(question, chunks, history)
                answer_text_parts: list[str] = []

                try:
                    stream = await client.aio.models.generate_content_stream(
                        model=GEMINI_CHAT_MODEL,
                        contents=prompt,
                        config=GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            thinking_config=ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
                        ),
                    )
                    async for chunk in stream:
                        if chunk.text:
                            answer_text_parts.append(chunk.text)
                            await websocket.send_json({"type": "token", "text": chunk.text})
                except genai_errors.ClientError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                answer_text = "".join(answer_text_parts)
                if not requested_conversation_id:
                    _cache_set(cache_key, chunks, answer_text)

            try:
                await _save_message_async(conversation_id, "user", question)
                await _save_message_async(conversation_id, "assistant", answer_text)
            except Exception:
                pass  # answer already delivered to the client; don't fail the turn over history logging

            await websocket.send_json(
                {"type": "done", "sources": _sources_payload(chunks), "cached": cached is not None}
            )
    except WebSocketDisconnect:
        pass
