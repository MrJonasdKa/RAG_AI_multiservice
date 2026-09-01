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

app = FastAPI(title="rag-ai-service")

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


class SourceChunk(BaseModel):
    document_title: str
    content: str
    distance: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the "
    "provided documentation excerpts below. If the excerpts don't contain "
    "the answer, say so honestly instead of guessing or using outside "
    "knowledge. Answer clearly and concisely."
)


def _retrieve_chunks(question: str, top_k: int) -> list[dict]:
    with httpx.Client(timeout=30) as http_client:
        resp = http_client.get(
            f"{DATA_SERVICE_URL}/chunks/search",
            params={"query": question, "top_k": top_k},
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"data-service /chunks/search failed: {resp.text}")
    return resp.json()


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['document_title']}]\n{c['content']}" for c in chunks
    )
    return (
        f"--- Documentation excerpts ---\n{context}\n\n"
        f"--- Question ---\n{question}"
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    chunks = _retrieve_chunks(req.question, req.top_k)
    if not chunks:
        raise HTTPException(404, "no relevant chunks found in the knowledge base")

    client = get_client()
    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=_build_prompt(req.question, chunks),
        config=GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            thinking_config=ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL),
        ),
    )

    sources = [
        SourceChunk(
            document_title=c["document_title"],
            content=c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
            distance=c["distance"],
        )
        for c in chunks
    ]

    return AnswerResponse(answer=response.text, sources=sources)


# ---------- answer streaming (WebSocket) ----------

async def _retrieve_chunks_async(question: str, top_k: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as http_client:
        resp = await http_client.get(
            f"{DATA_SERVICE_URL}/chunks/search",
            params={"query": question, "top_k": top_k},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"data-service /chunks/search failed: {resp.text}")
    return resp.json()


def _sources_payload(chunks: list[dict]) -> list[dict]:
    return [
        {
            "document_title": c["document_title"],
            "content": c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
            "distance": c["distance"],
        }
        for c in chunks
    ]


@app.websocket("/answer/stream")
async def answer_stream(websocket: WebSocket):
    """
    Protocol: client sends {"question": "...", "top_k": 5} as JSON.
    Server sends a sequence of JSON messages back:
      {"type": "token", "text": "..."}   - zero or more, as the answer streams in
      {"type": "done", "sources": [...]} - once generation finishes
      {"type": "error", "message": "..."} - on any failure for that question
    The connection stays open for follow-up questions until the client disconnects.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            question = (data.get("question") or "").strip()
            top_k = data.get("top_k", 5)

            if not question:
                await websocket.send_json({"type": "error", "message": "question must not be empty"})
                continue

            try:
                chunks = await _retrieve_chunks_async(question, top_k)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            if not chunks:
                await websocket.send_json(
                    {"type": "error", "message": "no relevant chunks found in the knowledge base"}
                )
                continue

            client = get_client()
            prompt = _build_prompt(question, chunks)

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
                        await websocket.send_json({"type": "token", "text": chunk.text})
            except genai_errors.ClientError as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            await websocket.send_json({"type": "done", "sources": _sources_payload(chunks)})
    except WebSocketDisconnect:
        pass
