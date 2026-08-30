"""
ai-service
Owns: embeddings, retrieval orchestration, re-ranking, LLM prompting/generation.
Nothing outside this service calls the Gemini API directly.
"""

import os
import re
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import EmbedContentConfig

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_DIM = int(os.environ.get("GEMINI_EMBEDDING_DIM", "768"))

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


# TODO next:
# POST /answer          -> full RAG flow: retrieve (via data-service) -> rerank -> generate
# WS   /answer/stream     -> same, but streams tokens back (used by gateway)
