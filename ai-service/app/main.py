"""
ai-service
Owns: embeddings, retrieval orchestration, re-ranking, LLM prompting/generation.
Nothing outside this service calls the Gemini API directly.
"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
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


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(400, "texts must not be empty")

    client = get_client()
    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=req.texts,
        config=EmbedContentConfig(
            task_type=req.task_type,
            output_dimensionality=GEMINI_EMBEDDING_DIM,
        ),
    )
    vectors = [e.values for e in response.embeddings]
    return EmbedResponse(embeddings=vectors)


# TODO next:
# POST /answer          -> full RAG flow: retrieve (via data-service) -> rerank -> generate
# WS   /answer/stream     -> same, but streams tokens back (used by gateway)
