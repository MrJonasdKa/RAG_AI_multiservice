"""
ai-service
Owns: embeddings, retrieval orchestration, re-ranking, LLM prompting/generation.
Nothing outside this service calls the Gemini API directly.
Talks to data-service (HTTP) for chunk storage/retrieval.
"""

from fastapi import FastAPI

app = FastAPI(title="rag-ai-service")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-service"}


# TODO: endpoints to come, e.g.
# POST /embed                -> embed text (calls Gemini gemini-embedding-001)
# POST /answer                -> full RAG flow: retrieve -> rerank -> generate
# WS   /answer/stream          -> same, but streams tokens back (used by gateway)
