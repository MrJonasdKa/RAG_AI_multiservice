"""
data-service
Owns: documents, chunks, embeddings, conversation history, feedback.
Nothing outside this service touches Postgres directly.
"""

from fastapi import FastAPI

app = FastAPI(title="rag-data-service")


@app.get("/health")
def health():
    return {"status": "ok", "service": "data-service"}


# TODO: endpoints to come, e.g.
# POST /documents            -> ingest a new document (chunk + store)
# GET  /chunks/search         -> vector similarity search (+ keyword/BM25 hybrid later)
# POST /conversations/{id}/messages
# POST /feedback
