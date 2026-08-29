"""
gateway
Owns: auth, public-facing REST + WebSocket endpoints, routing to ai-service/data-service.
This is the only service exposed to clients.
"""

from fastapi import FastAPI

app = FastAPI(title="rag-gateway")


@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


# TODO: endpoints to come, e.g.
# POST /auth/login
# WS   /chat                  -> client connects here, gateway proxies to ai-service stream
# GET  /documents              -> proxied to data-service (admin/ingestion UI)
