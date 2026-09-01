"""
gateway
Owns: auth, public-facing REST + WebSocket endpoints, routing to ai-service/data-service.
This is the only service exposed to clients.
"""

import asyncio
import os

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

AI_SERVICE_WS_URL = os.environ.get("AI_SERVICE_WS_URL", "ws://ai-service:8000/answer/stream")

app = FastAPI(title="rag-gateway")


@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


@app.websocket("/chat")
async def chat(websocket: WebSocket):
    """
    Client-facing WebSocket. Proxies messages to/from ai-service's
    /answer/stream, so clients never talk to ai-service directly.
    Same JSON protocol as ai-service/answer/stream — see that endpoint's
    docstring.
    """
    await websocket.accept()

    try:
        async with websockets.connect(AI_SERVICE_WS_URL) as ai_ws:

            async def client_to_ai():
                while True:
                    message = await websocket.receive_text()
                    await ai_ws.send(message)

            async def ai_to_client():
                async for message in ai_ws:
                    await websocket.send_text(message)

            client_task = asyncio.create_task(client_to_ai())
            ai_task = asyncio.create_task(ai_to_client())

            done, pending = await asyncio.wait(
                {client_task, ai_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.exception()  # surface (and swallow) any exception, don't crash the server

    except (WebSocketDisconnect, ConnectionClosed):
        pass


# TODO: endpoints to come, e.g.
# POST /auth/login
# GET  /documents -> proxied to data-service (admin/ingestion UI)
