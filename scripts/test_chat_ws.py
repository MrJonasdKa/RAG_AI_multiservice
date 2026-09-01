"""
Quick manual test for the gateway's WebSocket /chat endpoint.
Connects like a real client would, sends one question, prints the
streamed tokens as they arrive, then the final sources.

Run: python scripts/test_chat_ws.py "your question here"
"""

import asyncio
import json
import sys

import websockets

GATEWAY_WS_URL = "ws://localhost:8000/chat"


async def main(question: str):
    async with websockets.connect(GATEWAY_WS_URL) as ws:
        await ws.send(json.dumps({"question": question, "top_k": 5}))

        print("Answer: ", end="", flush=True)
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg["type"] == "token":
                print(msg["text"], end="", flush=True)
            elif msg["type"] == "done":
                print("\n\nSources:")
                for s in msg["sources"]:
                    print(f"  - {s['document_title']} (distance={s['distance']:.3f})")
                break
            elif msg["type"] == "error":
                print(f"\n[error] {msg['message']}")
                break


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how does an index speed up a query?"
    asyncio.run(main(question))
