"""
Diagnostic version: timestamps every message received so we can see
whether tokens are actually arriving progressively over time, or all
at once in a burst.

Run: python scripts/test_chat_ws_debug.py "your question here"
"""

import asyncio
import json
import sys
import time

import websockets

GATEWAY_WS_URL = "ws://localhost:8000/chat"


async def main(question: str):
    async with websockets.connect(GATEWAY_WS_URL) as ws:
        await ws.send(json.dumps({"question": question, "top_k": 5}))

        start = time.monotonic()
        token_count = 0

        while True:
            raw = await ws.recv()
            elapsed = time.monotonic() - start
            msg = json.loads(raw)

            if msg["type"] == "token":
                token_count += 1
                print(f"[{elapsed:6.2f}s] token #{token_count:3d} ({len(msg['text']):3d} chars): {msg['text']!r}")
            elif msg["type"] == "done":
                print(f"[{elapsed:6.2f}s] DONE — received {token_count} token message(s) total")
                break
            elif msg["type"] == "error":
                print(f"[{elapsed:6.2f}s] ERROR: {msg['message']}")
                break


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "explain b-tree vs hash indexes in detail"
    asyncio.run(main(question))
