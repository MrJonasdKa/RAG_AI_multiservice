"""
Proves conversation memory actually works: asks a first question, then a
DELIBERATELY VAGUE follow-up that only makes sense if the model remembers
the first answer. If the second answer correctly identifies what "it" or
"that" refers to without you spelling it out again, memory is working.

Run: python scripts/test_conversation_memory.py
"""

import asyncio
import json

import websockets

GATEWAY_WS_URL = "ws://localhost:8000/chat"


async def ask(ws, question: str, conversation_id: str | None):
    payload = {"question": question, "top_k": 5}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    await ws.send(json.dumps(payload))

    conv_id = None
    answer_parts = []

    while True:
        raw = await ws.recv()
        msg = json.loads(raw)

        if msg["type"] == "conversation":
            conv_id = msg["conversation_id"]
        elif msg["type"] == "token":
            answer_parts.append(msg["text"])
        elif msg["type"] == "done":
            break
        elif msg["type"] == "error":
            print(f"[error] {msg['message']}")
            break

    return conv_id, "".join(answer_parts)


async def main():
    async with websockets.connect(GATEWAY_WS_URL) as ws:
        print("--- Turn 1 ---")
        q1 = "what is a hash index in postgresql?"
        print(f"Q: {q1}")
        conv_id, a1 = await ask(ws, q1, conversation_id=None)
        print(f"A: {a1}\n")
        print(f"conversation_id: {conv_id}\n")

        print("--- Turn 2 (deliberately vague — no mention of 'hash index') ---")
        q2 = "what operator does it support, and how is that different from the default index type?"
        print(f"Q: {q2}")
        _, a2 = await ask(ws, q2, conversation_id=conv_id)
        print(f"A: {a2}\n")

        print("=" * 70)
        print("If A2 correctly talks about hash indexes (not something else),")
        print("conversation memory is working — the model resolved 'it' using")
        print("Turn 1's context, which was never repeated in Turn 2's question.")


if __name__ == "__main__":
    asyncio.run(main())
