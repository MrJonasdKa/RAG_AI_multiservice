from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


FAKE_CHUNKS = [
    {
        "chunk_id": "c1",
        "document_id": "d1",
        "document_title": "indices",
        "content": "An index allows the database server to find rows faster.",
        "distance": 0.12,
    }
]


class _FakeStreamChunk:
    def __init__(self, text: str):
        self.text = text


async def _fake_async_iter(items):
    for item in items:
        yield item


@patch("app.main._retrieve_chunks_async", new_callable=AsyncMock, return_value=[])
def test_stream_empty_question_sends_error(mock_retrieve):
    with client.websocket_connect("/answer/stream") as ws:
        ws.send_json({"question": "   "})
        msg = ws.receive_json()
        assert msg["type"] == "error"
    mock_retrieve.assert_not_called()


@patch("app.main._retrieve_chunks_async", new_callable=AsyncMock, return_value=[])
def test_stream_no_chunks_found_sends_error(mock_retrieve):
    with client.websocket_connect("/answer/stream") as ws:
        ws.send_json({"question": "something obscure"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "no relevant chunks" in msg["message"]


@patch("app.main._retrieve_chunks_async", new_callable=AsyncMock, return_value=FAKE_CHUNKS)
@patch("app.main.get_client")
def test_stream_happy_path_sends_tokens_then_done(mock_get_client, mock_retrieve):
    fake_chunks_stream = [_FakeStreamChunk("Hello "), _FakeStreamChunk("world.")]

    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=_fake_async_iter(fake_chunks_stream)
    )
    mock_get_client.return_value = mock_client

    with client.websocket_connect("/answer/stream") as ws:
        ws.send_json({"question": "how do indexes work?"})

        first = ws.receive_json()
        second = ws.receive_json()
        done = ws.receive_json()

        assert first == {"type": "token", "text": "Hello "}
        assert second == {"type": "token", "text": "world."}
        assert done["type"] == "done"
        assert done["sources"][0]["document_title"] == "indices"
