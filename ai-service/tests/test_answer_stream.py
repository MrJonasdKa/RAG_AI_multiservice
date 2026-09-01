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
        "score": 0.12,
    }
]

FAKE_HISTORY = [
    {"message_id": "m1", "role": "user", "content": "What is a B-tree index?"},
    {"message_id": "m2", "role": "assistant", "content": "A B-tree index is a sorted tree structure."},
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


@patch("app.main._save_message_async", new_callable=AsyncMock)
@patch("app.main._create_conversation_async", new_callable=AsyncMock, return_value="new-conv-id")
@patch("app.main._retrieve_chunks_async", new_callable=AsyncMock, return_value=FAKE_CHUNKS)
@patch("app.main.get_client")
def test_stream_without_conversation_id_creates_one_and_reports_it(
    mock_get_client, mock_retrieve, mock_create_conv, mock_save
):
    fake_chunks_stream = [_FakeStreamChunk("Hello "), _FakeStreamChunk("world.")]
    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=_fake_async_iter(fake_chunks_stream)
    )
    mock_get_client.return_value = mock_client

    with client.websocket_connect("/answer/stream") as ws:
        ws.send_json({"question": "how do indexes work?"})

        conv_msg = ws.receive_json()
        assert conv_msg == {"type": "conversation", "conversation_id": "new-conv-id"}

        first = ws.receive_json()
        second = ws.receive_json()
        done = ws.receive_json()

        assert first == {"type": "token", "text": "Hello "}
        assert second == {"type": "token", "text": "world."}
        assert done["type"] == "done"

    mock_create_conv.assert_called_once()
    assert mock_save.call_count == 2
    mock_save.assert_any_call("new-conv-id", "user", "how do indexes work?")
    mock_save.assert_any_call("new-conv-id", "assistant", "Hello world.")


@patch("app.main._save_message_async", new_callable=AsyncMock)
@patch("app.main._get_history_async", new_callable=AsyncMock, return_value=FAKE_HISTORY)
@patch("app.main._retrieve_chunks_async", new_callable=AsyncMock, return_value=FAKE_CHUNKS)
@patch("app.main.get_client")
def test_stream_with_conversation_id_uses_history_in_prompt(
    mock_get_client, mock_retrieve, mock_get_history, mock_save
):
    fake_chunks_stream = [_FakeStreamChunk("Hash indexes only support equality.")]
    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=_fake_async_iter(fake_chunks_stream)
    )
    mock_get_client.return_value = mock_client

    with client.websocket_connect("/answer/stream") as ws:
        ws.send_json({"question": "what about hash indexes?", "conversation_id": "existing-conv-id"})

        conv_msg = ws.receive_json()
        assert conv_msg == {"type": "conversation", "conversation_id": "existing-conv-id"}

        ws.receive_json()  # token
        ws.receive_json()  # done

    mock_get_history.assert_called_once_with("existing-conv-id")

    call_kwargs = mock_client.aio.models.generate_content_stream.call_args.kwargs
    assert "What is a B-tree index?" in call_kwargs["contents"]
    assert "A B-tree index is a sorted tree structure." in call_kwargs["contents"]
