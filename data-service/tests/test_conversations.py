import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class _FakeAcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquireCtx(self._conn)


def _fake_pool_with(conn) -> _FakePool:
    return _FakePool(conn)


@patch("app.main.get_pool")
def test_create_conversation_returns_new_id(mock_get_pool):
    fake_conn = MagicMock()
    fake_conn.fetchval = AsyncMock(return_value=uuid.UUID("11111111-1111-1111-1111-111111111111"))
    mock_get_pool.return_value = _fake_pool_with(fake_conn)

    resp = client.post("/conversations")

    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": "11111111-1111-1111-1111-111111111111"}


@patch("app.main.get_pool")
def test_add_message_rejects_bad_role(mock_get_pool):
    resp = client.post(
        "/conversations/11111111-1111-1111-1111-111111111111/messages",
        json={"role": "system", "content": "hi"},
    )
    assert resp.status_code == 400
    mock_get_pool.assert_not_called()


@patch("app.main.get_pool")
def test_add_message_rejects_invalid_uuid(mock_get_pool):
    resp = client.post(
        "/conversations/not-a-uuid/messages",
        json={"role": "user", "content": "hi"},
    )
    assert resp.status_code == 400


@patch("app.main.get_pool")
def test_add_message_404s_when_conversation_missing(mock_get_pool):
    fake_conn = MagicMock()
    fake_conn.fetchval = AsyncMock(return_value=None)  # conversation existence check fails
    mock_get_pool.return_value = _fake_pool_with(fake_conn)

    resp = client.post(
        "/conversations/11111111-1111-1111-1111-111111111111/messages",
        json={"role": "user", "content": "hi"},
    )
    assert resp.status_code == 404


@patch("app.main.get_pool")
def test_add_message_happy_path(mock_get_pool):
    fake_conn = MagicMock()
    fake_conn.fetchval = AsyncMock(
        side_effect=[True, uuid.UUID("22222222-2222-2222-2222-222222222222")]
    )
    mock_get_pool.return_value = _fake_pool_with(fake_conn)

    resp = client.post(
        "/conversations/11111111-1111-1111-1111-111111111111/messages",
        json={"role": "user", "content": "how do indexes work?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["message_id"] == "22222222-2222-2222-2222-222222222222"
    assert body["role"] == "user"
    assert body["content"] == "how do indexes work?"


@patch("app.main.get_pool")
def test_get_messages_returns_ordered_history(mock_get_pool):
    fake_conn = MagicMock()
    fake_conn.fetch = AsyncMock(
        return_value=[
            {"id": uuid.UUID("33333333-3333-3333-3333-333333333333"), "role": "user", "content": "q1"},
            {"id": uuid.UUID("44444444-4444-4444-4444-444444444444"), "role": "assistant", "content": "a1"},
        ]
    )
    mock_get_pool.return_value = _fake_pool_with(fake_conn)

    resp = client.get("/conversations/11111111-1111-1111-1111-111111111111/messages")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["role"] == "user"
    assert body[1]["role"] == "assistant"
