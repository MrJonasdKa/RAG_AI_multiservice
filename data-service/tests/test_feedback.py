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


MSG_ID = "11111111-1111-1111-1111-111111111111"


@patch("app.main.get_pool")
def test_feedback_rejects_invalid_rating(mock_get_pool):
    resp = client.post(f"/messages/{MSG_ID}/feedback", json={"rating": 5})
    assert resp.status_code == 400
    mock_get_pool.assert_not_called()


@patch("app.main.get_pool")
def test_feedback_rejects_invalid_message_id(mock_get_pool):
    resp = client.post("/messages/not-a-uuid/feedback", json={"rating": 1})
    assert resp.status_code == 400


@patch("app.main.get_pool")
def test_feedback_404s_when_message_missing(mock_get_pool):
    fake_conn = MagicMock()
    fake_conn.fetchval = AsyncMock(return_value=None)
    mock_get_pool.return_value = _FakePool(fake_conn)

    resp = client.post(f"/messages/{MSG_ID}/feedback", json={"rating": 1})
    assert resp.status_code == 404


@patch("app.main.get_pool")
def test_feedback_happy_path(mock_get_pool):
    fake_conn = MagicMock()
    fake_conn.fetchval = AsyncMock(
        side_effect=[True, uuid.UUID("22222222-2222-2222-2222-222222222222")]
    )
    mock_get_pool.return_value = _FakePool(fake_conn)

    resp = client.post(f"/messages/{MSG_ID}/feedback", json={"rating": -1})

    assert resp.status_code == 200
    body = resp.json()
    assert body["feedback_id"] == "22222222-2222-2222-2222-222222222222"
    assert body["message_id"] == MSG_ID
    assert body["rating"] == -1


@patch("app.main.get_pool")
def test_get_feedback_returns_up_down_counts(mock_get_pool):
    fake_conn = MagicMock()
    fake_conn.fetchrow = AsyncMock(return_value={"up": 3, "down": 1})
    mock_get_pool.return_value = _FakePool(fake_conn)

    resp = client.get(f"/messages/{MSG_ID}/feedback")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"message_id": MSG_ID, "up": 3, "down": 1}
