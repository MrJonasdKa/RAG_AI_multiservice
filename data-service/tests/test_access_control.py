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


@patch("app.main._embed", new_callable=AsyncMock, return_value=[[0.1, 0.2, 0.3]])
@patch("app.main.get_pool")
def test_search_without_access_group_passes_none_to_query(mock_get_pool, mock_embed):
    fake_conn = MagicMock()
    fake_conn.fetch = AsyncMock(side_effect=[[], []])
    mock_get_pool.return_value = _FakePool(fake_conn)

    client.get("/chunks/search", params={"query": "index"})

    # every fetch call's 3rd bound parameter is the access_group filter —
    # omitting it in the request should bind SQL NULL (Python None), which
    # only matches public (access_group IS NULL) documents
    for call in fake_conn.fetch.call_args_list:
        assert call.args[-1] is None


@patch("app.main._embed", new_callable=AsyncMock, return_value=[[0.1, 0.2, 0.3]])
@patch("app.main.get_pool")
def test_search_with_access_group_passes_it_to_query(mock_get_pool, mock_embed):
    fake_conn = MagicMock()
    fake_conn.fetch = AsyncMock(side_effect=[[], []])
    mock_get_pool.return_value = _FakePool(fake_conn)

    client.get("/chunks/search", params={"query": "index", "access_group": "team-a"})

    for call in fake_conn.fetch.call_args_list:
        assert call.args[-1] == "team-a"


@patch("app.main._embed", new_callable=AsyncMock, return_value=[[0.1] * 3])
@patch("app.main.get_pool")
def test_ingest_stores_access_group(mock_get_pool, mock_embed):
    fake_conn = MagicMock()
    fake_conn.fetchval = AsyncMock(
        side_effect=[uuid.UUID("11111111-1111-1111-1111-111111111111")]
    )
    fake_conn.execute = AsyncMock()
    mock_get_pool.return_value = _FakePool(fake_conn)

    resp = client.post(
        "/documents",
        json={"title": "internal doc", "content": "some content", "access_group": "team-a"},
    )

    assert resp.status_code == 200
    insert_call = fake_conn.fetchval.call_args_list[0]
    assert insert_call.args[1] == "internal doc"  # title
    assert insert_call.args[3] == "team-a"  # access_group
