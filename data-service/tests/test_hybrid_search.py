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


DOC_ID = uuid.UUID("aaaaaaaa-1111-1111-1111-111111111111")

CHUNK_A = {
    "chunk_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
    "document_id": DOC_ID,
    "document_title": "indices",
    "content": "Chunk A — appears in both vector and keyword results",
}
CHUNK_B = {
    "chunk_id": uuid.UUID("22222222-2222-2222-2222-222222222222"),
    "document_id": DOC_ID,
    "document_title": "indices",
    "content": "Chunk B — vector search only",
}
CHUNK_C = {
    "chunk_id": uuid.UUID("33333333-3333-3333-3333-333333333333"),
    "document_id": DOC_ID,
    "document_title": "mvcc",
    "content": "Chunk C — keyword search only",
}


@patch("app.main._embed", new_callable=AsyncMock, return_value=[[0.1, 0.2, 0.3]])
@patch("app.main.get_pool")
def test_hybrid_search_ranks_chunk_found_by_both_methods_highest(mock_get_pool, mock_embed):
    fake_conn = MagicMock()
    # first .fetch() call = vector results, second = keyword results
    fake_conn.fetch = AsyncMock(
        side_effect=[
            [CHUNK_A, CHUNK_B],  # vector: A rank 1, B rank 2
            [CHUNK_A, CHUNK_C],  # keyword: A rank 1, C rank 2
        ]
    )
    mock_get_pool.return_value = _FakePool(fake_conn)

    resp = client.get("/chunks/search", params={"query": "index performance"})

    assert resp.status_code == 200
    body = resp.json()
    ids = [r["chunk_id"] for r in body]

    # A was found by both methods at rank 1 each time — must fuse to the top
    assert ids[0] == str(CHUNK_A["chunk_id"])
    assert str(CHUNK_B["chunk_id"]) in ids
    assert str(CHUNK_C["chunk_id"]) in ids

    scores = {r["chunk_id"]: r["score"] for r in body}
    assert scores[str(CHUNK_A["chunk_id"])] > scores[str(CHUNK_B["chunk_id"])]
    assert scores[str(CHUNK_A["chunk_id"])] > scores[str(CHUNK_C["chunk_id"])]


@patch("app.main._embed", new_callable=AsyncMock, return_value=[[0.1, 0.2, 0.3]])
@patch("app.main.get_pool")
def test_hybrid_search_respects_top_k(mock_get_pool, mock_embed):
    fake_conn = MagicMock()
    fake_conn.fetch = AsyncMock(side_effect=[[CHUNK_A, CHUNK_B, CHUNK_C], []])
    mock_get_pool.return_value = _FakePool(fake_conn)

    resp = client.get("/chunks/search", params={"query": "index", "top_k": 2})

    assert resp.status_code == 200
    assert len(resp.json()) == 2


@patch("app.main._embed", new_callable=AsyncMock, return_value=[[0.1, 0.2, 0.3]])
@patch("app.main.get_pool")
def test_hybrid_search_includes_keyword_only_match_vector_search_would_miss(mock_get_pool, mock_embed):
    fake_conn = MagicMock()
    # vector search found nothing useful; keyword search found C
    fake_conn.fetch = AsyncMock(side_effect=[[], [CHUNK_C]])
    mock_get_pool.return_value = _FakePool(fake_conn)

    resp = client.get("/chunks/search", params={"query": "specific exact term"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["chunk_id"] == str(CHUNK_C["chunk_id"])
