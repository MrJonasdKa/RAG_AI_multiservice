from unittest.mock import MagicMock, patch

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


def test_answer_empty_question_is_rejected():
    resp = client.post("/answer", json={"question": "   "})
    assert resp.status_code == 400


@patch("app.main._retrieve_chunks", return_value=[])
def test_answer_with_no_matching_chunks_returns_404(mock_retrieve):
    resp = client.post("/answer", json={"question": "something obscure"})
    assert resp.status_code == 404


@patch("app.main._retrieve_chunks", return_value=FAKE_CHUNKS)
@patch("app.main.get_client")
def test_answer_happy_path(mock_get_client, mock_retrieve):
    fake_response = MagicMock()
    fake_response.text = "An index speeds up lookups by avoiding a full table scan."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response
    mock_get_client.return_value = mock_client

    resp = client.post("/answer", json={"question": "how do indexes work?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == fake_response.text
    assert len(body["sources"]) == 1
    assert body["sources"][0]["document_title"] == "indices"

    # confirm the prompt actually included the retrieved chunk content
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "An index allows the database server" in call_kwargs["contents"]
