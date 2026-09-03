from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_answer_cache():
    from app.main import _answer_cache

    _answer_cache.clear()
    yield
    _answer_cache.clear()


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


def test_answer_empty_question_is_rejected():
    resp = client.post("/answer", json={"question": "   "})
    assert resp.status_code == 400


@patch("app.main._retrieve_chunks", return_value=[])
def test_answer_with_no_matching_chunks_returns_404(mock_retrieve):
    resp = client.post("/answer", json={"question": "something obscure"})
    assert resp.status_code == 404


@patch("app.main._save_message")
@patch("app.main._create_conversation", return_value="new-conv-id")
@patch("app.main._retrieve_chunks", return_value=FAKE_CHUNKS)
@patch("app.main.get_client")
def test_answer_without_conversation_id_creates_a_new_conversation(
    mock_get_client, mock_retrieve, mock_create_conv, mock_save
):
    fake_response = MagicMock()
    fake_response.text = "An index speeds up lookups by avoiding a full table scan."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response
    mock_get_client.return_value = mock_client

    resp = client.post("/answer", json={"question": "how do indexes work?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "new-conv-id"
    mock_create_conv.assert_called_once()

    # both turns should be persisted
    assert mock_save.call_count == 2
    mock_save.assert_any_call("new-conv-id", "user", "how do indexes work?")
    mock_save.assert_any_call("new-conv-id", "assistant", fake_response.text)

    # confirm the prompt actually included the retrieved chunk content
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "An index allows the database server" in call_kwargs["contents"]


@patch("app.main._save_message")
@patch("app.main._get_history", return_value=FAKE_HISTORY)
@patch("app.main._retrieve_chunks", return_value=FAKE_CHUNKS)
@patch("app.main.get_client")
def test_answer_with_conversation_id_includes_prior_history_in_prompt(
    mock_get_client, mock_retrieve, mock_get_history, mock_save
):
    fake_response = MagicMock()
    fake_response.text = "Unlike B-tree, hash indexes only support equality."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response
    mock_get_client.return_value = mock_client

    resp = client.post(
        "/answer",
        json={"question": "what about hash indexes?", "conversation_id": "existing-conv-id"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "existing-conv-id"
    mock_get_history.assert_called_once_with("existing-conv-id")

    # the prior turn must actually reach the prompt sent to Gemini
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "What is a B-tree index?" in call_kwargs["contents"]
    assert "A B-tree index is a sorted tree structure." in call_kwargs["contents"]


@patch("app.main._save_message")
@patch("app.main._create_conversation", side_effect=["conv-1", "conv-2"])
@patch("app.main._retrieve_chunks", return_value=FAKE_CHUNKS)
@patch("app.main.get_client")
def test_answer_caches_repeated_fresh_questions(mock_get_client, mock_retrieve, mock_create_conv, mock_save):
    fake_response = MagicMock()
    fake_response.text = "An index speeds up lookups by avoiding a full table scan."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response
    mock_get_client.return_value = mock_client

    first = client.post("/answer", json={"question": "how do indexes work?"})
    second = client.post("/answer", json={"question": "How Do Indexes Work?  "})  # same question, different case/spacing

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["answer"] == first.json()["answer"]

    # Gemini should only have been called once — the second request was served from cache
    assert mock_client.models.generate_content.call_count == 1
    # retrieval is skipped entirely on a cache hit too — nothing left to redo
    assert mock_retrieve.call_count == 1


@patch("app.main._save_message")
@patch("app.main._get_history", return_value=[])
@patch("app.main._retrieve_chunks", return_value=FAKE_CHUNKS)
@patch("app.main.get_client")
def test_answer_with_conversation_id_never_uses_cache(mock_get_client, mock_retrieve, mock_get_history, mock_save):
    fake_response = MagicMock()
    fake_response.text = "An index speeds up lookups by avoiding a full table scan."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response
    mock_get_client.return_value = mock_client

    payload = {"question": "how do indexes work?", "conversation_id": "existing-conv-id"}
    first = client.post("/answer", json=payload)
    second = client.post("/answer", json=payload)

    assert first.json()["cached"] is False
    assert second.json()["cached"] is False
    # generation must run fresh both times — conversation turns are never cached
    assert mock_client.models.generate_content.call_count == 2
