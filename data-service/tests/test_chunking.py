from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_is_a_single_chunk():
    text = "hello world this is a short chunk"
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert chunks == [text]


def test_long_text_splits_into_multiple_overlapping_chunks():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1

    # consecutive chunks should share exactly the overlap region
    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=10, overlap=10)
