"""
Word-based text chunking with overlap.

Word count is a rough proxy for token count (~0.75 tokens/word for
English prose) — good enough for chunk sizing without pulling in a
tokenizer dependency. Overlap keeps context from being cut mid-thought
at chunk boundaries.
"""


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap

    return chunks
