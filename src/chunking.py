from __future__ import annotations

import re
from typing import List


WHITESPACE_RE = re.compile(r"\s+")
REPEATED_PUNCT_RE = re.compile(r"([^\w\s])\1{3,}")


def normalize_text(text: str) -> str:
    """Return cleaned page text suitable for chunking and retrieval."""
    cleaned = text.replace("\x00", " ").strip()
    cleaned = REPEATED_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int = 140, overlap: int = 30) -> List[str]:
    """Split cleaned text into overlapping word chunks."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    words = normalized.split()
    if len(words) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if len(chunk_words) < max(25, overlap):
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks

