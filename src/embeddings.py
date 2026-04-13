from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List


TOKEN_RE = re.compile(r"[a-z0-9]+")


class LocalEmbeddingModel:
    """Simple deterministic hashing-based embedding model for local retrieval."""

    def __init__(self, dimensions: int = 192) -> None:
        self.dimensions = dimensions

    def embed_text(self, text: str) -> List[float]:
        """Return a normalized dense vector for the provided text."""
        vector = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[index] += sign * weight
        return self._normalize(vector)

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return [self.embed_text(text) for text in texts]

    def cosine_similarity(self, left: List[float], right: List[float]) -> float:
        """Compute cosine similarity between normalized vectors."""
        if not left or not right:
            return 0.0
        return sum(a * b for a, b in zip(left, right))

    def _normalize(self, vector: List[float]) -> List[float]:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]

