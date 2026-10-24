from __future__ import annotations

import heapq
import math

from easyrag.schema import Chunk, RetrievedChunk
from easyrag.vectorstores.base import BaseVectorStore


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryVectorStore(BaseVectorStore):
    """Brute-force cosine search in pure Python. Fine for small corpora."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        scored = (
            RetrievedChunk(chunk=chunk, score=cosine_similarity(query_embedding, emb))
            for chunk, emb in zip(self._chunks, self._embeddings, strict=True)
        )
        return heapq.nlargest(top_k, scored, key=lambda r: r.score)

    def __len__(self) -> int:
        return len(self._chunks)
