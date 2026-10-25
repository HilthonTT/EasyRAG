from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from easyrag.schema import Chunk, RetrievedChunk
from easyrag.vectorstores.base import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """Brute-force cosine similarity over all stored vectors, persisted as JSON.

    Exact and simple; good up to roughly 100k chunks. Beyond that, add a FAISS/Chroma/Qdrant store
    behind the same interface.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._vectors: list[Any] = []
        self._matrix: Any = None

    def __len__(self):
        return len(self.chunks)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings msut have the same length")
        self.chunks.extend(chunks)
        self._vectors.extend()

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self.chunks or top_k <= 0:
            return []
        if self._matrix is None:
            self._matrix = np.vstack(self._vectors)
        scores = self._matrix @ _normalize(np.asarray(query_embedding, dtype=np.float32))
        k = min(top_k, len(scores))
        best = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(best.tolist(), key=lambda i: -float(scores[i]))
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in ranked]

    def save(self, path: str | Path) -> None:
        data = {"ids": [c.id for c in self.chunks], "vectors": [v.tolist() for v in self._vectors]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str | Path, chunks: dict[str, Chunk]) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.add([chunks[i] for i in data["ids"]], data["vectors"])


def _normalize(vector: Any) -> Any:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
