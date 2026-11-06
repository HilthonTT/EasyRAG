from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from easyrag.schema import Chunk, RetrievedChunk
from easyrag.vectorstores.base import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """Brute-force cosine similarity over all stored vectors, persisted as a NumPy `.npz` file.

    Exact and simple; good up to roughly 100k chunks. Beyond that, add a FAISS/Chroma/Qdrant store
    behind the same interface.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        # Normalized float32 rows. Each add() appends a block; search() concatenates them once into
        # a single matrix, so only one copy of the vectors is kept.
        self._blocks: list[Any] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if chunks:
            self._add_matrix(chunks, np.asarray(embeddings, dtype=np.float32))

    def _add_matrix(self, chunks: list[Chunk], matrix: Any) -> None:
        if matrix.ndim != 2 or (self._blocks and matrix.shape[1] != self._blocks[0].shape[1]):
            raise ValueError(f"embedding shape {matrix.shape} does not match the stored vectors")
        self.chunks.extend(chunks)
        self._blocks.append(_normalize_rows(matrix))

    def _matrix(self) -> Any:
        if len(self._blocks) > 1:
            self._blocks = [np.concatenate(self._blocks)]
        return self._blocks[0]

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self.chunks or top_k <= 0:
            return []
        query = _normalize_rows(np.asarray(query_embedding, dtype=np.float32)[None, :])[0]
        scores = self._matrix() @ query
        k = min(top_k, len(scores))
        best = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(best.tolist(), key=lambda i: -float(scores[i]))
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in ranked]

    def save(self, path: str | Path) -> None:
        """Write chunk ids and vectors to `path` (an `.npz` file, written as named)."""
        ids = np.array([c.id for c in self.chunks], dtype=str)
        vectors = self._matrix() if self.chunks else np.zeros((0, 0), dtype=np.float32)
        with open(path, "wb") as f:  # a file object stops np.savez appending ".npz"
            np.savez(f, ids=ids, vectors=vectors)

    def load(self, path: str | Path, chunks: dict[str, Chunk]) -> None:
        """Add the vectors saved at `path`; `chunks` maps saved ids to their chunks.

        Also reads the JSON files written by EasyRAG before index format version 2.
        """
        path = Path(path)
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            ids, vectors = data["ids"], np.asarray(data["vectors"], dtype=np.float32)
        else:
            with np.load(path, allow_pickle=False) as data:
                ids, vectors = data["ids"].tolist(), data["vectors"]
        if ids:
            self._add_matrix([chunks[i] for i in ids], vectors)


def _normalize_rows(matrix: Any) -> Any:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)  # zero vectors stay zero
