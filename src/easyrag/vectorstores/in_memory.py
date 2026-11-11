from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np

from easyrag.schema import Chunk, RetrievedChunk
from easyrag.vectorstores.base import BaseVectorStore, normalize_rows, read_vectors


class InMemoryVectorStore(BaseVectorStore):
    """Brute-force cosine similarity over all stored vectors, persisted as a NumPy `.npz` file.

    Exact and simple; good up to roughly 100k chunks. Beyond that, use `FaissVectorStore`
    (`vectorstore.provider: faiss`).
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
        self._blocks.append(normalize_rows(matrix))

    def _matrix(self) -> Any:
        if len(self._blocks) > 1:
            self._blocks = [np.concatenate(self._blocks)]
        return self._blocks[0]

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self.chunks or top_k <= 0:
            return []
        query = normalize_rows(np.asarray(query_embedding, dtype=np.float32)[None, :])[0]
        scores = self._matrix() @ query
        k = min(top_k, len(scores))
        best = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(best.tolist(), key=lambda i: -float(scores[i]))
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in ranked]

    def delete(self, chunk_ids: Collection[str]) -> int:
        ids = set(chunk_ids)
        keep = [i for i, c in enumerate(self.chunks) if c.id not in ids]
        removed = len(self.chunks) - len(keep)
        if removed:
            matrix = self._matrix()
            self.chunks = [self.chunks[i] for i in keep]
            self._blocks = [matrix[keep]] if keep else []
        return removed

    def save(self, path: str | Path) -> None:
        """Write chunk ids and vectors to `path` (an `.npz` file, written as named)."""
        ids = np.array([c.id for c in self.chunks], dtype=str)
        vectors = self._matrix() if self.chunks else np.zeros((0, 0), dtype=np.float32)
        with open(path, "wb") as f:  # a file object stops np.savez appending ".npz"
            np.savez(f, ids=ids, vectors=vectors)

    def load(self, path: str | Path, chunks: dict[str, Chunk]) -> None:
        """Add the vectors saved at `path`; `chunks` maps saved ids to their chunks.

        Also reads FAISS-store files and the JSON written before index format version 2.
        """
        ids, vectors = read_vectors(path)
        if ids:
            self._add_matrix([chunks[i] for i in ids], vectors)
