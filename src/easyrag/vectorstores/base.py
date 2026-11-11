from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np

from easyrag.schema import Chunk, RetrievedChunk


class BaseVectorStore(ABC):
    """Interface for storing chunk embeddings and searching them by similarity."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings (same order, same length)."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the `top_k` most similar chunks, best first."""

    @abstractmethod
    def __len__(self) -> int: ...

    def delete(self, chunk_ids: Collection[str]) -> int:
        """Remove the chunks with these ids. Returns how many were removed."""
        raise NotImplementedError(f"{type(self).__name__} does not support deleting")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support saving")

    def load(self, path: str | Path, chunks: dict[str, Chunk]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support loading")


def read_vectors(path: str | Path) -> tuple[list[str], Any]:
    """Chunk ids and a float32 matrix from any saved vector file.

    Reads the `.npz` written by `InMemoryVectorStore` (`vectors`) or `FaissVectorStore`
    (`faiss_index`), and the JSON of index format version 1, so an index saved with one store
    loads into another without re-embedding.
    """
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data["ids"]), np.asarray(data["vectors"], dtype=np.float32)
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"].tolist()
        if "vectors" in data:
            return ids, data["vectors"]
        from easyrag.vectorstores.faiss import deserialize_vectors

        return ids, deserialize_vectors(data["faiss_index"])


def normalize_rows(matrix: Any) -> Any:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)  # zero vectors stay zero
