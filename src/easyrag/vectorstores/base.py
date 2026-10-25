from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

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

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support saving")

    def load(self, path: str | Path, chunks: dict[str, Chunk]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support loading")
