from __future__ import annotations

from abc import ABC, abstractmethod

from easyrag.schema import Chunk, RetrievedChunk


class BaseVectorStore(ABC):
    """Interface for vector databases (in-memory, FAISS, Chroma, Qdrant, ...)."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their precomputed embeddings."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        """Return the `top_k` most similar chunks, best first."""

    @abstractmethod
    def __len__(self) -> int: ...
