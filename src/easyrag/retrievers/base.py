from __future__ import annotations

from abc import ABC, abstractmethod

from easyrag.schema import Chunk, RetrievedChunk


class BaseRetriever(ABC):
    """Interface for retrievers (dense, sparse/BM25, hybrid, ...)."""

    def __init__(self, top_k: int = 4) -> None:
        self.top_k = top_k

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks so they can be retrieved."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return up to `top_k` (default `self.top_k`) chunks, most relevant first."""
