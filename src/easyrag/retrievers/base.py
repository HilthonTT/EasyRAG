from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection

from easyrag.schema import Chunk, RetrievedChunk


class BaseRetriever(ABC):
    """Interface for retrievers (dense, sparse/BM25, hybrid, ...)."""

    def __init__(self, top_k: int = 4) -> None:
        self.top_k = top_k

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks so they can be retrieved."""

    def remove_chunks(self, chunk_ids: Collection[str]) -> None:
        """Remove the chunks with these ids from the index (incremental re-indexing)."""
        raise NotImplementedError(f"{type(self).__name__} does not support removing chunks")

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return up to `top_k` (default `self.top_k`) chunks, most relevant first."""
