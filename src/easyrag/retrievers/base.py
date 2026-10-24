from __future__ import annotations

from abc import ABC, abstractmethod

from easyrag.schema import RetrievedChunk


class BaseRetriever(ABC):
    """Interface for retrievers (dense, sparse/BM25, hybrid, ...)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return chunks relevant to `query`, best first."""
