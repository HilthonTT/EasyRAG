from __future__ import annotations

from abc import ABC, abstractmethod

from easyrag.schema import RetrievedChunk


class BaseReranker(ABC):
    """Interface for rescoring retrieved chunks against the query (second retrieval stage)."""

    @abstractmethod
    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return the `top_k` best of `results`, rescored, best first."""
