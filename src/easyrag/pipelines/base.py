from __future__ import annotations

from abc import ABC, abstractmethod

from easyrag.schema import RAGResponse


class BasePipeline(ABC):
    """Interface for RAG pipelines (naive, rerank, iterative, agentic, ...)."""

    @abstractmethod
    def run(self, question: str) -> RAGResponse:
        """Answer a single question."""

    def batch(self, questions: list[str]) -> list[RAGResponse]:
        return [self.run(q) for q in questions]
