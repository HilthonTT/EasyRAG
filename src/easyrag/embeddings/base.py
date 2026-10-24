from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Override if the model uses a separate query encoder."""
        return self.embed_documents([text])[0]

    def name(self) -> str:
        """Identifies the model, so a save index is never queried with a different one."""
        return type(self).__name__
