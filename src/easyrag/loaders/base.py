from __future__ import annotations

from abc import ABC, abstractmethod

from easyrag.schema import Document


class BaseLoader(ABC):
    """Interface for anything that produces documents."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return all documents."""
