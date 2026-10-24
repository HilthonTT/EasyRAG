from __future__ import annotations

from abc import ABC, abstractmethod

from easyrag.schema import Chunk, Document


class BaseSplitter(ABC):
    """Interface for splitting text into chunks."""

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split raw text into a list of chunk strings."""

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for i, piece in enumerate(self.split_text(doc.content)):
                chunks.append(
                    Chunk(
                        content=piece,
                        document_id=doc.id,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
        return chunks
