from __future__ import annotations

from abc import ABC, abstractmethod

from easyrag.schema import Chunk, Document


class BaseSplitter(ABC):
    """Interface for splitting text into chunks."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split one string into pieces of at most roughly `chunk_size` characters."""

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for i, piece in enumerate(self.split_text(doc.content)):
                metadata = {**doc.metadata, "chunk_index": i}
                chunks.append(Chunk(content=piece, document_id=doc.id, metadata=metadata))
        return chunks
