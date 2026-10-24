from __future__ import annotations

from easyrag.splitters.base import BaseSplitter


class CharacterSplitter(BaseSplitter):
    """Fixed-size character windows with overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        pieces = (text[i : i + self.chunk_size].strip() for i in range(0, len(text), step))
        return [p for p in pieces if p]
