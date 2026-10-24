from __future__ import annotations

from easyrag.splitters.base import BaseSplitter


class CharacterSplitter(BaseSplitter):
    """Fixed-size character windows with overlap. Simple, but cuts through words and sentences."""

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        pieces = (text[i : i + self.chunk_size].strip() for i in range(0, len(text), step))
        return [p for p in pieces if p]
