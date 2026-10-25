from __future__ import annotations

from easyrag.splitters.base import BaseSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


class RecursiveCharacterSplitter(BaseSplitter):
    def __init__(
        self, chunk_size: int = 800, chunk_overlap: int = 100, separators: list[str] | None = None
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or DEFAULT_SEPARATORS

    def _merge(self, splits: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        length = 0

        for piece in splits:
            if current and length + len(piece) > self.chunk_size:
                chunks.append("".join(current))
                # Keep a tail of the previous chunk as overlap while the next piece still fits.
                while current and (
                    length > self.chunk_overlap or length + len(piece) > self.chunk_size
                ):
                    length -= len(current.pop(0))
            current.append(piece)
            length += len(piece)

        if current:
            chunks.append("".join(current))
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        separator, remaining = separators[-1], []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator, remaining = sep, separators[i + 1 :]
                break

        if separator:
            parts = text.split(separator)
            splits = [p + separator for p in parts[:-1]] + [parts[-1]]
        else:
            splits = list(text)

        chunks: list[str] = []
        pending: list[str] = []

        for piece in splits:
            if len(piece) <= self.chunk_size:
                pending.append(piece)
                continue
            if pending:
                chunks.extend(self._merge(pending))
                pending = []
            if remaining:
                chunks.extend(self._split(piece, remaining))
            else:
                chunks.append(piece)
        if pending:
            chunks.extend(self._merge(pending))
        return chunks

    def split_text(self, text: str) -> list[str]:
        return [c for c in (c.strip() for c in self._split(text, self.separators)) if c]
