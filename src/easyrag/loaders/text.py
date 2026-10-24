from __future__ import annotations

from pathlib import Path

from easyrag.loaders.base import BaseLoader
from easyrag.schema import Document


class TextLoader(BaseLoader):
    """Load a single plain-text or Markdown file."""

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        content = self.path.read_text(encoding=self.encoding)
        return [Document(content=content, metadata={"source": str(self.path)})]
