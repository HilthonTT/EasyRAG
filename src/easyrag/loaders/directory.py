from __future__ import annotations

from pathlib import Path

from easyrag.loaders.base import BaseLoader
from easyrag.loaders.html import HTMLLoader
from easyrag.loaders.pdf import PDFLoader
from easyrag.loaders.text import TextLoader
from easyrag.schema import Document

LOADERS_BY_SUFFIX: dict[str, type[BaseLoader]] = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".rst": TextLoader,
    ".pdf": PDFLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
}


def _load_file(path: Path) -> list[Document]:
    loader_cls = LOADERS_BY_SUFFIX.get(path.suffix.lower(), TextLoader)
    return loader_cls(path).load()


class DirectoryLoader(BaseLoader):
    """Load every supported file under a directory, picking a loader by file extension."""

    def __init__(self, path: str | Path, glob: str = "**/*") -> None:
        self.path = Path(path)
        self.glob = glob

    def load(self) -> list[Document]:
        if self.path.is_file():
            return _load_file(self.path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"No such file or directory: {self.path}")
        documents: list[Document] = []

        for file in sorted(self.path.glob(self.glob)):
            if file.is_file() and file.suffix.lower() and LOADERS_BY_SUFFIX:
                documents.extend(_load_file(file))

        return documents
