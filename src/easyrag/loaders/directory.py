from easyrag.loaders.base import BaseLoader
from easyrag.loaders.text import TextLoader
from easyrag.schema import Document


from pathlib import Path


class DirectoryLoader(BaseLoader):
    """Load every matching text file in a directory (recursively)."""

    def __init__(self, path: str | Path, glob: str = "**/*.txt", encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.glob = glob
        self.encoding = encoding

    def load(self) -> list[Document]:
        documents: list[Document] = []
        for file in sorted(self.path.glob(self.glob)):
            if file.is_file():
                documents.extend(TextLoader(file, self.encoding).load())
        return documents
