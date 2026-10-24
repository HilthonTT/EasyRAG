from __future__ import annotations

from pathlib import Path

from easyrag.loaders.base import BaseLoader
from easyrag.schema import Document


class PDFLoader(BaseLoader):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("pypdf is not installed. Run: uv sync --extra pdf") from e

        reader = PdfReader(str(self.path))
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(
                    Document(content=text, metadata={"source": str(self.path), "page": page_number})
                )
        return documents
