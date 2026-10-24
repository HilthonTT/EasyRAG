from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from easyrag.loaders.base import BaseLoader
from easyrag.schema import Document

_SKIP_TAGS = {"script", "style", "noscript", "head", "nav", "footer"}
_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "section", "article", "h1", "h2", "h3", "h4", "h5"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


class HTMLLoader(BaseLoader):
    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        parser = _TextExtractor()
        parser.feed(self.path.read_text(encoding=self.encoding, errors="replace"))
        lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
        content = "\n".join(line for line in lines if line)
        metadata = {"source": str(self.path), "title": parser.title.strip()}
        return [Document(content=content, metadata=metadata)]
