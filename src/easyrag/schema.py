"""Core data types shared by every component."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _content_id(*parts: Any) -> str:
    """Stable id from content, so re-indexing the same input yields the same ids."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass
class Document:
    """A raw source document, e.g. one loaded file or one PDF page.

    Without an explicit `id`, it is derived from the content and metadata (source, page, ...).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _content_id(self.content, self.metadata)


@dataclass
class Chunk:
    """A piece of a document that gets embedded and indexed.

    Without an explicit `id`, it is derived from the document id, content and metadata (which
    includes `chunk_index`, so a paragraph repeated within one document still gets two ids).
    """

    content: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _content_id(self.document_id, self.content, self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**data)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))


@dataclass
class RetrievedChunk:
    """A chunk returned by a retriever, with its relevance score (higher is better)."""

    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    """The final output of a pipeline run."""

    question: str
    answer: str
    contexts: list[RetrievedChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
