"""Core data types shared by every component."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Document:
    """A raw source document, e.g. one loaded file or one PDF page."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)


@dataclass
class Chunk:
    """A piece of a document that gets embedded and indexed."""

    content: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)

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
