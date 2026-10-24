"""Core data types shared by every component."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A raw source document, e.g. one loaded file."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Chunk:
    """A piece of a document that gets embedded and indexed."""

    content: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class RetrievedChunk:
    """A chunk returned by a retriever, with its relevance score."""

    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    """The final output of a pipeline run."""

    question: str
    answer: str
    contexts: list[RetrievedChunk] = field(default_factory=list)
