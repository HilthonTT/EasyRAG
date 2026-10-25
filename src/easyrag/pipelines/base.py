from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, TypeVar

from easyrag.config import Config
from easyrag.generators.base import BaseGenerator
from easyrag.prompts import DEFAULT_RAG_PROMPT, SYSTEM_PROMPT
from easyrag.rerankers.base import BaseReranker
from easyrag.retrievers.base import BaseRetriever
from easyrag.schema import Chunk, Document, RAGResponse, RetrievedChunk
from easyrag.splitters.base import BaseSplitter
from easyrag.splitters.recursive import RecursiveCharacterSplitter

P = TypeVar("P", bound="BasePipeline")

INDEX_FORMAT_VERSION = 1


class BasePipeline(ABC):
    """Shared indexing, retrieval (with optional reranking) and persistence.

    Subclasses (naive, HyDE, iterative, ...) only decide how retrieval and generation are
    combined in `run`, as in FlashRAG's pipeline classes.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        splitter: BaseSplitter | None = None,
        reranker: BaseReranker | None = None,
        top_k: int = 4,
        fetch_k: int = 20,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.splitter = splitter or RecursiveCharacterSplitter()
        self.reranker = reranker
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.chunks: list[Chunk] = []

    @classmethod
    def from_config(cls: type[P], config: Config | None = None, **overrides: Any) -> P:
        """Build the pipeline and all its components from config"""

    @abstractmethod
    def run(self, question: str) -> RAGResponse:
        """Answer a single question."""

    def batch(self, questions: list[str]) -> list[RAGResponse]:
        return [self.run(q) for q in questions]

    def stream(self, question: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        """Return the contexts and an iterator over the answer text."""
        response = self.run(question)
        return response.contexts, iter([response.answer])

    def index(self, documents: list[Document]) -> int:
        """Split and index documents. Returns the number of chunks added."""
        chunks = self.splitter.split_documents(documents)
        self.retriever.add_chunks(chunks)
        self.chunks.extend(chunks)
        return len(chunks)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        if self.reranker is None:
            return self.retriever.retrieve(query, self.top_k)
        candidates = self.retriever.retrieve(query, max(self.fetch_k, self.top_k))
        return self.reranker.rerank(query, candidates, self.top_k)
