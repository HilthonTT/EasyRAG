from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any, TypeVar

from easyrag.config import Config
from easyrag.generators.base import BaseGenerator
from easyrag.prompts import DEFAULT_RAG_PROMPT, SYSTEM_PROMPT
from easyrag.rerankers.base import BaseReranker
from easyrag.retrievers.base import BaseRetriever
from easyrag.schema import Chunk, Document, RAGResponse, RetrievedChunk
from easyrag.splitters.base import BaseSplitter
from easyrag.splitters.recursive import RecursiveCharacterSplitter
from easyrag.retrievers.hybrid import HybridRetriever
from easyrag.retrievers.vector import VectorRetriever

P = TypeVar("P", bound="BasePipeline")

INDEX_FORMAT_VERSION = 1


class BasePipeline(ABC):
    """Shared indexing, retrieval (with optional reranking) and persistence.

    Subclasses (naive, HyDE, iterative, ...) only decide how retrieval and generation are
    combined in `run`.
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
        from easyrag.factory import build_components

        config = config or Config.load()
        return cls(**{**build_components(config), **cls.config_kwargs(config), **overrides})

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        """Extra constructor arguments a subclass reads from config."""
        return {}

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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        embeddings = []
        for i, retriever in enumerate(self._vector_retrievers()):
            retriever.vectorstore.save(path / f"vectors_{i}.json")
            embeddings.append(retriever.embeddings.name)
        manifest = {
            "version": INDEX_FORMAT_VERSION,
            "num_chunks": len(self.chunks),
            "embeddings": embeddings,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> int:
        """Load a saved index into this (empty) pipeline. Returns the number of chunks."""
        path = Path(path)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No index found at {path}. Run `rag index` first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        by_id = {c.id: c for c in chunks}
        saved_embeddings: list[str] = manifest.get("embeddings", [])

        vector_retrievers = self._vector_retrievers()
        for i, retriever in enumerate(vector_retrievers):
            vectors_path = path / f"vectors_{i}.json"
            same_model = (
                i < len(saved_embeddings) and saved_embeddings[i] == retriever.embeddings.name
            )
            if vectors_path.exists() and same_model:
                retriever.vectorstore.load(vectors_path, by_id)
            else:
                retriever.add_chunks(chunks)  # different model: re-embed

        for leaf in _leaf_retrievers(self.retriever):
            if leaf not in vector_retrievers:
                leaf.add_chunks(chunks)  # BM25 and other lexical indexes are cheap to rebuild

        self.chunks.extend(chunks)

        return len(chunks)

    def _vector_retrievers(self) -> list[VectorRetriever]:
        return [r for r in _leaf_retrievers(self.retriever) if isinstance(r, VectorRetriever)]


def _leaf_retrievers(retriever: BaseRetriever) -> list[BaseRetriever]:
    if isinstance(retriever, HybridRetriever):
        return [leaf for r in retriever.retrievers for leaf in _leaf_retrievers(r)]
    return [retriever]
