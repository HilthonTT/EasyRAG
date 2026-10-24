from __future__ import annotations

from easyrag.config import Config
from easyrag.embeddings import FakeEmbeddings
from easyrag.generators import BaseGenerator, EchoGenerator
from easyrag.pipelines.base import BasePipeline
from easyrag.prompts import DEFAULT_RAG_PROMPT, build_rag_prompt
from easyrag.retrievers import VectorRetriever
from easyrag.schema import Document, RAGResponse
from easyrag.splitters import BaseSplitter, CharacterSplitter
from easyrag.vectorstores import InMemoryVectorStore


class NaiveRAGPipeline(BasePipeline):
    """The classic retrieve-then-generate flow: split -> embed -> retrieve -> generate."""

    def __init__(
        self,
        retriever: VectorRetriever,
        generator: BaseGenerator,
        splitter: BaseSplitter,
        prompt_template: str = DEFAULT_RAG_PROMPT,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.splitter = splitter
        self.prompt_template = prompt_template

    @classmethod
    def from_config(cls, config: Config | None = None) -> NaiveRAGPipeline:
        """Build a pipeline from config. Only the built-in local components are wired so far."""
        config = config or Config()
        retriever = VectorRetriever(
            embeddings=FakeEmbeddings(),
            vectorstore=InMemoryVectorStore(),
            top_k=config.retriever.top_k,
        )
        splitter = CharacterSplitter(config.splitter.chunk_size, config.splitter.chunk_overlap)
        return cls(retriever=retriever, generator=EchoGenerator(), splitter=splitter)

    def index(self, documents: list[Document]) -> int:
        """Split and index documents. Returns the number of chunks added."""
        chunks = self.splitter.split_documents(documents)
        self.retriever.add_chunks(chunks)
        return len(chunks)

    def run(self, question: str) -> RAGResponse:
        results = self.retriever.retrieve(question)
        prompt = build_rag_prompt(question, results, self.prompt_template)
        answer = self.generator.generate(prompt)
        return RAGResponse(question=question, answer=answer, contexts=results)
