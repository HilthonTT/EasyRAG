from __future__ import annotations

from collections.abc import Iterator

from easyrag.pipelines.base import BasePipeline
from easyrag.prompts import build_rag_prompt
from easyrag.schema import RAGResponse, RetrievedChunk


class NaiveRAGPipeline(BasePipeline):
    """The classic retrieve-then-generate flow (FlashRAG's SequentialPipeline)."""

    def search_query(self, question: str) -> str:
        """The text used for retrieval. Subclasses may rewrite the question (see HyDE)."""
        return question

    def run(self, question: str) -> RAGResponse:
        contexts, answer = self.stream(question)
        return RAGResponse(question=question, answer="".join(answer), contexts=contexts)

    def stream(self, question: str) -> tuple[list[RetrievedChunk], Iterator[str]]:
        contexts = self.retrieve(self.search_query(question))
        prompt = build_rag_prompt(question, contexts, self.prompt_template)
        return contexts, self.generator.stream(prompt, system=self.system_prompt)
