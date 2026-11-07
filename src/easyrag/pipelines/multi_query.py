from __future__ import annotations

import re
from typing import Any

from easyrag.config import Config
from easyrag.pipelines.naive import NaiveRAGPipeline
from easyrag.prompts import MULTI_QUERY_SYSTEM_PROMPT
from easyrag.retrievers.hybrid import reciprocal_rank_fusion
from easyrag.schema import RAGResponse, RetrievedChunk

# "1. ", "2) ", "- ", "* ", "• " at the start of a line.
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


class MultiQueryPipeline(NaiveRAGPipeline):
    """Multi-query retrieval (LangChain MultiQueryRetriever; RAG-Fusion, Rackauckas 2024).

    One wording of a question can miss passages phrased differently, so the LLM writes
    `num_queries` rewrites, each is searched, and the result lists are merged by reciprocal rank
    fusion. The original question is always searched too, and the reranker (if any) judges
    relevance to the original question.
    """

    def __init__(self, *args: Any, num_queries: int = 3, rrf_k: int = 60, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_queries = max(0, num_queries)
        self.rrf_k = rrf_k

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"num_queries": config.pipeline.num_queries, "rrf_k": config.retriever.rrf_k}

    def search_queries(self, question: str) -> list[str]:
        """The original question followed by up to `num_queries` distinct LLM rewrites."""
        if self.num_queries == 0:
            return [question]
        raw = self.generator.generate(
            question, system=MULTI_QUERY_SYSTEM_PROMPT.format(n=self.num_queries)
        )
        return self._parse_queries(raw, question, self.num_queries)

    @staticmethod
    def _parse_queries(raw: str, question: str, limit: int) -> list[str]:
        queries = [question]
        seen = {question.strip().casefold()}
        for line in raw.splitlines():
            query = _LIST_MARKER.sub("", line).strip().strip('"').strip()
            if query and query.casefold() not in seen:
                seen.add(query.casefold())
                queries.append(query)
            if len(queries) > limit:
                break
        return queries

    def retrieve_for(self, question: str) -> list[RetrievedChunk]:
        return self.retrieve_queries(self.search_queries(question), question)

    def retrieve_queries(self, queries: list[str], question: str) -> list[RetrievedChunk]:
        """Search with each query, fuse the lists by rank, then rerank against `question`."""
        if len(queries) == 1:
            return self.retrieve(queries[0], question)
        k = max(self.fetch_k, self.top_k) if self.reranker is not None else self.top_k * 2
        fused = reciprocal_rank_fusion(
            [self.retriever.retrieve(q, k) for q in queries], rrf_k=self.rrf_k
        )
        if self.reranker is None:
            return fused[: self.top_k]
        return self.reranker.rerank(question, fused[:k], self.top_k)

    def run(self, question: str) -> RAGResponse:
        queries = self.search_queries(question)
        contexts = self.retrieve_queries(queries, question)
        answer = "".join(self.stream_answer(question, contexts))
        return RAGResponse(
            question=question, answer=answer, contexts=contexts, metadata={"queries": queries}
        )
