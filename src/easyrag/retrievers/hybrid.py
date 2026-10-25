from collections import defaultdict

from easyrag.retrievers.base import BaseRetriever
from easyrag.schema import Chunk, RetrievedChunk


class HybridRetriever(BaseRetriever):
    """Combine e.g. BM25 (exact terms) with vector search (meaning).

    RRF uses only ranks, so the incomparable raw scores of each retriever never need calibrating:
    score(chunk) = sum_i weight_i / (rrf_k + rank_i(chunk)).
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        weights: list[float] | None = None,
        top_k: int = 4,
        rrf_k: int = 60,
        fetch_k: int | None = None,
    ) -> None:
        super().__init__(top_k)

        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must be one entry per retriever")

        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for retriever in self.retrievers:
            retriever.add_chunks(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        fetch_k = max(self.fetch_k or 0, top_k * 2)
        scores: defaultdict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}

        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            for rank, result in enumerate(retriever.retrieve(query, fetch_k), start=1):
                scores[result.chunk.id] += weight / (self.rrf_k + rank)
                chunks[result.chunk.id] = result.chunk

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        return [RetrievedChunk(chunks[cid], score) for cid, score in ranked]
