from collections import defaultdict
from collections.abc import Collection, Sequence

from easyrag.retrievers.base import BaseRetriever
from easyrag.schema import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: Sequence[list[RetrievedChunk]],
    weights: Sequence[float] | None = None,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Merge ranked lists: score(chunk) = sum_i weight_i / (rrf_k + rank_i(chunk)), best first.

    Only ranks are used, so the raw scores of different retrievers or queries never need
    calibrating against each other.
    """
    weights = weights if weights is not None else [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must be one entry per result list")
    scores: defaultdict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}

    for results, weight in zip(result_lists, weights, strict=True):
        counted: set[str] = set()
        for rank, result in enumerate(results, start=1):
            if result.chunk.id in counted:  # a duplicate in one list must not add score twice
                continue
            counted.add(result.chunk.id)
            scores[result.chunk.id] += weight / (rrf_k + rank)
            chunks[result.chunk.id] = result.chunk

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [RetrievedChunk(chunks[cid], score) for cid, score in ranked]


class HybridRetriever(BaseRetriever):
    """Combine e.g. BM25 (exact terms) with vector search (meaning) by reciprocal rank fusion."""

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

    def remove_chunks(self, chunk_ids: Collection[str]) -> None:
        for retriever in self.retrievers:
            retriever.remove_chunks(chunk_ids)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        fetch_k = max(self.fetch_k or 0, top_k * 2)
        result_lists = [retriever.retrieve(query, fetch_k) for retriever in self.retrievers]
        return reciprocal_rank_fusion(result_lists, self.weights, self.rrf_k)[:top_k]
