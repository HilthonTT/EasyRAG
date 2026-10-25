from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from easyrag.retrievers.base import BaseRetriever
from easyrag.schema import Chunk, RetrievedChunk
from easyrag.text import tokenize


class BM25Retriever(BaseRetriever):
    """Sparse retrieval with an inverted index. Strong on exact terms: names, codes, rare words."""

    def __init__(self, top_k: int = 4, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(top_k)
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._postings: defaultdict[str, list[int]] = defaultdict(list)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            index = len(self.chunks)
            freqs = Counter(tokenize(chunk.content))
            self.chunks.append(chunk)
            self._term_freqs.append(freqs)
            self._doc_lengths.append(sum(freqs.values()))
            for term in freqs:
                self._postings[term].append(index)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        n = len(self.chunks)
        if n == 0:
            return []

        avg_len = sum(self._doc_lengths) / n or 1.0
        scores: defaultdict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for i in postings:
                tf = self._term_freqs[i][term]
                norm = self.k1 * (1 - self.b + self.b * self._doc_lengths[i] / avg_len)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        best = heapq.nlargest(top_k or self.top_k, scores.items(), key=lambda kv: kv[1])
        return [RetrievedChunk(self.chunks[i], score) for i, score in best]
