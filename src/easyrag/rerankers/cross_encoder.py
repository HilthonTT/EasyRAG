from __future__ import annotations

from easyrag.rerankers.base import BaseReranker
from easyrag.schema import RetrievedChunk

DEFAULT_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker(BaseReranker):
    """Reads query and passage together, so it is far more precise than embeddings but too slow
    to run over the whole corpus: retrieve ~20 candidates cheaply, rerank, keep the best few."""

    def __init__(
        self, model: str = DEFAULT_MODEL, batch_size: int = 16, device: str | None = None
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run: uv sync --extra huggingface"
            ) from e

        self.model = CrossEncoder(model, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not results:
            return []
        pairs = [(query, r.chunk.content) for r in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        rescored = [RetrievedChunk(r.chunk, float(s)) for r, s in zip(results, scores, strict=True)]
        return sorted(rescored, key=lambda r: r.score, reverse=True)[:top_k]
