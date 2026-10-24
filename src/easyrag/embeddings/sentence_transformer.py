from __future__ import annotations

from typing import Any

from easyrag.embeddings.base import BaseEmbeddings

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def default_prefixes(model: str) -> tuple[str, str]:
    """(query_prefix, passage_prefix) recommended by the model card."""
    name = model.lower()
    if "e5" in name:
        return "query: ", "passage"
    if "bge" in name and "-en" in name and "m3" not in name:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


class SentenceTransformerEmbeddings(BaseEmbeddings):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        **model_kwargs: Any,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run: uv sync --extra huggingface"
            ) from e

        self.model_name = model
        default_query, default_passage = default_prefixes(model)
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.passage_prefix = default_passage if passage_prefix is None else passage_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model, device=device, **model_kwargs)

    @property
    def name(self) -> str:
        return f"sentence_transformer:{self.model_name}"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine == dot product, scores comparable across queries
            show_progress_bar=len(texts) > 256,
        )
        return [list(map(float, v)) for v in vectors]

    def embed_documents(self, texts):
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text):
        return self._encode([self.query_prefix + text])[0]
