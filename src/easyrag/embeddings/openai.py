from __future__ import annotations

from easyrag.embeddings.base import BaseEmbeddings

DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 256) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai is not installed. Run: uv sync --extra openai") from e

        self.client = openai.OpenAI()
        self.model = model
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") or " " for t in texts[start : start + self.batch_size]]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors
