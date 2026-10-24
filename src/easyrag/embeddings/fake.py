from __future__ import annotations

import hashlib
import math
import re

from easyrag.embeddings.base import BaseEmbeddings


class FakeEmbeddings(BaseEmbeddings):
    """Deterministic bag-of-words hashing embeddings.

    Needs no model download, so it is useful for tests and quick experiments.
    Texts that share words end up with similar vectors.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]
