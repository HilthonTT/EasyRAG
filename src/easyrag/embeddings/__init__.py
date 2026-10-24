"""Embedding models: turn text into vectors."""

from easyrag.embeddings.base import BaseEmbeddings
from easyrag.embeddings.fake import FakeEmbeddings

__all__ = ["BaseEmbeddings", "FakeEmbeddings"]
