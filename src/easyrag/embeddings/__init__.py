"""Embedding models: turn text into vectors."""

from easyrag.embeddings.base import BaseEmbeddings
from easyrag.embeddings.fake import FakeEmbeddings
from easyrag.embeddings.sentence_transformer import SentenceTransformerEmbeddings

__all__ = ["BaseEmbeddings", "FakeEmbeddings", "SentenceTransformerEmbeddings"]
