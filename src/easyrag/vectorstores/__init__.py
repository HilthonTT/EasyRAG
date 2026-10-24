"""Vector stores: index chunk embeddings and run similarity search."""

from easyrag.vectorstores.base import BaseVectorStore
from easyrag.vectorstores.in_memory import InMemoryVectorStore

__all__ = ["BaseVectorStore", "InMemoryVectorStore"]
