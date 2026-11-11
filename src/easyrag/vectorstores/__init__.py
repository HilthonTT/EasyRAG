"""Vector stores: index chunk embeddings and run similarity search."""

from easyrag.vectorstores.base import BaseVectorStore
from easyrag.vectorstores.in_memory import InMemoryVectorStore

# FaissVectorStore lives in easyrag.vectorstores.faiss and needs the faiss extra.
__all__ = ["BaseVectorStore", "InMemoryVectorStore"]
