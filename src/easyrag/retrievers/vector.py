from __future__ import annotations

from easyrag.embeddings.base import BaseEmbeddings
from easyrag.retrievers.base import BaseRetriever
from easyrag.schema import Chunk, RetrievedChunk
from easyrag.vectorstores.base import BaseVectorStore


class VectorRetriever(BaseRetriever):
    """Dense retrieval: embed the query and search a vector store."""

    def __init__(
        self, embeddings: BaseEmbeddings, vectorstore: BaseVectorStore, top_k: int = 4
    ) -> None:
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.top_k = top_k

    def add_chunks(self, chunks: list[Chunk]) -> None:
        vectors = self.embeddings.embed_documents([c.content for c in chunks])
        self.vectorstore.add(chunks, vectors)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_vector = self.embeddings.embed_query(query)
        return self.vectorstore.search(query_vector, top_k or self.top_k)
