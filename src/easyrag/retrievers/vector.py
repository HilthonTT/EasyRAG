from __future__ import annotations

from easyrag.embeddings.base import BaseEmbeddings
from easyrag.retrievers.base import BaseRetriever
from easyrag.schema import Chunk, RetrievedChunk
from easyrag.vectorstores.base import BaseVectorStore
from easyrag.vectorstores.in_memory import InMemoryVectorStore


class VectorRetriever(BaseRetriever):
    """Dense retrieval: embed the query and search a vector store."""

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        vectorstore: BaseVectorStore | None = None,
        top_k: int = 4,
    ) -> None:
        super().__init__(top_k)
        self.embeddings = embeddings
        self.vectorstore = vectorstore if vectorstore is not None else InMemoryVectorStore()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if chunks:
            vectors = self.embeddings.embed_documents([c.content for c in chunks])
            self.vectorstore.add(chunks, vectors)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.vectorstore.search(self.embeddings.embed_query(query), top_k or self.top_k)
