from easyrag.embeddings import FakeEmbeddings
from easyrag.retrievers import VectorRetriever
from easyrag.schema import Document
from easyrag.splitters import CharacterSplitter
from easyrag.vectorstores import InMemoryVectorStore


def test_vector_retriever_finds_relevant_chunk(sample_documents: list[Document]) -> None:
    retriever = VectorRetriever(FakeEmbeddings(), InMemoryVectorStore(), top_k=1)
    retriever.add_chunks(CharacterSplitter().split_documents(sample_documents))

    results = retriever.retrieve("What is the capital of Germany?")

    assert len(results) == 1
    assert "Berlin" in results[0].chunk.content


def test_vectorstore_returns_results_sorted_by_score(sample_documents: list[Document]) -> None:
    retriever = VectorRetriever(FakeEmbeddings(), InMemoryVectorStore(), top_k=3)
    retriever.add_chunks(CharacterSplitter().split_documents(sample_documents))

    scores = [r.score for r in retriever.retrieve("capital of France")]

    assert scores == sorted(scores, reverse=True)
