from easyrag.embeddings import FakeEmbeddings
from easyrag.retrievers import BM25Retriever, VectorRetriever
from easyrag.schema import Chunk, Document
from easyrag.splitters import CharacterSplitter
from easyrag.text import tokenize
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


def test_tokenize_strips_quotes_but_keeps_contractions() -> None:
    assert tokenize("'Paris' isn't \"Rome\"") == ["paris", "isn't", "rome"]


def test_bm25_matches_single_quoted_word() -> None:
    retriever = BM25Retriever()
    retriever.add_chunks([Chunk(content="The city 'Paris' is lovely", document_id="d")])
    assert len(retriever.retrieve("paris")) == 1


def test_rrf_counts_a_chunk_once_per_list() -> None:
    from easyrag.retrievers.hybrid import reciprocal_rank_fusion
    from easyrag.schema import RetrievedChunk

    a, b = Chunk("a", "d"), Chunk("b", "d")
    fused = reciprocal_rank_fusion(
        [[RetrievedChunk(a, 1), RetrievedChunk(a, 1)], [RetrievedChunk(b, 1)]]
    )
    assert fused[0].score == fused[1].score
