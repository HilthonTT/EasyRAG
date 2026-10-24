import pytest

pytest.importorskip("sentence_transformers")

from easyrag.embeddings import SentenceTransformerEmbeddings  # noqa: E402
from easyrag.retrievers import VectorRetriever  # noqa: E402
from easyrag.schema import Chunk  # noqa: E402
from easyrag.vectorstores import InMemoryVectorStore  # noqa: E402


@pytest.fixture(scope="module")
def embeddings() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")


def test_vectors_are_normalized(embeddings: SentenceTransformerEmbeddings) -> None:
    vector = embeddings.embed_query("hello world")
    assert abs(sum(x * x for x in vector) - 1.0) < 1e-4


def test_retrieves_by_meaning_not_just_words(embeddings: SentenceTransformerEmbeddings) -> None:
    retriever = VectorRetriever(embeddings, InMemoryVectorStore(), top_k=1)
    retriever.add_chunks(
        [
            Chunk("Paris is the capital of France.", "1"),
            Chunk("Python is a programming language.", "2"),
            Chunk("Bananas are rich in potassium.", "3"),
        ]
    )

    result = retriever.retrieve("Which city is France's seat of government?")[0]

    assert "Paris" in result.chunk.content
