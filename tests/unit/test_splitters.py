import pytest

from easyrag.schema import Document
from easyrag.splitters import CharacterSplitter


def test_split_text_respects_chunk_size() -> None:
    splitter = CharacterSplitter(chunk_size=10, chunk_overlap=2)
    chunks = splitter.split_text("abcdefghij" * 3)
    assert all(len(c) <= 10 for c in chunks)
    assert len(chunks) == 4


def test_split_documents_keeps_metadata() -> None:
    doc = Document(content="hello world " * 20, metadata={"source": "a.txt"})
    chunks = CharacterSplitter(chunk_size=50, chunk_overlap=10).split_documents([doc])
    assert chunks
    assert all(c.document_id == doc.id for c in chunks)
    assert chunks[0].metadata == {"source": "a.txt", "chunk_index": 0}


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        CharacterSplitter(chunk_size=10, chunk_overlap=10)


def test_no_duplicate_tail_chunk() -> None:
    # The first window already covers the whole text, so there must not be a second one.
    assert CharacterSplitter(chunk_size=10, chunk_overlap=2).split_text("x" * 9) == ["x" * 9]
