import pytest

from easyrag.schema import Document
from easyrag.splitters import CharacterSplitter, RecursiveCharacterSplitter


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


# RecursiveCharacterSplitter (the default splitter)

SENTENCES = " ".join(f"Sentence number {i} is here." for i in range(20))


def test_recursive_respects_chunk_size_and_keeps_all_words() -> None:
    chunks = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=0).split_text(SENTENCES)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == SENTENCES.split()


def test_recursive_prefers_paragraph_boundaries() -> None:
    text = "First paragraph is short.\n\nSecond paragraph is short too."
    splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=0)
    assert splitter.split_text(text) == [
        "First paragraph is short.",
        "Second paragraph is short too.",
    ]


def test_recursive_splits_on_sentences_not_mid_word() -> None:
    chunks = RecursiveCharacterSplitter(chunk_size=60, chunk_overlap=0).split_text(SENTENCES)
    assert all(c.endswith(".") for c in chunks)


def test_recursive_overlap_repeats_trailing_pieces() -> None:
    words = " ".join(f"w{i:02d}" for i in range(30))  # pieces are "w00 ": 4 chars each
    chunks = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=12).split_text(words)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.split()[-3:] == current.split()[:3]  # 3 pieces fill the 12-char overlap


def test_recursive_overlap_is_whole_pieces_only() -> None:
    # Each sentence is longer than chunk_overlap, so none can be carried over as overlap.
    chunks = RecursiveCharacterSplitter(chunk_size=60, chunk_overlap=20).split_text(SENTENCES)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert not set(previous.split(".")) & set(current.split(".")) - {""}


def test_recursive_falls_back_to_characters_for_long_tokens() -> None:
    chunks = RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=0).split_text("x" * 25)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


def test_recursive_drops_whitespace_only_chunks() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
    assert splitter.split_text("\n\n   \n\nhello\n\n\n\n") == ["hello"]
