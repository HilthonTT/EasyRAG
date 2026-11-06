from easyrag.schema import Chunk, Document
from easyrag.splitters import RecursiveCharacterSplitter


def test_ids_are_deterministic() -> None:
    assert Document("x", {"source": "a.txt"}).id == Document("x", {"source": "a.txt"}).id
    assert Chunk("x", "d").id == Chunk("x", "d").id


def test_ids_differ_by_content_source_and_position() -> None:
    base = Document("x", {"source": "a.txt"}).id
    assert Document("y", {"source": "a.txt"}).id != base
    assert Document("x", {"source": "b.txt"}).id != base
    assert Document("x", {"source": "a.txt", "page": 2}).id != base
    assert Chunk("x", "d1").id != Chunk("x", "d2").id


def test_repeated_paragraph_gets_distinct_chunk_ids() -> None:
    doc = Document("Same paragraph.\n\nSame paragraph.", {"source": "a.txt"})
    chunks = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0).split_documents([doc])
    assert [c.content for c in chunks] == ["Same paragraph.", "Same paragraph."]
    assert chunks[0].id != chunks[1].id


def test_explicit_id_and_round_trip_are_kept() -> None:
    assert Document("x", id="mine").id == "mine"
    chunk = Chunk("x", "d", {"chunk_index": 0})
    assert Chunk.from_dict(chunk.to_dict()) == chunk
