import json
from pathlib import Path

import numpy as np
import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.schema import Chunk
from easyrag.vectorstores import InMemoryVectorStore

CHUNKS = [Chunk(f"text {i}", "d", {"chunk_index": i}) for i in range(3)]
VECTORS = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(CHUNKS, VECTORS)
    return store


def _ranking(store: InMemoryVectorStore, query: list[float]) -> list[tuple[str, float]]:
    return [(r.chunk.id, round(r.score, 6)) for r in store.search(query, 3)]


def test_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npz"
    _store().save(path)
    assert path.exists()  # written under the exact name, no extra ".npz" appended

    loaded = InMemoryVectorStore()
    loaded.load(path, {c.id: c for c in CHUNKS})
    assert _ranking(loaded, [1.0, 0.2]) == _ranking(_store(), [1.0, 0.2])


def test_empty_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npz"
    InMemoryVectorStore().save(path)
    loaded = InMemoryVectorStore()
    loaded.load(path, {})
    assert len(loaded) == 0
    assert loaded.search([1.0, 0.0], 3) == []


def test_loads_version_1_json(tmp_path: Path) -> None:
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps({"ids": [c.id for c in CHUNKS], "vectors": VECTORS}))
    loaded = InMemoryVectorStore()
    loaded.load(path, {c.id: c for c in CHUNKS})
    assert _ranking(loaded, [0.0, 1.0]) == _ranking(_store(), [0.0, 1.0])


def test_adds_after_search_are_searchable() -> None:
    store = InMemoryVectorStore()
    store.add(CHUNKS[:1], VECTORS[:1])
    assert [r.chunk.id for r in store.search([0.0, 1.0], 1)] == [CHUNKS[0].id]
    store.add(CHUNKS[1:], VECTORS[1:])
    assert [r.chunk.id for r in store.search([0.0, 1.0], 1)] == [CHUNKS[1].id]
    assert len(store) == 3


def test_vectors_are_normalized_and_zero_vectors_score_zero() -> None:
    store = InMemoryVectorStore()
    store.add(CHUNKS[:2], [[0.0, 5.0], [0.0, 0.0]])
    scores = {r.chunk.id: r.score for r in store.search([0.0, 3.0], 2)}
    assert scores == pytest.approx({CHUNKS[0].id: 1.0, CHUNKS[1].id: 0.0})


def test_rejects_mismatched_dimensions() -> None:
    store = _store()
    with pytest.raises(ValueError, match="does not match"):
        store.add(CHUNKS[:1], [[1.0, 0.0, 0.0]])


def test_pipeline_saves_npz_and_still_loads_a_version_1_index(
    sample_documents: list[Document], tmp_path: Path
) -> None:
    pipeline = NaiveRAGPipeline.from_config(Config.from_dict({"retriever": {"method": "vector"}}))
    pipeline.index(sample_documents)
    pipeline.save(tmp_path)
    assert (tmp_path / "vectors_0.npz").exists()
    expected = [r.chunk.id for r in pipeline.retrieve("capital of France")]

    # Rewrite the index as format version 1 (vectors as JSON) and load it.
    with np.load(tmp_path / "vectors_0.npz") as data:
        legacy = {"ids": data["ids"].tolist(), "vectors": data["vectors"].tolist()}
    (tmp_path / "vectors_0.npz").unlink()
    (tmp_path / "vectors_0.json").write_text(json.dumps(legacy))
    old = NaiveRAGPipeline.from_config(Config.from_dict({"retriever": {"method": "vector"}}))
    old.load(tmp_path)
    assert [r.chunk.id for r in old.retrieve("capital of France")] == expected

    # Saving again upgrades it: the stale JSON file is replaced by .npz.
    old.save(tmp_path)
    assert sorted(p.name for p in tmp_path.glob("vectors_*")) == ["vectors_0.npz"]


def test_delete_removes_chunks_and_their_vectors() -> None:
    store = _store()
    assert store.delete([CHUNKS[1].id, "unknown"]) == 1
    assert len(store) == 2
    assert [r.chunk.id for r in store.search([0.0, 1.0], 3)] == [CHUNKS[2].id, CHUNKS[0].id]
    assert store.delete([c.id for c in CHUNKS]) == 2
    assert store.search([0.0, 1.0], 3) == []
    store.add(CHUNKS[:1], VECTORS[:1])  # usable again after emptying
    assert len(store) == 1
