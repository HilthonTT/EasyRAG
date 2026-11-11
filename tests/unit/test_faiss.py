from pathlib import Path

import numpy as np
import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.factory import build_vectorstore
from easyrag.schema import Chunk
from easyrag.vectorstores import InMemoryVectorStore

pytest.importorskip("faiss")

from easyrag.vectorstores.faiss import FaissVectorStore  # noqa: E402

RNG = np.random.default_rng(0)
CHUNKS = [Chunk(f"text {i}", "d", {"chunk_index": i}) for i in range(40)]
VECTORS = RNG.normal(size=(40, 8)).tolist()
QUERY = RNG.normal(size=8).tolist()
BY_ID = {c.id: c for c in CHUNKS}


def _ids(store: object, k: int = 5) -> list[str]:
    return [r.chunk.id for r in store.search(QUERY, k)]  # type: ignore[attr-defined]


def _reference() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(CHUNKS, VECTORS)
    return store


@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_matches_exact_search_and_scores(index_type: str) -> None:
    store = FaissVectorStore(index_type)
    store.add(CHUNKS[:20], VECTORS[:20])
    store.add(CHUNKS[20:], VECTORS[20:])  # adding in blocks keeps positions aligned
    assert _ids(store) == _ids(_reference())
    expected = [round(r.score, 5) for r in _reference().search(QUERY, 5)]
    assert [round(r.score, 5) for r in store.search(QUERY, 5)] == expected
    assert len(store.search(QUERY, 100)) == 40


@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_delete_keeps_the_rest_searchable(index_type: str) -> None:
    store, reference = FaissVectorStore(index_type), _reference()
    store.add(CHUNKS, VECTORS)
    gone = [c.id for c in CHUNKS[::3]]

    assert store.delete(gone) == reference.delete(gone) == 14
    assert store.delete(["unknown"]) == 0
    assert len(store) == 26
    assert _ids(store, 10) == _ids(reference, 10)


@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_save_load_round_trip_reuses_the_index(index_type: str, tmp_path: Path) -> None:
    store = FaissVectorStore(index_type)
    store.add(CHUNKS, VECTORS)
    path = tmp_path / "vectors.npz"
    store.save(path)

    loaded = FaissVectorStore(index_type)
    loaded.load(path, BY_ID)
    assert _ids(loaded) == _ids(store)
    assert loaded._index.ntotal == 40


def test_files_load_across_store_types(tmp_path: Path) -> None:
    faiss_file, memory_file = tmp_path / "f.npz", tmp_path / "m.npz"
    faiss_store = FaissVectorStore("hnsw")
    faiss_store.add(CHUNKS, VECTORS)
    faiss_store.save(faiss_file)
    _reference().save(memory_file)

    memory = InMemoryVectorStore()
    memory.load(faiss_file, BY_ID)
    flat = FaissVectorStore("flat")
    flat.load(faiss_file, BY_ID)  # hnsw file into a flat store: re-added from its vectors
    from_memory = FaissVectorStore()
    from_memory.load(memory_file, BY_ID)

    assert _ids(memory) == _ids(flat) == _ids(from_memory) == _ids(_reference())


def test_empty_store(tmp_path: Path) -> None:
    store = FaissVectorStore()
    assert store.search(QUERY, 3) == []
    store.save(tmp_path / "v.npz")
    loaded = FaissVectorStore()
    loaded.load(tmp_path / "v.npz", {})
    assert len(loaded) == 0


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="index_type"):
        FaissVectorStore("ivf")
    store = FaissVectorStore()
    store.add(CHUNKS[:1], VECTORS[:1])
    with pytest.raises(ValueError, match="does not match"):
        store.add(CHUNKS[1:2], [[1.0, 0.0]])


def test_pipeline_with_faiss_store_saves_and_loads(
    sample_documents: list[Document], tmp_path: Path
) -> None:
    config = Config.from_dict({"vectorstore": {"provider": "faiss", "index": "hnsw"}})
    assert isinstance(build_vectorstore(config.vectorstore), FaissVectorStore)
    pipeline = NaiveRAGPipeline.from_config(config)
    pipeline.index(sample_documents)
    pipeline.save(tmp_path)

    loaded = NaiveRAGPipeline.from_config(config)
    loaded.load(tmp_path)
    assert loaded.retrieve("capital of France")[0].chunk.content.startswith("Paris")


def test_unknown_vectorstore_provider() -> None:
    with pytest.raises(ValueError, match="Unknown vectorstore.provider"):
        build_vectorstore(Config.from_dict({"vectorstore": {"provider": "pinecone"}}).vectorstore)
