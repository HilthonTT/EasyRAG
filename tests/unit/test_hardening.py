"""Regression tests for failure modes: interrupted saves, bad config values, bad input files."""

import json
from pathlib import Path
from typing import Any

import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.pipelines.base import INDEX_FORMAT_VERSION

# --- index persistence ------------------------------------------------------------------------


def _vector_pipeline() -> NaiveRAGPipeline:
    config = Config()
    config.retriever.method = "hybrid"
    return NaiveRAGPipeline.from_config(config)


def test_save_leaves_no_temporary_files(sample_documents: list[Document], tmp_path: Path) -> None:
    pipeline = _vector_pipeline()
    pipeline.index(sample_documents)
    pipeline.save(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "chunks.jsonl",
        "manifest.json",
        "vectors_0.npz",
    ]


def test_interrupted_save_keeps_the_previous_index(
    sample_documents: list[Document], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _vector_pipeline()
    pipeline.index(sample_documents[:2])
    pipeline.save(tmp_path)

    pipeline.index(sample_documents[2:])
    store = pipeline._vector_retrievers()[0].vectorstore

    def crash(path: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "save", crash)
    with pytest.raises(KeyboardInterrupt):
        pipeline.save(tmp_path)

    fresh = _vector_pipeline()
    assert fresh.load(tmp_path) == 2
    assert not list(tmp_path.glob(".*.tmp"))


def test_load_re_embeds_when_vectors_do_not_match_the_chunks(
    sample_documents: list[Document], tmp_path: Path
) -> None:
    old = _vector_pipeline()
    old.index(sample_documents[:1])
    old.save(tmp_path / "old")
    new = _vector_pipeline()
    new.index(sample_documents)
    new.save(tmp_path / "new")
    # The state a crash mid-save used to leave: new chunks next to old vectors.
    (tmp_path / "new" / "vectors_0.npz").write_bytes(
        (tmp_path / "old" / "vectors_0.npz").read_bytes()
    )

    fresh = _vector_pipeline()
    assert fresh.load(tmp_path / "new") == 3
    assert len(fresh._vector_retrievers()[0].vectorstore) == 3
    assert fresh.run("What is the capital of Germany?").contexts


def test_load_rejects_a_newer_index_format(
    sample_documents: list[Document], tmp_path: Path
) -> None:
    pipeline = _vector_pipeline()
    pipeline.index(sample_documents)
    pipeline.save(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = INDEX_FORMAT_VERSION + 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="format version"):
        _vector_pipeline().load(tmp_path)


def test_save_removes_vector_files_of_retrievers_that_are_gone(
    sample_documents: list[Document], tmp_path: Path
) -> None:
    pipeline = _vector_pipeline()
    pipeline.index(sample_documents)
    pipeline.save(tmp_path)

    config = Config()
    config.retriever.method = "bm25"
    lexical = NaiveRAGPipeline.from_config(config)
    lexical.index(sample_documents)
    lexical.save(tmp_path)

    assert not list(tmp_path.glob("vectors_*"))


# --- config -----------------------------------------------------------------------------------


def test_misspelled_boolean_is_an_error_not_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EASYRAG_CACHE_LLM", "ture")
    with pytest.raises(ValueError, match="EASYRAG_CACHE_LLM"):
        Config().apply_env_overrides()


@pytest.mark.parametrize("value, expected", [("off", False), ("0", False), ("YES", True)])
def test_boolean_spellings(value: str, expected: bool) -> None:
    assert Config.from_dict({"cache": {"llm": value}}).cache.llm is expected


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"retriever": {"top_k": 0}}, "retriever.top_k"),
        ({"pipeline": {"max_iterations": -1}}, "pipeline.max_iterations"),
        ({"reranker": {"fetch_k": 0}}, "reranker.fetch_k"),
        ({"splitter": {"chunk_size": 100, "chunk_overlap": 100}}, "chunk_overlap"),
    ],
)
def test_out_of_range_values_are_rejected(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NaiveRAGPipeline.from_config(Config.from_dict(raw))
