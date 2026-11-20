"""Regression tests for failure modes: interrupted saves, bad config values, bad input files."""

from pathlib import Path
from typing import Any

import pytest

from easyrag import Config, Document, NaiveRAGPipeline

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
