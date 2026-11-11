"""Updating a saved index: re-index changed files, remove deleted ones."""

from pathlib import Path

import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.cli import main
from easyrag.embeddings.base import BaseEmbeddings
from easyrag.embeddings.fake import FakeEmbeddings
from easyrag.retrievers import BM25Retriever, HybridRetriever, VectorRetriever


class _CountingEmbeddings(FakeEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed_documents(texts)


def _pipeline(embeddings: BaseEmbeddings | None = None) -> NaiveRAGPipeline:
    retriever = HybridRetriever([BM25Retriever(), VectorRetriever(embeddings or FakeEmbeddings())])
    return NaiveRAGPipeline.from_config(Config(), retriever=retriever)


def _sources(pipeline: NaiveRAGPipeline, query: str) -> list[str]:
    return [r.chunk.source for r in pipeline.retrieve(query)]


def test_update_replaces_changed_files_and_skips_unchanged_ones() -> None:
    embeddings = _CountingEmbeddings()
    pipeline = _pipeline(embeddings)
    pipeline.index(
        [
            Document("Paris is the capital of France.", {"source": "a.txt"}),
            Document("Berlin is the capital of Germany.", {"source": "b.txt"}),
        ]
    )
    embeddings.embedded.clear()

    added, removed = pipeline.update(
        [
            Document("Lyon is the capital of France.", {"source": "a.txt"}),  # changed
            Document("Berlin is the capital of Germany.", {"source": "b.txt"}),  # unchanged
            Document("Rome is the capital of Italy.", {"source": "c.txt"}),  # new
        ]
    )

    assert (added, removed) == (2, 1)
    assert embeddings.embedded == [
        "Lyon is the capital of France.",
        "Rome is the capital of Italy.",
    ]
    contents = {c.content for c in pipeline.chunks}
    assert "Paris is the capital of France." not in contents
    for retriever in pipeline.retriever.retrievers:  # both indexes dropped the old chunk
        results = retriever.retrieve("Paris", 10)
        assert all("Paris" not in r.chunk.content for r in results)


def test_remove_matches_sources_however_they_are_spelled(tmp_path: Path) -> None:
    pipeline = _pipeline()
    a = tmp_path / "docs" / "a.txt"
    pipeline.index(
        [
            Document("Paris is the capital of France.", {"source": str(a)}),
            Document("Berlin is the capital of Germany.", {"source": "b.txt"}),
        ]
    )

    assert pipeline.remove([str(tmp_path / "docs" / "." / "a.txt")]) == 1
    assert pipeline.remove(["missing.txt"]) == 0
    assert [c.source for c in pipeline.chunks] == ["b.txt"]
    assert _sources(pipeline, "Paris capital France") == ["b.txt"]


def test_remove_missing_only_touches_deleted_files_under_root(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    kept = docs / "kept.txt"
    kept.write_text("x")
    pipeline = _pipeline()
    pipeline.index(
        [
            Document("kept file", {"source": str(kept)}),
            Document("deleted file", {"source": str(docs / "gone.txt")}),
            Document("deleted elsewhere", {"source": str(tmp_path / "docs2" / "gone.txt")}),
            Document("indexed from a string"),
        ]
    )

    assert pipeline.remove_missing(docs) == 1
    assert {c.content for c in pipeline.chunks} == {
        "kept file",
        "deleted elsewhere",
        "indexed from a string",
    }


def test_bm25_remove_rebuilds_postings() -> None:
    pipeline = NaiveRAGPipeline.from_config(Config.from_dict({"retriever": {"method": "bm25"}}))
    pipeline.index(
        [Document("alpha beta", {"source": "1"}), Document("beta gamma", {"source": "2"})]
    )
    pipeline.remove(["1"])
    assert [r.chunk.content for r in pipeline.retrieve("alpha beta")] == ["beta gamma"]
    pipeline.index([Document("alpha delta", {"source": "3"})])
    assert pipeline.retrieve("alpha")[0].chunk.content == "alpha delta"


def test_cli_index_update_and_remove(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Paris is the capital of France.", encoding="utf-8")
    (docs / "b.txt").write_text("Berlin is the capital of Germany.", encoding="utf-8")
    index_dir = str(tmp_path / "index")
    main(["--index-dir", index_dir, "index", str(docs)])

    (docs / "a.txt").write_text("Lyon is the capital of France.", encoding="utf-8")
    (docs / "b.txt").unlink()
    (docs / "c.txt").write_text("Rome is the capital of Italy.", encoding="utf-8")
    main(["--index-dir", index_dir, "index", "--update", str(docs)])
    assert "2 chunks added, 2 removed, 2 total" in capsys.readouterr().out

    reloaded = NaiveRAGPipeline.from_config(Config())
    reloaded.load(index_dir)
    assert sorted(c.content for c in reloaded.chunks) == [
        "Lyon is the capital of France.",
        "Rome is the capital of Italy.",
    ]

    main(["--index-dir", index_dir, "remove", str(docs / "c.txt")])
    assert "Removed 1 chunks; 1 left" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="No indexed chunks come from"):
        main(["--index-dir", index_dir, "remove", str(docs / "c.txt")])


def test_cli_index_update_without_a_saved_index_creates_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Paris is the capital of France.", encoding="utf-8")
    index_dir = tmp_path / "index"

    main(["--index-dir", str(index_dir), "index", "--update", str(docs)])

    assert "1 chunks added, 0 removed, 1 total" in capsys.readouterr().out
    assert (index_dir / "manifest.json").exists()
