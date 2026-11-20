"""Regression tests for failure modes: interrupted saves, bad config values, bad input files."""

import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.cli import main
from easyrag.config import load_dotenv
from easyrag.evaluation import load_examples
from easyrag.generators.ollama import OllamaGenerator
from easyrag.loaders import DirectoryLoader
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


def test_dotenv_handles_export_and_only_strips_matching_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("export EASYRAG_T1='quoted'\nEASYRAG_T2=ends-with\"\n", encoding="utf-8")
    monkeypatch.delenv("EASYRAG_T1", raising=False)
    monkeypatch.delenv("EASYRAG_T2", raising=False)

    load_dotenv(env)

    assert os.environ["EASYRAG_T1"] == "quoted"
    assert os.environ["EASYRAG_T2"] == 'ends-with"'


# --- loaders and eval sets --------------------------------------------------------------------


def test_directory_loader_skips_unreadable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "good.txt").write_text("fine", encoding="utf-8")
    (tmp_path / "bad.html").write_text("<p>x</p>", encoding="utf-8")

    def broken(self: Any) -> Any:
        raise OSError("disk error")

    monkeypatch.setattr("easyrag.loaders.html.HTMLLoader.load", broken)
    loader = DirectoryLoader(tmp_path)

    documents = loader.load()

    assert [d.content for d in documents] == ["fine"]
    assert [p.name for p, _ in loader.errors] == ["bad.html"]
    with pytest.raises(OSError):
        DirectoryLoader(tmp_path, skip_errors=False).load()


@pytest.mark.parametrize(
    "line, match",
    [
        ("not json", "line 2"),
        ('{"answers": ["x"]}', '"question"'),
        ('["a list"]', "JSON object"),
        ('{"question": "q", "answers": "x"}', "must be lists"),
    ],
)
def test_load_examples_names_the_bad_line(tmp_path: Path, line: str, match: str) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text('{"question": "ok"}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_examples(path)


def test_cli_eval_reports_a_missing_dataset_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("text", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["eval", str(tmp_path / "missing.jsonl"), "--docs", str(docs), "--retrieval-only"])
    assert exc.value.code == 2
    assert "missing.jsonl" in capsys.readouterr().err


# --- ollama -----------------------------------------------------------------------------------


def test_ollama_stream_raises_on_an_error_line(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = b'{"message": {"content": "Hel"}}\n{"error": "model ran out of memory"}\n'
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: io.BytesIO(lines))

    pieces = OllamaGenerator(max_retries=0).stream("hi")

    assert next(pieces) == "Hel"
    with pytest.raises(RuntimeError, match="out of memory"):
        next(pieces)
