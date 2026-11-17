from pathlib import Path

import numpy as np
import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.cli import main
from easyrag.embeddings import FakeEmbeddings
from easyrag.filters import compile_filter, parse_filter_args
from easyrag.pipelines.multi_query import MultiQueryPipeline
from easyrag.retrievers import BM25Retriever, HybridRetriever, VectorRetriever
from easyrag.schema import Chunk
from easyrag.vectorstores import InMemoryVectorStore

DOCS = [
    Document("Paris is the capital of France.", {"source": "docs/europe/france.txt"}),
    Document("Berlin is the capital of Germany.", {"source": "docs/europe/germany.txt"}),
    Document("Tokyo is the capital of Japan.", {"source": "docs/asia/japan.txt"}),
    Document("Lyon is a large city in France.", {"source": "docs/europe/france.pdf", "page": 2}),
]


def _chunk(**metadata: object) -> Chunk:
    return Chunk("text", "d", dict(metadata))


def _sources(results: list) -> set[str]:  # type: ignore[type-arg]
    return {r.chunk.source for r in results}


# --- filter semantics ---------------------------------------------------------------------


def test_empty_or_missing_filter_means_no_filter() -> None:
    assert compile_filter(None) is None
    assert compile_filter({}) is None


def test_callable_filter_is_used_as_is() -> None:
    def predicate(chunk: Chunk) -> bool:
        return True

    assert compile_filter(predicate) is predicate


def test_mapping_filter_rules() -> None:
    match = compile_filter({"source": "*europe/*", "page": [1, 2]})
    assert match is not None
    assert match(_chunk(source="docs/europe/france.pdf", page=2))
    assert not match(_chunk(source="docs/europe/france.pdf", page=3))
    assert not match(_chunk(source="docs/asia/japan.pdf", page=2))
    assert not match(_chunk(source="docs/europe/france.pdf"))  # missing key


def test_cli_strings_match_numbers_and_windows_paths() -> None:
    match = compile_filter({"page": "3", "source": "docs/*.md"})
    assert match is not None
    assert match(_chunk(page=3, source="docs\\a.md"))


def test_exact_value_without_glob_chars() -> None:
    match = compile_filter({"lang": "en"})
    assert match is not None
    assert match(_chunk(lang="en"))
    assert not match(_chunk(lang="english"))


def test_parse_filter_args_groups_repeated_keys() -> None:
    parsed = parse_filter_args(["source=*.md", "page=1", "page=2", "page=3", "q=a=b"])
    assert parsed == {"source": "*.md", "page": ["1", "2", "3"], "q": "a=b"}


@pytest.mark.parametrize("bad", ["source", "=x"])
def test_parse_filter_args_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_filter_args([bad])


# --- retrievers ---------------------------------------------------------------------------


def _chunks() -> list[Chunk]:
    return [Chunk(d.content, d.id, d.metadata) for d in DOCS]


def test_vector_retriever_filter() -> None:
    retriever = VectorRetriever(FakeEmbeddings(), InMemoryVectorStore())
    retriever.add_chunks(_chunks())

    results = retriever.retrieve("capital of France", 10, filter={"source": "*asia/*"})

    assert _sources(results) == {"docs/asia/japan.txt"}
    assert retriever.retrieve("France", 10, filter={"source": "nowhere"}) == []


def test_bm25_filter() -> None:
    retriever = BM25Retriever()
    retriever.add_chunks(_chunks())

    results = retriever.retrieve("France", 10, filter={"source": "*.txt"})

    assert _sources(results) == {"docs/europe/france.txt"}  # the PDF also says France


def test_hybrid_filter_reaches_every_retriever() -> None:
    hybrid = HybridRetriever([BM25Retriever(), VectorRetriever(FakeEmbeddings())])
    hybrid.add_chunks(_chunks())

    results = hybrid.retrieve("capital", 10, filter={"page": 2})

    assert _sources(results) == {"docs/europe/france.pdf"}


def test_in_memory_filter_ranks_within_matches() -> None:
    rng = np.random.default_rng(0)
    chunks = [Chunk(f"t{i}", "d", {"even": i % 2 == 0}) for i in range(30)]
    vectors = rng.normal(size=(30, 6)).tolist()
    query = rng.normal(size=6).tolist()
    store = InMemoryVectorStore()
    store.add(chunks, vectors)

    filtered = store.search(query, 5, filter=lambda c: bool(c.metadata["even"]))
    expected = [r for r in store.search(query, 30) if r.chunk.metadata["even"]][:5]

    assert [r.chunk.id for r in filtered] == [r.chunk.id for r in expected]
    assert [r.score for r in filtered] == pytest.approx([r.score for r in expected])


@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_faiss_filter_matches_brute_force(index_type: str) -> None:
    pytest.importorskip("faiss")
    from easyrag.vectorstores.faiss import FaissVectorStore

    rng = np.random.default_rng(1)
    chunks = [Chunk(f"t{i}", "d", {"group": i % 7}) for i in range(200)]
    vectors = rng.normal(size=(200, 8)).tolist()
    query = rng.normal(size=8).tolist()
    store = FaissVectorStore(index_type)
    store.add(chunks, vectors)
    reference = InMemoryVectorStore()
    reference.add(chunks, vectors)

    def rare(c: Chunk) -> bool:  # 29 of 200 chunks
        return bool(c.metadata["group"] == 3)

    got = store.search(query, 5, filter=rare)
    want = reference.search(query, 5, filter=rare)

    assert [r.chunk.id for r in got] == [r.chunk.id for r in want]
    assert store.search(query, 5, filter=lambda c: False) == []


# --- pipeline and CLI ---------------------------------------------------------------------


def test_pipeline_filter_applies_to_every_search() -> None:
    pipeline = NaiveRAGPipeline.from_config(Config.from_dict({"retriever": {"top_k": 4}}))
    pipeline.index(DOCS)

    pipeline.filter = {"source": "*europe/*"}
    response = pipeline.run("What is the capital of Japan?")

    assert response.contexts
    assert all("/europe/" in s for s in _sources(response.contexts))

    pipeline.filter = None
    assert "docs/asia/japan.txt" in _sources(pipeline.run("capital of Japan").contexts)


def test_multi_query_pipeline_uses_filter() -> None:
    pipeline = MultiQueryPipeline.from_config(filter={"source": "*asia/*"})
    pipeline.index(DOCS)

    assert _sources(pipeline.run("capital of France").contexts) == {"docs/asia/japan.txt"}


def test_cli_ask_with_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs = tmp_path / "docs"
    (docs / "a").mkdir(parents=True)
    (docs / "b").mkdir()
    (docs / "a" / "france.txt").write_text("Paris is the capital of France.", encoding="utf-8")
    (docs / "b" / "france.txt").write_text("France borders Spain.", encoding="utf-8")
    index_dir = str(tmp_path / "index")
    main(["--index-dir", index_dir, "index", str(docs)])
    capsys.readouterr()

    main(["--index-dir", index_dir, "ask", "France", "--show-context", "--filter", "source=*/b/*"])

    captured = capsys.readouterr()
    assert "Filter source=*/b/*: 1 of 2 chunks" in captured.err
    assert "France borders Spain" in captured.out
    assert "Paris" not in captured.out


def test_cli_filter_matching_nothing_exits(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("Some text.", encoding="utf-8")

    with pytest.raises(SystemExit, match="No indexed chunks match"):
        main(["ask", "q", "--docs", str(tmp_path), "--filter", "source=*.pdf"])


def test_cli_rejects_malformed_filter(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "q", "--filter", "nonsense"])

    assert exc.value.code == 2
    assert "KEY=VALUE" in capsys.readouterr().err
