from pathlib import Path

import pytest

from easyrag import Config, Document, NaiveRAGPipeline
from easyrag.cli import main
from easyrag.evaluation import EvalExample, LLMJudge, evaluate
from easyrag.evaluation.citations import (
    citation_validity,
    extract_claims,
    invalid_citations,
    parse_markers,
)
from easyrag.generators import BaseGenerator
from easyrag.schema import Chunk, RetrievedChunk


class _Generator(BaseGenerator):
    def __init__(self, reply: str | list[str]) -> None:
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _contexts(*texts: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(Chunk(t, "d", {"chunk_index": i}), 1.0) for i, t in enumerate(texts)]


def test_parse_markers() -> None:
    assert parse_markers("A [1]. B [2][3]. C [1, 4]. D [2-4]. E [x] [].") == [
        1,
        2,
        3,
        1,
        4,
        2,
        3,
        4,
    ]


def test_extract_claims_attaches_trailing_markers_to_the_sentence() -> None:
    claims = extract_claims("Paris is the capital [1]. It is large. [2]\nBerlin too [3].")
    assert [(c.text, c.citations) for c in claims] == [
        ("Paris is the capital.", [1]),
        ("It is large.", [2]),
        ("Berlin too.", [3]),
    ]


def test_validity_and_invalid_citations() -> None:
    assert citation_validity("No citations here.", 2) is None
    assert citation_validity("A [1]. B [2].", 2) == 1.0
    assert citation_validity("A [1]. B [3]. C [0].", 2) == pytest.approx(1 / 3)
    assert invalid_citations("A [3]. B [3][0]. C [1].", 2) == [0, 3]


def test_judge_citation_support_checks_each_cited_sentence() -> None:
    generator = _Generator(['{"score": 1, "reason": "ok"}', '{"score": 0, "reason": "no"}'])
    judge = LLMJudge(generator)
    contexts = _contexts("Paris is the capital of France.", "Berlin is in Germany.")

    score, reason = judge.citation_support(
        "Paris is the capital [1]. Berlin is in Spain [2]. Uncited.", contexts
    ) or (None, "")

    assert score == 0.5
    assert "Berlin is in Spain" in reason
    assert "Paris is the capital of France." in generator.prompts[0]
    assert "Berlin is in Germany." in generator.prompts[1]
    assert "Paris" not in generator.prompts[1]  # only the cited passage is shown


def test_judge_citation_support_out_of_range_is_unsupported_without_a_call() -> None:
    generator = _Generator('{"score": 1, "reason": "ok"}')
    result = LLMJudge(generator).citation_support("Claim [5].", _contexts("x"))
    assert result == (0.0, "unsupported: Claim.")
    assert generator.prompts == []
    assert LLMJudge(generator).citation_support("Nothing cited.", _contexts("x")) is None


def test_evaluate_reports_citation_metrics(sample_documents: list[Document]) -> None:
    pipeline = NaiveRAGPipeline.from_config(
        Config.from_dict({"retriever": {"top_k": 2}}), generator=_Generator("Paris [1] [9].")
    )
    pipeline.index(sample_documents)
    judge = LLMJudge(_Generator('{"score": 1, "reason": "ok"}'))

    report = evaluate(pipeline, [EvalExample("capital of France?", answers=["Paris"])], judge=judge)

    scores = report.results[0].scores
    assert scores["citation_valid"] == 0.5
    assert scores["citation_support"] == 1.0  # [1] supports it; [9] is ignored once [1] exists


def test_cli_warns_about_citations_to_missing_passages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "france.txt").write_text("Paris is the capital of France.", encoding="utf-8")
    monkeypatch.setattr(
        "easyrag.factory.EchoGenerator", lambda: _Generator("Paris [1], see also [4].")
    )

    main(["ask", "--docs", str(docs), "Capital of France?"])

    out = capsys.readouterr().out
    assert "Warning: the answer cites [4], but only 1 passages were given." in out
