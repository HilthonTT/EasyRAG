"""Checks that the example evaluation set stays consistent with its corpus."""

from pathlib import Path

import pytest

from easyrag.evaluation import EvalExample, load_examples
from easyrag.evaluation.evaluator import source_matches
from easyrag.evaluation.metrics import normalize_answer

EVAL_DIR = Path(__file__).resolve().parents[2] / "examples" / "eval"
DOCS = sorted(p for p in (EVAL_DIR / "docs").rglob("*.md"))
EXAMPLES = load_examples(EVAL_DIR / "questions.jsonl")


def _files_for(source: str) -> list[Path]:
    return [p for p in DOCS if source_matches(p.as_posix(), source)]


def test_eval_set_is_not_empty() -> None:
    assert len(EXAMPLES) >= 20
    assert len({e.question for e in EXAMPLES}) == len(EXAMPLES), "duplicate question"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.question[:50])
def test_each_source_names_exactly_one_file(example: EvalExample) -> None:
    assert example.sources and example.answers
    for source in example.sources:
        assert len(_files_for(source)) == 1, f"{source!r} matches {_files_for(source)}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.question[:50])
def test_an_accepted_answer_appears_in_the_sources(example: EvalExample) -> None:
    text = normalize_answer(
        " ".join(f.read_text(encoding="utf-8") for s in example.sources for f in _files_for(s))
    )
    assert any(normalize_answer(a) in text for a in example.answers), example.answers
