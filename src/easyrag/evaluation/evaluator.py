from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from easyrag.evaluation.citations import citation_validity
from easyrag.evaluation.judge import LLMJudge
from easyrag.evaluation.metrics import (
    contains_answer,
    exact_match,
    hit_rate,
    reciprocal_rank,
    token_f1,
)
from easyrag.generators.base import Usage, track_usage
from easyrag.pipelines.base import BasePipeline
from easyrag.prompts import format_context


@dataclass
class EvalExample:
    question: str
    answers: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    example: EvalExample
    answer: str
    retrieved_sources: list[str]
    scores: dict[str, float]
    judge_reasons: dict[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)  # LLM use answering the question
    judge_usage: Usage = field(default_factory=Usage)  # LLM use grading the answer


@dataclass
class EvalReport:
    results: list[EvalResult]

    @property
    def usage(self) -> Usage:
        """LLM calls and tokens spent answering, summed over all questions."""
        return sum((r.usage for r in self.results), Usage())

    @property
    def judge_usage(self) -> Usage:
        """LLM calls and tokens spent by the judge, summed over all questions."""
        return sum((r.judge_usage for r in self.results), Usage())

    @property
    def summary(self) -> dict[str, float]:
        # Examples without sources or answers skip those metrics: average each where present.
        totals: dict[str, list[float]] = {}
        for r in self.results:
            for k, v in r.scores.items():
                totals.setdefault(k, []).append(v)
        return {k: sum(vs) / len(vs) for k, vs in totals.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "usage": {"generation": asdict(self.usage), "judge": asdict(self.judge_usage)},
            "results": [
                {
                    "question": r.example.question,
                    "answer": r.answer,
                    "expected": r.example.answers,
                    "expected_sources": r.example.sources,
                    "retrieved_sources": r.retrieved_sources,
                    "scores": r.scores,
                    "judge_reasons": r.judge_reasons,
                    "usage": asdict(r.usage),
                    "judge_usage": asdict(r.judge_usage),
                }
                for r in self.results
            ],
        }


def load_examples(path: str | Path) -> list[EvalExample]:
    """Read a JSONL eval set. Raises ValueError naming the line of any malformed row."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
                question = data.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError('needs a non-empty "question" string')
                answers, sources = data.get("answers", []), data.get("sources", [])
                if not isinstance(answers, list) or not isinstance(sources, list):
                    raise ValueError('"answers" and "sources" must be lists')
            except ValueError as e:  # json.JSONDecodeError is a ValueError too
                raise ValueError(f"{path}, line {number}: {e}") from None
            examples.append(
                EvalExample(
                    question=question,
                    answers=[str(a) for a in answers],
                    sources=[str(s) for s in sources],
                )
            )
    return examples


def source_matches(retrieved: str, expected: str) -> bool:
    """True if `expected` names the same file as `retrieved`, matching trailing path parts.

    "intro.md" matches any file called intro.md; "guide/intro.md" only the one under guide/.
    Backslashes are treated as separators, so eval sets work across Windows and POSIX.
    """
    want = PurePosixPath(expected.replace("\\", "/")).parts
    have = PurePosixPath(retrieved.replace("\\", "/")).parts
    want = tuple(p for p in want if p != ".")
    return bool(want) and have[-len(want) :] == want


def _match_sources(retrieved: list[str], expected: list[str]) -> list[str]:
    """Replace each retrieved source by the expected source it matches, so the id-based metrics
    can compare the two lists directly."""
    return [next((e for e in expected if source_matches(r, e)), r) for r in retrieved]


def evaluate(
    pipeline: BasePipeline,
    examples: list[EvalExample],
    generate: bool = True,
    judge: LLMJudge | None = None,
    workers: int = 1,
) -> EvalReport:
    """Score each example. With `generate=False` only retrieval is measured (fast, no LLM cost).
    With a `judge`, answers are also graded for correctness and faithfulness by an LLM.

    `workers > 1` answers that many questions at once (in threads), which mostly helps with
    API models where each question waits on the network. Results keep the order of `examples`.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if not generate and pipeline.retrieval_needs_generation:
        raise ValueError(
            f"{type(pipeline).__name__} retrieves based on generated text, so retrieval-only "
            "scores would only reflect its first search. Evaluate with generation, or use "
            "pipeline.type naive for retrieval-only numbers."
        )

    def score(example: EvalExample) -> EvalResult:
        return _evaluate_one(pipeline, example, generate, judge)

    if workers == 1 or len(examples) < 2:
        return EvalReport([score(e) for e in examples])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return EvalReport(list(pool.map(score, examples)))


def _evaluate_one(
    pipeline: BasePipeline, example: EvalExample, generate: bool, judge: LLMJudge | None
) -> EvalResult:
    # Usage is tracked per thread, so each question is measured on its own even while others
    # run in parallel. Answering and judging are tracked apart, which also works when the judge
    # shares the pipeline's generator.
    with track_usage() as usage:
        if generate:
            response = pipeline.run(example.question)
            contexts, answer = response.contexts, response.answer
        else:
            contexts, answer = pipeline.retrieve_for(example.question), ""

    retrieved = [Path(c.chunk.source).as_posix() for c in contexts]
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    if example.sources:
        matched = _match_sources(retrieved, example.sources)
        scores["hit_rate"] = hit_rate(matched, example.sources)
        scores["mrr"] = reciprocal_rank(matched, example.sources)
    if generate and example.answers:
        scores["exact_match"] = exact_match(answer, example.answers)
        scores["contains_answer"] = contains_answer(answer, example.answers)
        scores["f1"] = token_f1(answer, example.answers)
    if generate and (validity := citation_validity(answer, len(contexts))) is not None:
        scores["citation_valid"] = validity
    with track_usage() as judge_usage:
        if generate and judge is not None:
            if example.answers:
                scores["judge_correct"], reasons["judge_correct"] = judge.correctness(
                    example.question,
                    answer,
                    example.answers,
                )
            scores["faithfulness"], reasons["faithfulness"] = judge.faithfulness(
                answer, format_context(contexts)
            )
            if (support := judge.citation_support(answer, contexts)) is not None:
                scores["citation_support"], reasons["citation_support"] = support
    return EvalResult(example, answer, retrieved, scores, reasons, usage, judge_usage)
