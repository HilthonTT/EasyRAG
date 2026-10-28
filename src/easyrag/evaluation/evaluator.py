from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from easyrag.evaluation.metrics import (
    contains_answer,
    exact_match,
    hit_rate,
    reciprocal_rank,
    token_f1,
)
from easyrag.pipelines.base import BasePipeline


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


@dataclass
class EvalReport:
    results: list[EvalResult]

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
            "results": [
                {
                    "question": r.example.question,
                    "answer": r.answer,
                    "expected": r.example.answers,
                    "expected_sources": r.example.sources,
                    "retrieved_sources": r.retrieved_sources,
                    "scores": r.scores,
                }
                for r in self.results
            ],
        }


def load_examples(path: str | Path) -> list[EvalExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append(
                    EvalExample(
                        question=data["question"],
                        answers=list(data.get("answers", [])),
                        sources=list(data.get("sources", [])),
                    )
                )
    return examples


def evaluate(
    pipeline: BasePipeline, examples: list[EvalExample], generate: bool = True
) -> EvalReport:
    """Score each example. With `generate=False` only retrieval is measured (fast, no LLM cost)."""
    results = []
    for example in examples:
        if generate:
            response = pipeline.run(example.question)
            contexts, answer = response.contexts, response.answer
        else:
            contexts, answer = pipeline.retrieve(pipeline.search_query(example.question)), ""

        retrieved = [Path(c.chunk.source).name for c in contexts]
        relevant = [Path(s).name for s in example.sources]
        scores: dict[str, float] = {}
        if relevant:
            scores["hit_rate"] = hit_rate(retrieved, relevant)
            scores["mrr"] = reciprocal_rank(retrieved, relevant)
        if generate and example.answers:
            scores["exact_match"] = exact_match(answer, example.answers)
            scores["contains_answer"] = contains_answer(answer, example.answers)
            scores["f1"] = token_f1(answer, example.answers)
        results.append(EvalResult(example, answer, retrieved, scores))
    return EvalReport(results)
