"""Common QA metrics, in the style used by RAG benchmarks."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, articles and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(ref) for ref in references))


def token_f1(prediction: str, references: list[str]) -> float:
    def f1(pred: str, ref: str) -> float:
        pred_tokens, ref_tokens = normalize_answer(pred).split(), normalize_answer(ref).split()
        common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
        if common == 0:
            return 0.0
        precision, recall = common / len(pred_tokens), common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    return max((f1(prediction, ref) for ref in references), default=0.0)


def hit_rate(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1.0 if any relevant item was retrieved, else 0.0."""
    return float(bool(set(retrieved_ids) & set(relevant_ids)))
