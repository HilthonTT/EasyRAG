"""QA and retrieval metrics, in the style used by RAG benchmarks (FlashRAG, KILT, SQuAD)."""

from __future__ import annotations

import re
import string
from collections import Counter

_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, articles and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(ref) for ref in references))


def contains_answer(prediction: str, references: list[str]) -> float:
    """1.0 if any reference appears in the prediction. Fairer than EM for chatty LLM answers."""
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(ref) in pred for ref in references if ref.strip()))


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


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int | None = None) -> float:
    """Fraction of relevant items found in the first `k` retrieved."""
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids if k is None else retrieved_ids[:k])
    return len(top & set(relevant_ids)) / len(set(relevant_ids))


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / rank of the first relevant item (0 if none). Averaged over queries this is MRR."""
    relevant = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
