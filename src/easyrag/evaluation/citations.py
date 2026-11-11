"""Check the [n] citations in an answer against the numbered passages it was written from.

`citation_validity` needs no LLM: do the numbers point at passages that exist? Whether a cited
passage actually supports its sentence needs a judge (`LLMJudge.citation_support`), in the style
of ALCE's citation recall (Gao et al., 2023).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# [1], [2][3], [1, 2] and [1-3]
_MARKER = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Claim:
    """One sentence of an answer, without its citation markers, and the passages it cites."""

    text: str
    citations: list[int] = field(default_factory=list)


def parse_markers(text: str) -> list[int]:
    """Every passage number cited in `text`, in order, ranges expanded ("[1-3]" -> 1, 2, 3)."""
    numbers: list[int] = []
    for marker in _MARKER.findall(text):
        for part in re.split(r"\s*,\s*", marker):
            bounds = [int(x) for x in re.split(r"\s*[–-]\s*", part)]
            if len(bounds) == 2 and bounds[0] <= bounds[1] <= bounds[0] + 50:
                numbers.extend(range(bounds[0], bounds[1] + 1))
            else:
                numbers.extend(bounds)
    return numbers


def extract_claims(answer: str) -> list[Claim]:
    """Split `answer` into sentences with the citations each carries.

    A marker standing alone after a sentence ("Paris is the capital. [1]") belongs to that
    sentence.
    """
    claims: list[Claim] = []
    for sentence in _SENTENCE_END.split(answer):
        citations = parse_markers(sentence)
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(_MARKER.sub("", sentence).split()))
        if not text.strip(" .,;:"):
            if claims:
                claims[-1].citations.extend(citations)
            continue
        claims.append(Claim(text, citations))
    return claims


def invalid_citations(answer: str, num_passages: int) -> list[int]:
    """Cited numbers that point at no passage (below 1 or above `num_passages`), de-duplicated."""
    return sorted({n for n in parse_markers(answer) if not 1 <= n <= num_passages})


def citation_validity(answer: str, num_passages: int) -> float | None:
    """Fraction of citation markers that point at an existing passage; None if nothing is cited."""
    numbers = parse_markers(answer)
    if not numbers:
        return None
    return sum(1 <= n <= num_passages for n in numbers) / len(numbers)
