from __future__ import annotations

import json
import re

from easyrag.evaluation.citations import extract_claims
from easyrag.generators.base import BaseGenerator
from easyrag.prompts import format_context
from easyrag.schema import RetrievedChunk

CORRECTNESS_PROMPT = """Question: {question}
Reference answer(s): {references}
Candidate answer: {answer}

Does the candidate answer mean the same as a reference answer? Ignore wording and citations.
Reply with JSON only: {{"score": 0 or 1, "reason": "<one sentence>"}}"""

FAITHFULNESS_PROMPT = """Context passages:
{context}

Answer: {answer}

Is every factual claim in the answer supported by the context passages?
Reply with JSON only: {{"score": 0 or 1, "reason": "<one sentence>"}}"""


CITATION_PROMPT = """Cited passages:
{context}

Claim: {claim}

Do the cited passages, taken together, support the claim? Ignore citation numbers in it.
Reply with JSON only: {{"score": 0 or 1, "reason": "<one sentence>"}}"""


class LLMJudge:
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator

    def _score(self, prompt: str) -> tuple[float, str]:
        raw = self.generator.generate(prompt, system="You are a strict grader.")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return 0.0, f"unparseable judge reply: {raw[:200]}"
        try:
            data = json.loads(match.group())
            return min(max(float(data["score"]), 0.0), 1.0), str(data.get("reason", ""))
        except (ValueError, KeyError, TypeError):
            return 0.0, f"unparseable judge reply: {raw[:200]}"

    def correctness(self, question: str, answer: str, references: list[str]) -> tuple[float, str]:
        return self._score(
            CORRECTNESS_PROMPT.format(
                question=question,
                references=" | ".join(references),
                answer=answer,
            )
        )

    def faithfulness(self, answer: str, context: str) -> tuple[float, str]:
        return self._score(FAITHFULNESS_PROMPT.format(context=context, answer=answer))

    def citation_support(
        self, answer: str, contexts: list[RetrievedChunk]
    ) -> tuple[float, str] | None:
        """Fraction of cited sentences whose cited passages support them (ALCE citation recall).

        A sentence citing only numbers with no passage counts as unsupported. Returns None when
        the answer cites nothing. One judge call per cited sentence.
        """
        claims = [c for c in extract_claims(answer) if c.citations]
        if not claims:
            return None
        unsupported: list[str] = []
        for claim in claims:
            cited = [
                contexts[n - 1] for n in dict.fromkeys(claim.citations) if 1 <= n <= len(contexts)
            ]
            score = 0.0
            if cited:
                score, _ = self._score(
                    CITATION_PROMPT.format(context=format_context(cited), claim=claim.text)
                )
            if score < 0.5:
                unsupported.append(claim.text)
        support = 1 - len(unsupported) / len(claims)
        reason = "unsupported: " + " | ".join(unsupported) if unsupported else "all supported"
        return support, reason
