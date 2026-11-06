from __future__ import annotations

import re
from typing import Any

from easyrag.config import Config
from easyrag.pipelines.base import BasePipeline, merge_contexts
from easyrag.prompts import (
    IRCOT_PROMPT,
    IRCOT_SYSTEM_PROMPT,
    REASONING_ANSWER_PROMPT,
    format_context,
)
from easyrag.schema import RAGResponse

_ANSWER = re.compile(r"So the answer is:\s*(.+)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class IRCOTPipeline(BasePipeline):
    """Interleaving Retrieval with Chain-of-Thought (IRCoT, Trivedi et al., 2023; FlashRAG
    IRCOTPipeline).

    Alternates one reasoning sentence with one retrieval: each new thought becomes the next search
    query, so later steps can find passages that only an earlier step knew to look for.
    """

    retrieval_needs_generation = True

    def __init__(self, *args: Any, max_iterations: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max(1, max_iterations)

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"max_iterations": config.pipeline.max_iterations}

    def run(self, question: str) -> RAGResponse:
        contexts = self.retrieve(question)
        thoughts: list[str] = []
        answer: str | None = None

        for _ in range(self.max_iterations):
            prompt = IRCOT_PROMPT.format(
                context=format_context(contexts),
                question=question,
                thoughts="\n".join(thoughts) or "(none)",
            )
            output = self.generator.generate(prompt, system=IRCOT_SYSTEM_PROMPT).strip()
            answer = self._extract_answer(output)
            if answer is not None:
                thoughts.append(output)
                break
            thought = _first_sentence(output)
            if not thought:
                break
            thoughts.append(thought)
            contexts = merge_contexts(contexts, self.retrieve(thought, question))

        if answer is None:
            prompt = REASONING_ANSWER_PROMPT.format(
                context=format_context(contexts),
                question=question,
                reasoning="\n".join(thoughts) or "(none)",
            )
            answer = self.generator.generate(prompt, system=self.system_prompt)

        return RAGResponse(
            question=question, answer=answer, contexts=contexts, metadata={"thoughts": thoughts}
        )

    @staticmethod
    def _extract_answer(thought: str) -> str | None:
        """Return the answer if `thought` contains the final-answer marker ("So the answer is:")."""
        match = _ANSWER.search(thought)
        return (match.group(1).strip() or None) if match else None


def _first_sentence(text: str) -> str:
    """IRCoT keeps one sentence per step, even when the model writes more."""
    return _SENTENCE_END.split(" ".join(text.split()), maxsplit=1)[0]
