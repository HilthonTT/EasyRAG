from __future__ import annotations

import re
from typing import Any

from easyrag.config import Config
from easyrag.pipelines.base import BasePipeline, merge_contexts
from easyrag.prompts import (
    REASONING_ANSWER_PROMPT,
    SELF_ASK_SCRATCHPAD,
    SELF_ASK_SYSTEM_PROMPT,
    build_rag_prompt,
    format_context,
)
from easyrag.schema import RAGResponse, RetrievedChunk

_FOLLOW_UP = re.compile(r"Follow up:\s*(.+)", re.IGNORECASE)
_FINAL_ANSWER = re.compile(r"So the final answer is:\s*(.+)", re.IGNORECASE)
_INTERMEDIATE = re.compile(r"Intermediate answer:", re.IGNORECASE)


class SelfAskPipeline(BasePipeline):
    """Self-Ask (Press et al., 2023; FlashRAG SelfAskPipeline).

    The model breaks a multi-hop question into "Follow up:" sub-questions. Each one is answered
    from retrieved passages, and the answers are fed back until the model writes the final answer.
    """

    retrieval_needs_generation = True

    def __init__(self, *args: Any, max_iterations: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max(1, max_iterations)

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"max_iterations": config.pipeline.max_iterations}

    def run(self, question: str) -> RAGResponse:
        scratchpad = SELF_ASK_SCRATCHPAD.format(question=question)
        contexts: list[RetrievedChunk] = []
        steps: list[dict[str, str]] = []
        final_answer: str | None = None

        for _ in range(self.max_iterations):
            step = _strip_invented_answer(
                self.generator.generate(scratchpad, system=SELF_ASK_SYSTEM_PROMPT)
            )
            final_answer = self._parse_final_answer(step)
            if final_answer is not None:
                break
            follow_up = self._parse_follow_up(step)
            if follow_up is None:  # the model wrote neither marker: answer from what we have
                break

            results = self.retrieve(follow_up, question)
            contexts = merge_contexts(contexts, results)
            prompt = build_rag_prompt(follow_up, results, self.prompt_template)
            answer = " ".join(self.generator.generate(prompt, system=self.system_prompt).split())

            if not steps:
                scratchpad += " Yes."
            scratchpad += f"\nFollow up: {follow_up}\nIntermediate answer: {answer}"
            steps.append({"follow_up": follow_up, "answer": answer})

        if final_answer is None:
            if not contexts:  # no follow-ups at all: fall back to plain retrieval
                contexts = self.retrieve(question)
            reasoning = "\n".join(
                f"Follow up: {s['follow_up']}\nIntermediate answer: {s['answer']}" for s in steps
            )
            prompt = REASONING_ANSWER_PROMPT.format(
                context=format_context(contexts), question=question, reasoning=reasoning or "(none)"
            )
            final_answer = self.generator.generate(prompt, system=self.system_prompt)

        return RAGResponse(
            question=question, answer=final_answer, contexts=contexts, metadata={"steps": steps}
        )

    @staticmethod
    def _parse_follow_up(text: str) -> str | None:
        """Return the sub-question after "Follow up:", or None if there isn't one."""
        match = _FOLLOW_UP.search(text)
        return (match.group(1).strip() or None) if match else None

    @staticmethod
    def _parse_final_answer(text: str) -> str | None:
        """Return the answer after "So the final answer is:", or None if there isn't one."""
        match = _FINAL_ANSWER.search(text)
        return (match.group(1).strip() or None) if match else None


def _strip_invented_answer(text: str) -> str:
    """Cut the step at "Intermediate answer:". Models without stop sequences often keep writing
    and invent the answer themselves; that part must come from retrieval instead."""
    return _INTERMEDIATE.split(text, maxsplit=1)[0]
