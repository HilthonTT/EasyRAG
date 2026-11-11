from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from easyrag.config import Config
from easyrag.pipelines.base import BasePipeline
from easyrag.prompts import (
    AGENT_SCRATCHPAD,
    AGENT_SYSTEM_PROMPT,
    REASONING_ANSWER_PROMPT,
    format_context,
)
from easyrag.schema import RAGResponse, RetrievedChunk

_ACTION = re.compile(r"Action:\s*(search|finish)\s*\[", re.IGNORECASE)
_THOUGHT = re.compile(r"Thought:\s*(.*)", re.IGNORECASE)
_OBSERVATION = re.compile(r"Observation:", re.IGNORECASE)


class AgentPipeline(BasePipeline):
    """Agent-style RAG: a ReAct loop (Yao et al., 2023) with a search tool.

    Unlike the fixed multi-step pipelines, the model decides what to search for, whether to
    search again with a different query, and when it knows enough to answer. Each step it writes
    a thought and an action, `search[query]` or `finish[answer]`; search results come back as an
    observation. Passages are numbered once, in the order they are first found, so citations in
    the answer match `RAGResponse.contexts`.

    Uses a plain-text action format, so it works with every generator. At most `max_iterations`
    searches; if the model has not finished by then, it answers from everything found.
    """

    retrieval_needs_generation = True

    def __init__(self, *args: Any, max_iterations: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max(1, max_iterations)

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"max_iterations": config.pipeline.max_iterations}

    def run(self, question: str) -> RAGResponse:
        trajectory = AGENT_SCRATCHPAD.format(question=question)
        contexts: list[RetrievedChunk] = []
        steps: list[dict[str, Any]] = []
        answer: str | None = None

        for _ in range(self.max_iterations + 1):  # the extra step lets the model finish
            output = _cut_observation(
                self.generator.generate(trajectory, system=AGENT_SYSTEM_PROMPT)
            ).strip()
            action = self._parse_action(output)
            if action is None:  # no action: answer from what was found
                break
            kind, argument = action
            thought = self._parse_thought(output)
            trajectory += f"\n{output}"
            if kind == "finish":
                answer = argument
                steps.append({"thought": thought, "action": "finish"})
                break
            if len(steps) >= self.max_iterations or not argument:
                break
            numbers, observation = self._observe(contexts, self.retrieve(argument, question))
            trajectory += f"\nObservation:\n{observation}"
            steps.append(
                {"thought": thought, "action": "search", "query": argument, "passages": numbers}
            )

        if answer is None:
            if not contexts:
                contexts = self.retrieve(question)
            reasoning = "\n".join(
                f"Searched: {s['query']}" for s in steps if s["action"] == "search"
            )
            prompt = REASONING_ANSWER_PROMPT.format(
                context=format_context(contexts), question=question, reasoning=reasoning or "(none)"
            )
            answer = self.generator.generate(prompt, system=self.system_prompt)

        return RAGResponse(
            question=question, answer=answer, contexts=contexts, metadata={"steps": steps}
        )

    @staticmethod
    def _observe(
        contexts: list[RetrievedChunk], results: list[RetrievedChunk]
    ) -> tuple[list[int], str]:
        """Number `results` after the passages already in `contexts` (appending the new ones)
        and render them; passages shown before are only referred to by number."""
        if not results:
            return [], "No passages found."
        position = {c.chunk.id: i for i, c in enumerate(contexts, start=1)}
        numbers, parts = [], []
        for r in results:
            if r.chunk.id in position:
                n = position[r.chunk.id]
                parts.append(f"[{n}] (shown above)")
            else:
                contexts.append(r)
                n = position[r.chunk.id] = len(contexts)
                source = Path(r.chunk.source).name
                header = f"[{n}] (source: {source})" if source else f"[{n}]"
                parts.append(f"{header}\n{r.chunk.content}")
            numbers.append(n)
        return numbers, "\n\n".join(parts)

    @staticmethod
    def _parse_action(text: str) -> tuple[str, str] | None:
        """("search", query) or ("finish", answer) from the "Action:" line, or None."""
        match = _ACTION.search(text)
        if match is None:
            return None
        kind = match.group(1).lower()
        argument = text[match.end() :].strip()
        if kind == "search":
            argument = argument.split("\n", 1)[0].strip()
        if argument.endswith("]"):
            argument = argument[:-1]
        return kind, argument.strip()

    @staticmethod
    def _parse_thought(text: str) -> str:
        match = _THOUGHT.search(text)
        return match.group(1).strip() if match else ""


def _cut_observation(text: str) -> str:
    """Drop anything from an "Observation:" on: models without stop sequences sometimes invent
    search results themselves, and those must come from retrieval instead."""
    return _OBSERVATION.split(text, maxsplit=1)[0]
