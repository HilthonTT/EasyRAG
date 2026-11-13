"""Multi-turn chat on top of any pipeline.

Pipelines answer one self-contained question. In a conversation, follow-ups such as "and for
part-time staff?" or "who leads it?" only make sense with the earlier turns, and searching for
them as written finds the wrong passages. `Conversation` keeps the recent turns and, before each
follow-up, asks the LLM to rewrite it as a standalone question (LangChain's "condense question"
step). The pipeline then searches for and answers that question, so this works with every
`pipeline.type`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from easyrag.pipelines.base import BasePipeline
from easyrag.prompts import CONDENSE_QUESTION_SYSTEM_PROMPT
from easyrag.schema import RAGResponse, RetrievedChunk

_PREFIX = re.compile(r"^\s*(?:standalone\s+question|question)\s*:\s*", re.IGNORECASE)


@dataclass
class Turn:
    """One exchange. `standalone` is what was searched for (the question, rewritten if needed)."""

    question: str
    standalone: str
    answer: str


class Conversation:
    """A chat session: answers each question in the light of the previous `max_turns` turns."""

    def __init__(
        self, pipeline: BasePipeline, max_turns: int = 5, max_answer_chars: int = 600
    ) -> None:
        self.pipeline = pipeline
        self.max_turns = max(0, max_turns)
        self.max_answer_chars = max_answer_chars
        self.history: list[Turn] = []

    def reset(self) -> None:
        self.history.clear()

    def standalone_question(self, question: str) -> str:
        """`question` rewritten to make sense without the conversation (unchanged on turn one)."""
        turns = self._recent_turns()
        if not turns:
            return question
        # The transcript goes in the system prompt and the user prompt is only the question, so
        # a generator that echoes its prompt (the offline default) leaves the question unchanged.
        system = CONDENSE_QUESTION_SYSTEM_PROMPT.format(history=self._format(turns))
        raw = self.pipeline.generator.generate(question, system=system)
        return self._parse(raw) or question

    def stream(self, question: str) -> tuple[str, list[RetrievedChunk], Iterator[str]]:
        """Return the standalone question, the contexts and an iterator over the answer.

        The turn is added to the history once the answer has been read to the end.
        """
        standalone = self.standalone_question(question)
        contexts, answer = self.pipeline.stream(standalone)
        return standalone, contexts, self._record(question, standalone, answer)

    def run(self, question: str) -> RAGResponse:
        standalone, contexts, answer = self.stream(question)
        return RAGResponse(
            question=question,
            answer="".join(answer),
            contexts=contexts,
            metadata={"standalone_question": standalone},
        )

    def _record(self, question: str, standalone: str, answer: Iterator[str]) -> Iterator[str]:
        pieces = []
        for piece in answer:
            pieces.append(piece)
            yield piece
        self.history.append(Turn(question, standalone, "".join(pieces).strip()))

    def _recent_turns(self) -> list[Turn]:
        return self.history[-self.max_turns :] if self.max_turns else []

    def _format(self, turns: list[Turn]) -> str:
        lines = []
        for turn in turns:
            answer = " ".join(turn.answer.split())
            if len(answer) > self.max_answer_chars:
                answer = answer[: self.max_answer_chars].rstrip() + " ..."
            lines.append(f"User: {turn.standalone}\nAssistant: {answer}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> str:
        for line in raw.splitlines():
            line = _PREFIX.sub("", line).strip().strip('"').strip()
            if line:
                return line
        return ""
