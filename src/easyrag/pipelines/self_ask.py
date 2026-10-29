from __future__ import annotations

from typing import Any

from easyrag.config import Config
from easyrag.pipelines.base import BasePipeline
from easyrag.schema import RAGResponse


class SelfAskPipeline(BasePipeline):
    """Self-Ask (Press et al., 2023; FlashRAG SelfAskPipeline).

    The model breaks a multi-hop question into "Follow up:" sub-questions. Each one is answered
    from retrieved passages, and the answers are fed back until the model writes the final answer.
    """

    def __init__(self, *args: Any, max_iterations: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max(1, max_iterations)

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"max_iterations": config.pipeline.max_iterations}

    def run(self, question: str) -> RAGResponse:
        # TODO:
        # 1. Start a scratchpad with the question (few-shot Self-Ask examples in the prompt)
        # 2. Up to max_iterations times:
        #    - generate the next step from the scratchpad
        #    - if it has a final answer (_parse_final_answer), stop
        #    - otherwise take the follow-up (_parse_follow_up), retrieve for it
        #      (self.retrieve(follow_up, question)), answer it from those passages and append
        #      "Follow up: ... / Intermediate answer: ..." to the scratchpad
        # 3. If no final answer appeared, generate one from the scratchpad
        # 4. Return RAGResponse with all retrieved contexts and metadata={"steps": [...]}
        raise NotImplementedError

    @staticmethod
    def _parse_follow_up(text: str) -> str | None:
        """Return the sub-question after "Follow up:", or None if there isn't one."""
        raise NotImplementedError

    @staticmethod
    def _parse_final_answer(text: str) -> str | None:
        """Return the answer after "So the final answer is:", or None if there isn't one."""
        raise NotImplementedError
