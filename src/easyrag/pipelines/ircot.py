from __future__ import annotations

from typing import Any

from easyrag.config import Config
from easyrag.pipelines.base import BasePipeline
from easyrag.schema import RAGResponse


class IRCOTPipeline(BasePipeline):
    """Interleaving Retrieval with Chain-of-Thought (IRCoT, Trivedi et al., 2023; FlashRAG
    IRCOTPipeline).

    Alternates one reasoning sentence with one retrieval: each new thought becomes the next search
    query, so later steps can find passages that only an earlier step knew to look for.
    """

    def __init__(self, *args: Any, max_iterations: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max(1, max_iterations)

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"max_iterations": config.pipeline.max_iterations}

    def run(self, question: str) -> RAGResponse:
        # TODO:
        # 1. contexts = self.retrieve(question)
        # 2. Up to max_iterations times:
        #    - prompt with the question, all passages so far and the thoughts so far; ask for the
        #      next single reasoning sentence
        #    - stop if it contains the answer marker (see _extract_answer)
        #    - otherwise retrieve with that sentence (self.retrieve(thought, question)) and merge
        #      new chunks into contexts, de-duplicated by chunk id, best score first
        # 3. If no answer marker appeared, generate a final answer from all contexts
        # 4. Return RAGResponse with metadata={"thoughts": [...]}
        raise NotImplementedError

    @staticmethod
    def _extract_answer(thought: str) -> str | None:
        """Return the answer if `thought` contains the final-answer marker ("So the answer is:")."""
        raise NotImplementedError
