from __future__ import annotations

from typing import Any

from easyrag.config import Config
from easyrag.pipelines.base import BasePipeline
from easyrag.prompts import ITERATIVE_QUERY_TEMPLATE, build_rag_prompt
from easyrag.schema import RAGResponse


class IterativeRAGPipeline(BasePipeline):
    """Iterative retrieval-generation (ITER-RETGEN, Shao et al., 2023; FlashRAG IterativePipeline).

    Each round retrieves with the question plus the previous answer, so facts surfaced in one round
    pull in the passages needed for the next. Helps on multi-hop questions.
    """

    def __init__(self, *args: Any, max_iterations: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = max(1, max_iterations)

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"max_iterations": config.pipeline.max_iterations}

    def run(self, question: str) -> RAGResponse:
        answer = ""
        contexts = []
        history = []
        for _ in range(self.max_iterations):
            query = (
                ITERATIVE_QUERY_TEMPLATE.format(question=question, previous_answer=answer)
                if answer
                else question
            )
            contexts = self.retrieve(query, question)
            prompt = build_rag_prompt(question, contexts, self.prompt_template)
            answer = self.generator.generate(prompt, system=self.system_prompt)
            history.append({"query": query, "answer": answer})
        return RAGResponse(
            question=question, answer=answer, contexts=contexts, metadata={"iterations": history}
        )
