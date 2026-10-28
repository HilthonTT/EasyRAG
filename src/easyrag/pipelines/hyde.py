from __future__ import annotations

from easyrag.pipelines.naive import NaiveRAGPipeline
from easyrag.prompts import HYDE_PROMPT


class HyDEPipeline(NaiveRAGPipeline):
    """Hypothetical Document Embeddings (Gao et al., 2022).

    Questions and answers are phrased differently, so the LLM first writes a hypothetical answer
    and retrieval searches with that, which lands closer to real answer passages.
    """

    def search_query(self, question: str) -> str:
        hypothetical = self.generator.generate(HYDE_PROMPT.format(question=question))
        return f"{question}\n{hypothetical}"
