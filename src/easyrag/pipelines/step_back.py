from __future__ import annotations

import re
from typing import Any

from easyrag.config import Config
from easyrag.pipelines.multi_query import MultiQueryPipeline
from easyrag.prompts import STEP_BACK_SYSTEM_PROMPT

_LABEL = re.compile(r"^\s*step[- ]?back question:\s*", re.IGNORECASE)


class StepBackPipeline(MultiQueryPipeline):
    """Step-back prompting (Zheng et al., 2023).

    A specific question ("how many vacation days does a part-timer get?") often misses the
    passage stating the general rule it depends on. The LLM writes a more generic "step-back"
    question; both are searched and the results fused by rank, so the answer sees the specific
    facts and the principle behind them.
    """

    def __init__(self, *args: Any, rrf_k: int = 60, **kwargs: Any) -> None:
        kwargs.pop("num_queries", None)
        super().__init__(*args, num_queries=1, rrf_k=rrf_k, **kwargs)

    @classmethod
    def config_kwargs(cls, config: Config) -> dict[str, Any]:
        return {"rrf_k": config.retriever.rrf_k}

    def search_queries(self, question: str) -> list[str]:
        """The original question, then the step-back question (unless it is the same)."""
        raw = self.generator.generate(question, system=STEP_BACK_SYSTEM_PROMPT)
        raw = "\n".join(_LABEL.sub("", line) for line in raw.splitlines())
        return self._parse_queries(raw, question, 1)
