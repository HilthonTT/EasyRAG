from __future__ import annotations

from easyrag.generators.base import BaseGenerator


class EchoGenerator(BaseGenerator):
    """Returns the prompt unchanged. Lets you check a pipeline without calling an LLM."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        self._record()  # no tokenizer, so only the call is counted
        return prompt
