from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from easyrag.generators.base import BaseGenerator

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIGenerator(BaseGenerator):
    def __init__(
        self, model: str = DEFAULT_MODEL, max_tokens: int = 4096, base_url: str | None = None
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai is not installed. Run: uv sync --extra openai") from e

        self.client = openai.OpenAI(base_url=base_url or None)
        self.model = model
        self.max_tokens = max_tokens

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, Any]]:
        messages = [{"role": "system", "content": system}] if system else []
        return [*messages, {"role": "user", "content": prompt}]

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, messages=self._messages(prompt, system)
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self._messages(prompt, system),
            stream=True,
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
