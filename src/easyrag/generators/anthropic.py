from collections.abc import Iterator
from typing import Any

from easyrag.generators.base import BaseGenerator

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicGenerator(BaseGenerator):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        max_tokens: int = 16_000,
        max_retries: int = 2,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic is not installed. Run: uv sync --extra anthropic") from e

        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}:effort={self.effort}:max_tokens={self.max_tokens}"

    def _params(self, prompt: str, system: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            params["system"] = system
        return params

    def _record_usage(self, usage: Any) -> None:
        # Cached prompt tokens are reported separately from input_tokens; count them all as input
        self._record(
            (usage.input_tokens or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0),
            usage.output_tokens,
        )

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = self.client.beta.messages.create(**self._params(prompt, system))
        self._record_usage(response.usage)
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined to answer this request.")
        return "".join(b.text for b in response.content if b.type == "text")

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        with self.client.beta.messages.stream(**self._params(prompt, system)) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
            self._record_usage(final.usage)
            if final.stop_reason == "refusal":
                raise RuntimeError("The model declined to answer this request.")
