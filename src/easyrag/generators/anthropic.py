from typing import Any

from easyrag.generators.base import BaseGenerator


class AnthropicGenerator(BaseGenerator):
    def __init__(
        self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 16_000
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic is not installed. Run: uv sync --extra anthropic") from e

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

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
        return system

    def generate(self, prompt: str, system: str | None) -> str:
        response = self.client.beta.messages.create(**self._params(prompt, system))
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined to answer this request.")
        return "".join(b.text for b in response.content if b.type == "text")

    def stream(self, prompt: str, system: str | None = None):
        with self.client.beta.messages.stream(**self._params(prompt, system)) as stream:
            yield from stream.text_stream
            if stream.get_final_message().stop_reason == "refusal":
                raise RuntimeError("The model declined to answer this request.")
