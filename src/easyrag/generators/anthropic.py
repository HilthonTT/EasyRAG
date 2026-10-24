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

    def generate(self, prompt: str) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )

        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined to answer this request.")

        return "".join(b.text for b in response.content if b.type == "text")
