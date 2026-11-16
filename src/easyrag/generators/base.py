from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Usage:
    """LLM calls and tokens a generator has used. Subtract two snapshots to measure one step."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def __sub__(self, other: Usage) -> Usage:
        return Usage(
            self.calls - other.calls,
            self.input_tokens - other.input_tokens,
            self.output_tokens - other.output_tokens,
        )


class BaseGenerator(ABC):
    """Interface for text generation backends (OpenAI, Anthropic, HF, Ollama, ...)."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's completion for `prompt`."""

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield the response in pieces as it is produced. Defaults to one piece."""
        yield self.generate(prompt, system)

    @property
    def usage(self) -> Usage:
        """Running totals. Copy with `dataclasses.replace` before a step to diff against later."""
        # Created on first use, so the subclasses' __init__ methods don't have to call super()
        if not hasattr(self, "_usage"):
            self._usage = Usage()
        return self._usage

    def _record(self, input_tokens: int | None = 0, output_tokens: int | None = 0) -> None:
        """Count one call. Subclasses call this; token counts the backend omits count as 0."""
        self.usage.calls += 1
        self.usage.input_tokens += input_tokens or 0
        self.usage.output_tokens += output_tokens or 0
