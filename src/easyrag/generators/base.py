from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class BaseGenerator(ABC):
    """Interface for text generation backends (OpenAI, Anthropic, HF, Ollama, ...)."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's completion for `prompt`."""

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield the response in pieces as it is produced. Defaults to one piece."""
        yield self.generate(prompt, system)
