from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Interface for text generation backends (OpenAI, Anthropic, HF, Ollama, ...)."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's completion for `prompt`."""
