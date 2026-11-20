from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# Guards the running totals: generators may be called from several threads (parallel eval).
_USAGE_LOCK = threading.RLock()
# Usage objects opened by `track_usage` in the current thread / context.
_SCOPES: ContextVar[tuple[Usage, ...]] = ContextVar("easyrag_usage_scopes", default=())


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


@contextmanager
def track_usage() -> Iterator[Usage]:
    """Count the LLM use of every generator called inside the block, in this thread only.

    Unlike diffing `generator.usage` snapshots, this stays correct while other threads use the
    same generator, so questions can be answered in parallel and still be measured one by one.
    """
    usage = Usage()
    token = _SCOPES.set((*_SCOPES.get(), usage))
    try:
        yield usage
    finally:
        _SCOPES.reset(token)


class BaseGenerator(ABC):
    """Interface for text generation backends (OpenAI, Anthropic, HF, Ollama, ...)."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's completion for `prompt`."""

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield the response in pieces as it is produced. Defaults to one piece."""
        yield self.generate(prompt, system)

    @property
    def name(self) -> str:
        """Identifies the model and the settings that change its output (used as a cache key)."""
        return type(self).__name__

    @property
    def usage(self) -> Usage:
        """Running totals. Copy with `dataclasses.replace` before a step to diff against later,
        or use `track_usage` when other threads may call the generator meanwhile."""
        # Created on first use, so the subclasses' __init__ methods don't have to call super()
        with _USAGE_LOCK:
            if not hasattr(self, "_usage"):
                self._usage = Usage()
            return self._usage

    def _record(self, input_tokens: int | None = 0, output_tokens: int | None = 0) -> None:
        """Count one call. Subclasses call this; token counts the backend omits count as 0."""
        call = Usage(1, input_tokens or 0, output_tokens or 0)
        with _USAGE_LOCK:
            for usage in (self.usage, *_SCOPES.get()):
                usage.calls += call.calls
                usage.input_tokens += call.input_tokens
                usage.output_tokens += call.output_tokens
