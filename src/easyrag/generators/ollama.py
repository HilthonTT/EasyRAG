from __future__ import annotations

import http.client
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from easyrag.generators.base import BaseGenerator

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"

# Worth another try: the server is busy, restarting or still loading the model.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)
MAX_BACKOFF = 30.0  # seconds, also the cap on a server's Retry-After

T = TypeVar("T")


class OllamaGenerator(BaseGenerator):
    """Chat with a model served by Ollama.

    Failed requests (connection errors, timeouts, HTTP 408/429/5xx) are retried up to
    `max_retries` times with exponential backoff, like the OpenAI and Anthropic SDKs do. A
    streamed answer is only retried before its first piece arrives.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300,
        max_retries: int = 2,
    ):
        # urlopen also follows file:// and ftp:// URLs; only ever talk HTTP to a server.
        if urllib.parse.urlsplit(base_url).scheme.lower() not in ("http", "https"):
            raise ValueError(f"Ollama base_url must be an http(s) URL, got {base_url!r}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    @property
    def name(self) -> str:
        return f"ollama:{self.base_url}:{self.model}"

    def _request(self, prompt: str, system: str | None, stream: bool) -> Any:
        messages = [{"role": "system", "content": system}] if system else []
        body = {
            "model": self.model,
            "messages": [*messages, {"role": "user", "content": prompt}],
            "stream": stream,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=self.timeout)

    def _with_retries(self, attempt: Callable[[], T]) -> T:
        for retry in range(self.max_retries + 1):
            try:
                return attempt()
            except _TRANSIENT_ERRORS as e:
                status = e.code if isinstance(e, urllib.error.HTTPError) else None
                permanent = status is not None and status not in RETRY_STATUSES
                if permanent or retry == self.max_retries:
                    raise
                time.sleep(_backoff(retry, e))
        raise AssertionError("unreachable")  # the last attempt returns or raises

    def generate(self, prompt: str, system: str | None = None) -> str:
        def attempt() -> dict[str, Any]:
            with self._request(prompt, system, stream=False) as response:
                data: dict[str, Any] = json.loads(response.read())
                return data

        data = self._with_retries(attempt)
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")
        self._record_usage(data)
        return str(data["message"]["content"])

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        response = self._with_retries(lambda: self._request(prompt, system, stream=True))
        with response:
            final: dict[str, Any] = {}
            for line in response:
                if line.strip():
                    data = json.loads(line)
                    if "error" in data:  # e.g. the model crashed or ran out of memory mid-answer
                        raise RuntimeError(f"Ollama error: {data['error']}")
                    if data.get("done"):  # the last line carries the token counts
                        final = data
                    yield data.get("message", {}).get("content", "")
        self._record_usage(final)

    def _record_usage(self, data: dict[str, Any]) -> None:
        self._record(data.get("prompt_eval_count"), data.get("eval_count"))


def _backoff(retry: int, error: Exception) -> float:
    """Seconds to wait before retry number `retry + 1`: the server's Retry-After if it sent one,
    else 0.5s, 1s, 2s, ... with jitter so parallel clients don't retry in lockstep."""
    if isinstance(error, urllib.error.HTTPError) and error.headers is not None:
        try:
            retry_after = float(str(error.headers.get("Retry-After", "")))
            return min(MAX_BACKOFF, max(0.0, retry_after))
        except ValueError:
            pass  # absent, or an HTTP date: use the backoff below
    return min(MAX_BACKOFF, 0.5 * 2.0**retry) * random.uniform(0.75, 1.25)
