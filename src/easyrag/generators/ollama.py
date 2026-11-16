from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from typing import Any

from easyrag.generators.base import BaseGenerator

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaGenerator(BaseGenerator):
    def __init__(
        self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, timeout: float = 300
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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

    def generate(self, prompt: str, system: str | None = None) -> str:
        with self._request(prompt, system, stream=False) as response:
            data = json.loads(response.read())
        self._record_usage(data)
        return str(data["message"]["content"])

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        with self._request(prompt, system, stream=True) as response:
            final: dict[str, Any] = {}
            for line in response:
                if line.strip():
                    data = json.loads(line)
                    if data.get("done"):  # the last line carries the token counts
                        final = data
                    yield data.get("message", {}).get("content", "")
        self._record_usage(final)

    def _record_usage(self, data: dict[str, Any]) -> None:
        self._record(data.get("prompt_eval_count"), data.get("eval_count"))
