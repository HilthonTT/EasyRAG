"""On-disk caches for embeddings and LLM calls, so repeated runs don't pay twice.

Wrap any embedding model or generator:

    embeddings = CachedEmbeddings(OpenAIEmbeddings(), ".rag_cache/embeddings.sqlite")
    generator = CachedGenerator(AnthropicGenerator(), ".rag_cache/llm.sqlite")

or set `cache.embeddings` / `cache.llm` in the config. Entries are keyed by the model's `name`
(which includes the settings that change its output) and the exact input text, so changing the
model, a prompt or a passage is a miss. Each cache is one SQLite file, safe to share between
threads and processes.

The LLM cache replays an answer instead of sampling a new one: handy to re-score an eval set or
to change only the judge, but turn it off to measure how much answers vary.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from easyrag.embeddings.base import BaseEmbeddings
from easyrag.generators.base import BaseGenerator, Usage

CACHE_VERSION = "1"  # bump when the stored format changes, so old entries become misses


class SQLiteCache:
    """A persistent key -> bytes table."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        with self._lock, self._db:
            self._db.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)")

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        unique = list(dict.fromkeys(keys))
        with self._lock:
            for start in range(0, len(unique), 500):  # stay under SQLite's variable limit
                batch = unique[start : start + 500]
                marks = ",".join("?" * len(batch))
                rows = self._db.execute(
                    f"SELECT key, value FROM cache WHERE key IN ({marks})", batch
                ).fetchall()
                found.update((k, bytes(v)) for k, v in rows)
        return found

    def set_many(self, items: dict[str, bytes]) -> None:
        if items:
            with self._lock, self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", items.items()
                )

    def set(self, key: str, value: bytes) -> None:
        self.set_many({key: value})

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


def cache_key(*parts: str | None) -> str:
    return hashlib.sha256(json.dumps([CACHE_VERSION, *parts]).encode("utf-8")).hexdigest()


class CachedEmbeddings(BaseEmbeddings):
    """Embeds each distinct text once per model; later calls read the stored vector.

    Keeps the wrapped model's `name`, so a saved index works with or without the cache.
    """

    def __init__(self, embeddings: BaseEmbeddings, path: str | Path) -> None:
        self.embeddings = embeddings
        self.cache = SQLiteCache(path)
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self.embeddings.name

    def _key(self, kind: str, text: str) -> str:
        # Queries and documents are kept apart: some models embed them differently.
        return cache_key(self.name, kind, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key("document", t) for t in texts]
        found = self.cache.get_many(keys)
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in found))
        if missing:
            vectors = self.embeddings.embed_documents(missing)
            new = {
                self._key("document", t): _pack(v) for t, v in zip(missing, vectors, strict=True)
            }
            self.cache.set_many(new)
            found.update(new)
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        return [_unpack(found[k]) for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._key("query", text)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return _unpack(cached)
        self.misses += 1
        vector = self.embeddings.embed_query(text)
        self.cache.set(key, _pack(vector))
        return vector


class CachedGenerator(BaseGenerator):
    """Answers a prompt it has seen before (same model, settings and system prompt) from the
    cache. Cache hits are not LLM calls, so they don't count towards `usage`."""

    def __init__(self, generator: BaseGenerator, path: str | Path) -> None:
        self.generator = generator
        self.cache = SQLiteCache(path)
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self.generator.name

    @property
    def usage(self) -> Usage:
        return self.generator.usage

    def _key(self, prompt: str, system: str | None) -> str:
        return cache_key(self.name, system, prompt)

    def generate(self, prompt: str, system: str | None = None) -> str:
        key = self._key(prompt, system)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached.decode("utf-8")
        self.misses += 1
        answer = self.generator.generate(prompt, system)
        self.cache.set(key, answer.encode("utf-8"))
        return answer

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        key = self._key(prompt, system)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            yield cached.decode("utf-8")
            return
        self.misses += 1
        pieces = []
        for piece in self.generator.stream(prompt, system):
            pieces.append(piece)
            yield piece
        # Only reached when the stream completed: an interrupted answer is not cached.
        self.cache.set(key, "".join(pieces).encode("utf-8"))


def _pack(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes) -> list[float]:
    return [float(x) for x in np.frombuffer(blob, dtype=np.float32)]
