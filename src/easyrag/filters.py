"""Metadata filters: restrict retrieval to chunks whose metadata matches.

A filter is either a callable `chunk -> bool` or a mapping of metadata keys to wanted values:

    {"source": "*handbook/*"}            # glob on the value (`*` also matches `/`)
    {"source": ["*leave.md", "*travel.md"]}  # a list matches any of its values
    {"page": 3}                          # equality; "3" also matches 3 (handy from the CLI)
    {"source": "*.pdf", "page": [1, 2]}  # every key must match

A chunk without the key does not match. Glob patterns (`*`, `?`, `[`) are matched
case-sensitively, with backslashes in both pattern and value treated as `/`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from typing import Any

from easyrag.schema import Chunk

ChunkPredicate = Callable[[Chunk], bool]
MetadataFilter = Mapping[str, Any] | ChunkPredicate

_GLOB_CHARS = frozenset("*?[")


def compile_filter(filter: MetadataFilter | None) -> ChunkPredicate | None:
    """Turn a filter into a predicate; `None` or an empty mapping means "no filter"."""
    if filter is None:
        return None
    if callable(filter):
        return filter
    if not isinstance(filter, Mapping):
        raise TypeError(f"filter must be a mapping or a callable, got {type(filter).__name__}")
    if not filter:
        return None
    conditions = list(filter.items())

    def predicate(chunk: Chunk) -> bool:
        metadata = chunk.metadata
        return all(key in metadata and _matches(metadata[key], want) for key, want in conditions)

    return predicate


def _matches(value: Any, want: Any) -> bool:
    if isinstance(want, list | tuple | set | frozenset):
        return any(_matches(value, w) for w in want)
    if isinstance(value, bool) or isinstance(want, bool):
        # True == 1 in Python, but a flag should not match a page number; "true" from the CLI
        # should match True.
        return str(value).lower() == str(want).lower()
    if bool(value == want) or str(value) == str(want):
        return True  # checked first, so a literal value like "notes[1].md" matches itself
    if isinstance(want, str) and _GLOB_CHARS.intersection(want):
        return fnmatchcase(str(value).replace("\\", "/"), want.replace("\\", "/"))
    return False


def parse_filter_args(items: list[str]) -> dict[str, Any]:
    """`["source=*.md", "page=1", "page=2"]` -> `{"source": "*.md", "page": ["1", "2"]}`."""
    result: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"filter must look like KEY=VALUE, got {item!r}")
        if key in result:
            previous = result[key]
            result[key] = [*previous, value] if isinstance(previous, list) else [previous, value]
        else:
            result[key] = value
    return result
