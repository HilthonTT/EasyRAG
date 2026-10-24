"""Configuration loading: YAML file defaults, overridden by environment variables."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SplitterConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class RetrieverConfig:
    top_k: int = 4


@dataclass
class EmbeddingConfig:
    provider: str = "fake"  # fake | openai | huggingface
    model: str = ""


@dataclass
class GeneratorConfig:
    provider: str = "echo"  # echo | openai
    model: str = ""
    temperature: float = 0.0


@dataclass
class Config:
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        config = cls()
        for section in fields(cls):
            values = raw.get(section.name, {})
            current = getattr(config, section.name)
            for key, value in values.items():
                if not hasattr(current, key):
                    raise ValueError(f"Unknown config key: {section.name}.{key}")
                setattr(current, key, value)
        config._apply_env_overrides()
        return config

    def _apply_env_overrides(self) -> None:
        """Override values from env vars named EASYRAG_<SECTION>_<KEY>."""
        for section in fields(self):
            current = getattr(self, section.name)
            for f in fields(current):
                env_value = os.environ.get(f"EASYRAG_{section.name}_{f.name}".upper())
                if env_value is not None:
                    setattr(current, f.name, type(getattr(current, f.name))(env_value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
