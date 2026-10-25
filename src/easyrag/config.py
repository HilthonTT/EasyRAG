from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
ENV_PREFIX = "EASYRAG"


@dataclass
class SplitterConfig:
    method: str = "recursive"  # recursive | character
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class EmbeddingConfig:
    provider: str = "hash"  # hash | sentence_transformer | openai
    model: str = ""
    batch_size: int = 32


@dataclass
class RetrieverConfig:
    method: str = "hybrid"  # vector | bm25 | hybrid
    top_k: int = 4
    rrf_k: int = 60  # reciprocal rank fusion constant for hybrid search


@dataclass
class RerankerConfig:
    provider: str = "none"  # none | cross_encoder
    model: str = ""
    fetch_k: int = 20  # candidates retrieved before reranking down to top_k


@dataclass
class GeneratorConfig:
    provider: str = "echo"  # echo | anthropic | openai | ollama
    model: str = ""
    max_tokens: int = 16000
    effort: str = "low"  # Anthropic only: low | medium | high | xhigh | max
    base_url: str = ""  # Ollama / OpenAI-compatible servers


@dataclass
class PipelineConfig:
    type: str = "naive"  # naive | hyde | iterative
    max_iterations: int = 3  # iterative only
    index_dir: str = ".rag_index"


@dataclass
class Config:
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load `path`, else `configs/default.yaml` if present, then apply .env and env vars."""
        load_dotenv()
        if path is None and DEFAULT_CONFIG_PATH.exists():
            path = DEFAULT_CONFIG_PATH
        config = cls.from_yaml(path) if path else cls()
        config.apply_env_overrides()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        config = cls()
        for section_name, values in raw.items():
            if not hasattr(config, section_name):
                raise ValueError(f"Unknown config section: {section_name}")
            section = getattr(config, section_name)
            for key, value in (values or {}).items():
                if not hasattr(section, key):
                    raise ValueError(f"Unknown config key: {section_name}.{key}")
                setattr(section, key, _coerce(value, type(getattr(section, key))))
        return config

    def apply_env_overrides(self) -> None:
        """Override values from env vars named EASYRAG_<SECTION>_<KEY>."""
        for section in fields(self):
            current = getattr(self, section.name)
            for f in fields(current):
                env_value = os.environ.get(f"{ENV_PREFIX}_{section.name}_{f.name}".upper())
                if env_value is not None:
                    setattr(current, f.name, _coerce(env_value, type(getattr(current, f.name))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(value: Any, target: type) -> Any:
    if isinstance(value, target):
        return value
    if target is bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return target()
    return target(value)


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)
