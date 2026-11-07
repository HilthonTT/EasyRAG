from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Optional, relative to the working directory. The dataclass defaults below are the real defaults
# (configs/default.yaml mirrors them, checked by a test), so running from elsewhere is the same.
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
ENV_PREFIX = "EASYRAG"


@dataclass
class SplitterConfig:
    method: str = "recursive"  # recursive | character
    chunk_size: int = 500
    chunk_overlap: int = 50


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
    type: str = "naive"  # naive | hyde | multi_query | iterative | self_ask | ircot
    max_iterations: int = 3  # iterative, self_ask and ircot
    num_queries: int = 3  # multi_query: LLM rewrites searched alongside the question
    index_dir: str = ".rag_index"


@dataclass
class Config:
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    judge: GeneratorConfig = field(
        default_factory=lambda: GeneratorConfig(provider="")
    )  # "" = same as generator
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
        if not isinstance(raw, dict):
            raise ValueError("Config must be a mapping of sections")
        config = cls()
        for section_name, values in raw.items():
            if not hasattr(config, section_name):
                raise ValueError(f"Unknown config section: {section_name}")
            section = getattr(config, section_name)
            if values is not None and not isinstance(values, dict):
                raise ValueError(f"Config section {section_name} must be a mapping of keys")
            for key, value in (values or {}).items():
                if not hasattr(section, key):
                    raise ValueError(f"Unknown config key: {section_name}.{key}")
                try:
                    setattr(section, key, _coerce(value, type(getattr(section, key))))
                except ValueError as e:
                    raise ValueError(f"Invalid config value for {section_name}.{key}: {e}") from e
        return config

    def apply_env_overrides(self) -> None:
        """Override values from env vars named EASYRAG_<SECTION>_<KEY>."""
        for section in fields(self):
            current = getattr(self, section.name)
            for f in fields(current):
                name = f"{ENV_PREFIX}_{section.name}_{f.name}".upper()
                env_value = os.environ.get(name)
                if env_value is not None:
                    try:
                        value = _coerce(env_value, type(getattr(current, f.name)))
                    except ValueError as e:
                        raise ValueError(f"Invalid value for {name}: {e}") from e
                    setattr(current, f.name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(value: Any, target: type) -> Any:
    if target is bool:
        return value if isinstance(value, bool) else str(value).strip().lower() in _TRUE
    if isinstance(value, bool):  # bool is an int subclass: `top_k: true` must not pass as 1
        raise ValueError(f"expected {target.__name__}, got {value!r}")
    if isinstance(value, target):
        return value
    if value is None:
        return target()
    if isinstance(value, dict | list):  # str(["a"]) would silently become "['a']"
        raise ValueError(f"expected {target.__name__}, got {value!r}")
    if target is int and isinstance(value, float) and not value.is_integer():
        raise ValueError(f"expected an integer, got {value!r}")  # int(2.7) would silently be 2
    try:
        return target(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"expected {target.__name__}, got {value!r}") from e


_TRUE = {"1", "true", "yes", "on"}


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
