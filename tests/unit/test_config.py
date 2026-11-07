from pathlib import Path
from typing import Any

import pytest

from easyrag.config import Config

ROOT = Path(__file__).resolve().parents[2]


def test_default_yaml_matches_dataclass_defaults() -> None:
    # Config.load only finds configs/default.yaml from the repo root; elsewhere the dataclass
    # defaults apply. They must agree, or results depend on the working directory.
    assert Config.from_yaml(ROOT / "configs" / "default.yaml") == Config()


def test_from_dict_rejects_section_that_is_not_a_mapping() -> None:
    with pytest.raises(ValueError, match="pipeline"):
        Config.from_dict({"pipeline": "naive"})


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"retriever": {"top_k": True}}, "retriever.top_k"),
        ({"splitter": {"chunk_size": 2.7}}, "splitter.chunk_size"),
        ({"retriever": {"top_k": "many"}}, "retriever.top_k"),
        ({"splitter": {"method": ["recursive"]}}, "splitter.method"),
    ],
)
def test_from_dict_rejects_values_of_the_wrong_type(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Config.from_dict(raw)


def test_from_dict_rejects_a_top_level_that_is_not_a_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        Config.from_dict(["splitter"])  # type: ignore[arg-type]


def test_from_dict_accepts_whole_floats_and_numeric_strings() -> None:
    config = Config.from_dict({"retriever": {"top_k": 8.0, "rrf_k": "30"}})
    assert (config.retriever.top_k, config.retriever.rrf_k) == (8, 30)


def test_bad_env_override_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EASYRAG_RETRIEVER_TOP_K", "lots")
    with pytest.raises(ValueError, match="EASYRAG_RETRIEVER_TOP_K"):
        Config().apply_env_overrides()
