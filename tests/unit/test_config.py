from pathlib import Path

from easyrag.config import Config

ROOT = Path(__file__).resolve().parents[2]


def test_default_yaml_matches_dataclass_defaults() -> None:
    # Config.load only finds configs/default.yaml from the repo root; elsewhere the dataclass
    # defaults apply. They must agree, or results depend on the working directory.
    assert Config.from_yaml(ROOT / "configs" / "default.yaml") == Config()
