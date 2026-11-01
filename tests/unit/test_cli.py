from pathlib import Path

import pytest

from easyrag.cli import main
from easyrag.config import Config
from easyrag.factory import build_judge
from easyrag.generators.echo import EchoGenerator


def test_cli_reports_bad_config_as_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EASYRAG_PIPELINE_TYPE", "bogus")

    with pytest.raises(SystemExit) as exc:
        main(["ask", "hi"])

    assert exc.value.code == 2
    assert "Unknown pipeline.type: 'bogus'" in capsys.readouterr().err


def test_cli_index_then_ask(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "france.txt").write_text("Paris is the capital of France.", encoding="utf-8")
    index_dir = str(tmp_path / "index")

    main(["--index-dir", index_dir, "index", str(docs)])
    main(["--index-dir", index_dir, "ask", "What is the capital of France?"])

    out = capsys.readouterr().out
    assert "Indexed 1 documents into 1 chunks" in out
    assert "Sources: [1] france.txt" in out


def test_cli_judge_rejects_retrieval_only(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["eval", "x.jsonl", "--judge", "--retrieval-only"])

    assert exc.value.code == 2
    assert "--judge needs generated answers" in capsys.readouterr().err


def test_build_judge_falls_back_to_generator() -> None:
    config = Config()
    assert isinstance(build_judge(config).generator, EchoGenerator)

    config.judge.provider = "bogus"
    with pytest.raises(ValueError, match="Unknown judge.provider: 'bogus'"):
        build_judge(config)
