from pathlib import Path

import pytest

from easyrag.cli import main


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
