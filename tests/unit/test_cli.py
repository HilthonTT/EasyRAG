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
