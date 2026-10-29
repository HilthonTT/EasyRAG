from pathlib import Path

from easyrag.loaders import DirectoryLoader


def test_directory_loader_tolerates_non_utf8_file(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))

    docs = DirectoryLoader(tmp_path).load()

    assert sorted(Path(d.metadata["source"]).name for d in docs) == ["good.txt", "latin1.txt"]
