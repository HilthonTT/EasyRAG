from pathlib import Path

from easyrag.loaders import DirectoryLoader, HTMLLoader


def test_html_loader_separates_table_cells(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text(
        "<html><head><title>T</title></head><body>"
        "<table><tr><td>Alpha</td><td>Beta</td></tr></table><h6>Six</h6></body></html>",
        encoding="utf-8",
    )

    doc = HTMLLoader(page).load()[0]

    assert doc.content.splitlines() == ["Alpha", "Beta", "Six"]
    assert doc.metadata["title"] == "T"


def test_directory_loader_tolerates_non_utf8_file(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))

    docs = DirectoryLoader(tmp_path).load()

    assert sorted(Path(d.metadata["source"]).name for d in docs) == ["good.txt", "latin1.txt"]
