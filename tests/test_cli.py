from click.testing import CliRunner

from conftest import build_synthetic_epub
from seiyuu.cli import main


def test_cli_help_runs() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Seiyuu" in result.output


def test_cli_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0


def test_cli_ingest_extracts_cover(tmp_path) -> None:
    png_cover = b"\x89PNG\r\n\x1a\n" + b"cli-cover"
    src = build_synthetic_epub(tmp_path / "covered.epub", cover_image=png_cover)
    result = CliRunner().invoke(
        main,
        [
            "ingest",
            str(src),
            "--books-dir",
            str(tmp_path / "books"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "cover:" in result.output
    book_dir = next(d for d in (tmp_path / "out").iterdir() if d.is_dir())
    assert (book_dir / "cover.png").read_bytes() == png_cover
