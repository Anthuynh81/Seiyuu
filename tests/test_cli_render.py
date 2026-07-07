from pathlib import Path

import pytest
from click.testing import CliRunner

import seiyuu.engines
from fake_engine import FakeEngine
from seiyuu.cli import main
from seiyuu.ingest import parse_epub, write_normalized


@pytest.fixture
def ingested_book(synthetic_epub: Path, tmp_path: Path) -> tuple[Path, Path, str]:
    result = parse_epub(synthetic_epub)
    write_normalized(result.book, tmp_path / "books")
    return tmp_path / "books", tmp_path / "output", result.book.book_meta.book_id


def test_render_command(ingested_book, monkeypatch) -> None:
    books_dir, output_dir, book_id = ingested_book
    fake = FakeEngine()
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: fake)

    result = CliRunner().invoke(
        main,
        [
            "render",
            book_id,
            "--voice",
            "test_voice",
            "--books-dir",
            str(books_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "segments synthesized" in result.output
    assert (output_dir / book_id / "manifest.json").is_file()
    assert len(fake.calls) > 0


def test_render_command_book_id_prefix(ingested_book, monkeypatch) -> None:
    books_dir, output_dir, book_id = ingested_book
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: FakeEngine())

    result = CliRunner().invoke(
        main,
        [
            "render",
            "synthetic",  # unambiguous prefix
            "--voice",
            "test_voice",
            "--books-dir",
            str(books_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / book_id / "manifest.json").is_file()


def test_render_command_unknown_book(tmp_path) -> None:
    (tmp_path / "books").mkdir()
    result = CliRunner().invoke(main, ["render", "nope", "--books-dir", str(tmp_path / "books")])
    assert result.exit_code != 0
    assert "not found" in result.output
    assert "seiyuu ingest" in result.output


def test_estimate_cost_indextts2_missing_checkpoints_is_clean_error(ingested_book) -> None:
    """indextts2 is the only engine whose model_version can raise (SynthesisError when
    checkpoints aren't configured). The single-voice estimate path must surface that as a clean
    click error, not an uncaught traceback (real engine, not mocked; no checkpoints in defaults)."""
    from seiyuu.engines.base import SynthesisError

    books_dir, output_dir, book_id = ingested_book
    result = CliRunner().invoke(
        main,
        [
            "estimate-cost", book_id, "--engine", "indextts2", "--voice", "whatever",
            "--books-dir", str(books_dir), "--output-dir", str(output_dir),
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    # caught and re-raised as a click error, NOT propagated as a raw SynthesisError traceback
    assert not isinstance(result.exception, SynthesisError), result.output
    assert "indextts2_checkpoints_dir" in result.output
