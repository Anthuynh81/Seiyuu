from pathlib import Path

import pytest
from click.testing import CliRunner
from ebooklib import epub

from factories import make_book  # noqa: F401  (re-exported for other tests)
from fake_engine import FakeEngine
from seiyuu.cli import main
from test_assemble import ffprobe_duration


@pytest.fixture
def fake_engine(monkeypatch) -> FakeEngine:
    import seiyuu.engines

    engine = FakeEngine()
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: engine)
    return engine


def run_convert(tmp_path: Path, epub_path: Path, *extra: str):
    return CliRunner().invoke(
        main,
        [
            "convert",
            str(epub_path),
            "--voice",
            "test_voice",
            "--books-dir",
            str(tmp_path / "books"),
            "--output-dir",
            str(tmp_path / "out"),
            *extra,
        ],  # fmt: skip
    )


def find_book_dir(root: Path) -> Path:
    return next(d for d in root.iterdir() if d.is_dir())


def test_convert_end_to_end(synthetic_epub, tmp_path, fake_engine) -> None:
    result = run_convert(tmp_path, synthetic_epub)
    assert result.exit_code == 0, result.output
    for stage in ("== ingest ==", "== render ==", "== assemble =="):
        assert stage in result.output

    books_book = find_book_dir(tmp_path / "books")
    assert (books_book / "normalized.json").is_file()
    out_book = find_book_dir(tmp_path / "out")
    assert (out_book / "manifest.json").is_file()
    mp3s = sorted((out_book / "chapters").glob("*.mp3"))
    assert [p.name for p in mp3s] == ["ch001.mp3", "ch002.mp3", "ch003.mp3"]
    assert len(fake_engine.calls) > 0


def test_convert_chapter_subset(synthetic_epub, tmp_path, fake_engine) -> None:
    result = run_convert(tmp_path, synthetic_epub, "--chapter", "3")
    assert result.exit_code == 0, result.output
    out_book = find_book_dir(tmp_path / "out")
    assert [p.name for p in sorted((out_book / "chapters").glob("*.mp3"))] == ["ch003.mp3"]


def test_pause_overrides_change_duration(synthetic_epub, tmp_path, fake_engine) -> None:
    base = run_convert(tmp_path, synthetic_epub, "--chapter", "3")
    assert base.exit_code == 0, base.output
    out_book = find_book_dir(tmp_path / "out")
    short = ffprobe_duration(out_book / "chapters" / "ch003.mp3")

    longer = run_convert(tmp_path, synthetic_epub, "--chapter", "3", "--pause-after-heading", "4.2")
    assert longer.exit_code == 0, longer.output
    long = ffprobe_duration(out_book / "chapters" / "ch003.mp3")
    # chapter 3 is heading + one paragraph: the gap grows from 1.2s to 4.2s
    assert long - short == pytest.approx(3.0, abs=0.2)


def make_big_epub(path: Path, paragraphs: int) -> Path:
    book = epub.EpubBook()
    book.set_identifier("big-test-001")
    book.set_title("Big Test Book")
    book.set_language("en")
    body = "<h2>Chapter 1</h2>" + "".join(
        f"<p>Paragraph number {i} content here.</p>" for i in range(paragraphs)
    )
    item = epub.EpubHtml(title="C1", file_name="c1.xhtml", lang="en")
    item.set_content(f"<html><body>{body}</body></html>")
    book.add_item(item)
    book.spine = [item]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


def test_full_book_confirmation_aborts(tmp_path, fake_engine) -> None:
    big = make_big_epub(tmp_path / "big.epub", paragraphs=400)
    result = CliRunner().invoke(
        main,
        [
            "convert",
            str(big),
            "--voice",
            "test_voice",
            "--books-dir",
            str(tmp_path / "books"),
            "--output-dir",
            str(tmp_path / "out"),
        ],  # fmt: skip
        input="n\n",
    )
    assert result.exit_code != 0
    assert "Full-book render" in result.output
    assert len(fake_engine.calls) == 0
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_full_book_confirmation_yes_flag(tmp_path, fake_engine) -> None:
    big = make_big_epub(tmp_path / "big.epub", paragraphs=400)
    result = run_convert(tmp_path, big, "--yes")
    assert result.exit_code == 0, result.output
    assert "Full-book render" not in result.output
    assert len(fake_engine.calls) == 401  # heading + 400 paragraphs
