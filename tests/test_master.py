"""Master stage: render manifest → one chaptered .m4b (AAC) with markers and optional cover."""

import json
import subprocess

import pytest
from click.testing import CliRunner

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.assemble import master_book
from seiyuu.assemble.pipeline import _ffmetadata
from seiyuu.cli import main
from seiyuu.render import render_book


def _ffprobe(path, *show):
    out = subprocess.run(
        ["ffprobe", "-v", "error", *show, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.fixture
def book_dir(tmp_path):
    render_book(make_book(), FakeEngine(), "test_voice", tmp_path / "book")
    return tmp_path / "book"


def test_ffmetadata_format_and_escaping():
    meta = _ffmetadata("My Book", [("Chapter I", 0, 1000), ("Chapter; II", 1000, 2500)])
    assert meta.startswith(";FFMETADATA1\ntitle=My Book\n")
    assert "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\ntitle=Chapter I" in meta
    assert "START=1000\nEND=2500\ntitle=Chapter\\; II" in meta  # ';' escaped


def test_master_builds_chaptered_m4b(book_dir):
    result = master_book(book_dir, loudness=None)
    assert result.m4b_path.is_file()
    assert result.chapters == 2
    chapters = _ffprobe(result.m4b_path, "-show_chapters")["chapters"]
    assert [c["tags"]["title"] for c in chapters] == ["Chapter 1", "Chapter 2"]
    assert not (book_dir / "master").exists()  # working dir cleaned up


def test_master_audio_is_aac_44100(book_dir):
    result = master_book(book_dir, loudness=None)
    streams = _ffprobe(result.m4b_path, "-show_streams")["streams"]
    audio = next(s for s in streams if s["codec_type"] == "audio")
    assert audio["codec_name"] == "aac"
    assert audio["sample_rate"] == "44100"


def test_master_embeds_cover(book_dir, tmp_path):
    cover = tmp_path / "cover.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=blue:s=32x32", "-frames:v", "1", str(cover)],
        check=True,
    )  # fmt: skip
    result = master_book(book_dir, loudness=None, cover=cover)
    streams = _ffprobe(result.m4b_path, "-show_streams")["streams"]
    assert any(s["codec_type"] == "video" for s in streams)  # attached cover art


def test_master_cli(tmp_path, monkeypatch):
    import seiyuu.engines
    from seiyuu.ingest import write_normalized

    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: FakeEngine())
    write_normalized(make_book(), tmp_path / "books")
    runner = CliRunner()
    rendered = runner.invoke(
        main,
        ["render", "test-book", "--voice", "v",
         "--books-dir", str(tmp_path / "books"), "--output-dir", str(tmp_path / "out")],
    )  # fmt: skip
    assert rendered.exit_code == 0, rendered.output
    result = runner.invoke(
        main, ["master", "test-book", "--no-loudness", "--output-dir", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    assert "2 chapters" in result.output
    assert next((tmp_path / "out").glob("*/*.m4b")).is_file()
