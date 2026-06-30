import subprocess

import numpy as np
import pytest
from click.testing import CliRunner

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.assemble import AssembleError, PauseProfile, assemble_book
from seiyuu.assemble.pipeline import _chapter_samples
from seiyuu.cli import main
from seiyuu.engines import CANONICAL_SAMPLE_RATE, AudioFile
from seiyuu.ingest.models import BlockType
from seiyuu.render import RenderedChapter, RenderedSegment, render_book

PAUSES = PauseProfile()


def ffprobe_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


@pytest.fixture
def rendered_book_dir(tmp_path):
    render_book(make_book(), FakeEngine(), "test_voice", tmp_path / "book")
    return tmp_path / "book"


def one_second_wav(book_dir, name) -> str:
    rel = f"cache/{name}.wav"
    AudioFile(samples=np.full(CANONICAL_SAMPLE_RATE, 0.1, dtype=np.float32)).save(book_dir / rel)
    return rel


def seg(block_id, kind, wav=None, voice=None) -> RenderedSegment:
    return RenderedSegment(
        block_id=block_id,
        type=kind,
        wav=wav,
        duration_seconds=1.0 if wav else 0.0,
        voice_id=voice,
    )


def test_pause_math_exact(tmp_path) -> None:
    a = one_second_wav(tmp_path, "a")
    b = one_second_wav(tmp_path, "b")
    c = one_second_wav(tmp_path, "c")
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.HEADING, a),
            seg("ch001_b0002", BlockType.PARAGRAPH, b),
            seg("ch001_b0003", BlockType.SCENE_BREAK),
            seg("ch001_b0004", BlockType.PARAGRAPH, c),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES)
    parts_seconds = [
        PAUSES.chapter_lead_in,
        1.0,  # heading audio
        PAUSES.after_heading,
        1.0,  # paragraph b
        PAUSES.scene_break,  # replaces the paragraph gap
        1.0,  # paragraph c
        PAUSES.chapter_lead_out,
    ]
    assert len(samples) == sum(round(s * CANONICAL_SAMPLE_RATE) for s in parts_seconds)


def test_paragraph_gap_between_paragraphs(tmp_path) -> None:
    a = one_second_wav(tmp_path, "a")
    b = one_second_wav(tmp_path, "b")
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a),
            seg("ch001_b0002", BlockType.PARAGRAPH, b),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES)
    parts = [PAUSES.chapter_lead_in, 1.0, PAUSES.paragraph, 1.0, PAUSES.chapter_lead_out]
    assert len(samples) == sum(round(s * CANONICAL_SAMPLE_RATE) for s in parts)


def test_no_gap_between_segments_of_one_block(tmp_path) -> None:
    # A multi-voice paragraph yields two segments sharing one block_id: no pause between them,
    # but the normal paragraph gap before the next block.
    a = one_second_wav(tmp_path, "a")
    b = one_second_wav(tmp_path, "b")
    c = one_second_wav(tmp_path, "c")
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a),
            seg("ch001_b0001", BlockType.PARAGRAPH, b),  # same block, second voice
            seg("ch001_b0002", BlockType.PARAGRAPH, c),  # next block
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES)
    parts = [PAUSES.chapter_lead_in, 1.0, 1.0, PAUSES.paragraph, 1.0, PAUSES.chapter_lead_out]
    assert len(samples) == sum(round(s * CANONICAL_SAMPLE_RATE) for s in parts)


def test_dialogue_exchange_uses_short_gap(tmp_path) -> None:
    # narration -> dialogue -> dialogue -> narration: only the dialogue<->dialogue transition
    # gets the short beat; the others get the paragraph gap.
    a = one_second_wav(tmp_path, "a")
    b = one_second_wav(tmp_path, "b")
    c = one_second_wav(tmp_path, "c")
    d = one_second_wav(tmp_path, "d")
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="narr"),
            seg("ch001_b0002", BlockType.PARAGRAPH, b, voice="alice"),
            seg("ch001_b0003", BlockType.PARAGRAPH, c, voice="bob"),
            seg("ch001_b0004", BlockType.PARAGRAPH, d, voice="narr"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES, "narr")
    parts = [
        PAUSES.chapter_lead_in,
        1.0,
        PAUSES.paragraph,  # narration -> dialogue
        1.0,
        PAUSES.dialogue,  # dialogue -> dialogue (the short beat)
        1.0,
        PAUSES.paragraph,  # dialogue -> narration
        1.0,
        PAUSES.chapter_lead_out,
    ]
    assert len(samples) == sum(round(s * CANONICAL_SAMPLE_RATE) for s in parts)


def test_no_narrator_id_disables_dialogue_pacing(tmp_path) -> None:
    # single-voice (no narrator id): two dialogue-looking blocks still get the paragraph gap
    a = one_second_wav(tmp_path, "a")
    b = one_second_wav(tmp_path, "b")
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="alice"),
            seg("ch001_b0002", BlockType.PARAGRAPH, b, voice="bob"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES)  # narrator_voice_id defaults to None
    parts = [PAUSES.chapter_lead_in, 1.0, PAUSES.paragraph, 1.0, PAUSES.chapter_lead_out]
    assert len(samples) == sum(round(s * CANONICAL_SAMPLE_RATE) for s in parts)


def test_assemble_end_to_end(rendered_book_dir) -> None:
    result = assemble_book(rendered_book_dir)
    assert [p.name for p in result.mp3_paths] == ["ch001.mp3", "ch002.mp3"]
    for path in result.mp3_paths:
        assert path.is_file()
    # mp3 durations match the assembled sample counts (mp3 padding tolerance)
    probed = sum(ffprobe_duration(p) for p in result.mp3_paths)
    assert probed == pytest.approx(result.total_seconds, abs=0.2)


def test_mp3_metadata(rendered_book_dir) -> None:
    result = assemble_book(rendered_book_dir)
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=title,track,album",
            "-of",
            "default=noprint_wrappers=1",
            str(result.mp3_paths[0]),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "TAG:title=Chapter 1" in out
    assert "TAG:track=1" in out
    assert "TAG:album=Test Book" in out


def test_missing_segment_audio_is_loud(rendered_book_dir) -> None:
    wavs = list((rendered_book_dir / "cache").glob("*.wav"))
    wavs[0].unlink()
    with pytest.raises(AssembleError, match="missing segment audio"):
        assemble_book(rendered_book_dir)


def test_no_manifest_is_loud(tmp_path) -> None:
    with pytest.raises(AssembleError, match="seiyuu render"):
        assemble_book(tmp_path)


def test_assemble_cli(tmp_path, monkeypatch) -> None:
    import seiyuu.engines
    from seiyuu.ingest import write_normalized

    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: FakeEngine())
    book_id = "test-book-00000000"
    write_normalized(make_book(), tmp_path / "books")
    runner = CliRunner()
    rendered = runner.invoke(
        main,
        [
            "render",
            book_id,
            "--voice",
            "v",
            "--books-dir",
            str(tmp_path / "books"),
            "--output-dir",
            str(tmp_path / "out"),
        ],  # fmt: skip
    )
    assert rendered.exit_code == 0, rendered.output

    result = runner.invoke(main, ["assemble", "test-book", "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "chapter MP3s" in result.output
    assert (tmp_path / "out" / book_id / "chapters" / "ch001.mp3").is_file()
    assert (tmp_path / "out" / book_id / "chapters" / "ch002.mp3").is_file()


def test_assemble_cli_without_render(tmp_path) -> None:
    (tmp_path / "out").mkdir()
    result = CliRunner().invoke(main, ["assemble", "nope", "--output-dir", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "seiyuu render" in result.output
