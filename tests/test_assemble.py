import subprocess

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.assemble import AssembleError, LoudnessTarget, PauseProfile, assemble_book
from seiyuu.assemble.pipeline import _chapter_samples, _loudnorm_filter, _measure_loudness
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


SR = CANONICAL_SAMPLE_RATE


def edged_wav(book_dir, name, edge: float, *, level: float = 0.3, floor: float = 0.0) -> str:
    """A 1 s constant 'voiced' clip with `edge` seconds of edge silence (at `floor`) per side."""
    pad = np.full(round(edge * SR), floor, dtype=np.float32)
    samples = np.concatenate([pad, np.full(SR, level, dtype=np.float32), pad])
    rel = f"cache/{name}.wav"
    AudioFile(samples=samples).save(book_dir / rel)
    return rel


def read_clip(book_dir, rel) -> np.ndarray:
    return sf.read(str(book_dir / rel), dtype="float32")[0]


def test_quote_split_join_trims_edge_silence(tmp_path) -> None:
    # “Yes,” he said — two clips of ONE block, each with 0.5 s of edge silence: the join keeps
    # 0.20 s of the first clip's tail + 0.12 s of the second's head (a comma-length pause).
    a = edged_wav(tmp_path, "a", 0.5)
    b = edged_wav(tmp_path, "b", 0.5)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="alice"),
            seg("ch001_b0001", BlockType.PARAGRAPH, b, voice="narr"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES, "narr")
    voiced = np.flatnonzero(samples)
    first_end = round((PAUSES.chapter_lead_in + 0.5 + 1.0) * SR)
    gap = np.diff(voiced).max() - 1  # the only run of silence between voiced samples
    assert gap == round(PAUSES.join_tail_keep * SR) + round(PAUSES.join_head_keep * SR)
    assert voiced[0] == round((PAUSES.chapter_lead_in + 0.5) * SR)
    assert voiced[np.argmax(np.diff(voiced))] == first_end - 1
    parts = [PAUSES.chapter_lead_in, 0.5, 1.0, PAUSES.join_tail_keep]
    parts += [PAUSES.join_head_keep, 1.0, 0.5, PAUSES.chapter_lead_out]
    assert len(samples) == sum(round(s * SR) for s in parts)


def test_quote_split_join_keeps_tail_then_head_with_faded_cuts(tmp_path) -> None:
    # Different floors tell the kept edges apart (zeros can't): exactly join_tail_keep of the
    # first clip's -54 dBFS tail, then join_head_keep of the second clip's -60 dBFS head, and
    # each cut ramps to zero over 5 ms so a cut in the noise floor can't click.
    a = edged_wav(tmp_path, "a", 0.5, floor=0.002)
    b = edged_wav(tmp_path, "b", 0.5, floor=0.001)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="alice"),
            seg("ch001_b0001", BlockType.PARAGRAPH, b, voice="narr"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES, "narr")
    clip_a, clip_b = read_clip(tmp_path, a), read_clip(tmp_path, b)
    edge, fade = round(0.5 * SR), SR // 200
    tail_keep, head_keep = round(PAUSES.join_tail_keep * SR), round(PAUSES.join_head_keep * SR)
    start = round(PAUSES.chapter_lead_in * SR) + edge + SR  # just past the first clip's voice
    assert samples[start - 1] == clip_a[edge + SR - 1]  # the voice itself is untouched

    tail = samples[start : start + tail_keep]
    expected_tail = np.full(tail_keep, clip_a[-1], dtype=np.float32)
    expected_tail[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    np.testing.assert_allclose(tail, expected_tail, rtol=1e-6, atol=0)
    assert tail[-1] == 0.0 and tail[-fade - 1] == clip_a[-1]

    head = samples[start + tail_keep : start + tail_keep + head_keep]
    expected_head = np.full(head_keep, clip_b[0], dtype=np.float32)
    expected_head[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    np.testing.assert_allclose(head, expected_head, rtol=1e-6, atol=0)
    assert head[0] == 0.0 and head[fade] == clip_b[0]
    assert clip_a[-1] != clip_b[0]  # the floors really differ, so a swap would show
    assert samples[start + tail_keep + head_keep] == clip_b[edge]  # then the second voice


def test_quote_split_join_keeps_quiet_speech_above_the_threshold(tmp_path) -> None:
    # A decaying speech tail at -40 dBFS (well under the voice, over the -50 dBFS line) is
    # speech: all 0.3 s of it survives, and only the true silence after it is trimmed.
    voiced, quiet, gap = SR, round(0.3 * SR), round(0.5 * SR)
    a = "cache/a.wav"
    AudioFile(
        samples=np.concatenate(
            [
                np.full(voiced, 0.3, dtype=np.float32),
                np.full(quiet, 0.01, dtype=np.float32),
                np.zeros(gap, dtype=np.float32),
            ]
        )
    ).save(tmp_path / a)
    b = edged_wav(tmp_path, "b", 0.5)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="alice"),
            seg("ch001_b0001", BlockType.PARAGRAPH, b, voice="narr"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES, "narr")
    start = round(PAUSES.chapter_lead_in * SR) + voiced
    clip_a = read_clip(tmp_path, a)
    np.testing.assert_array_equal(samples[start : start + quiet], clip_a[voiced : voiced + quiet])
    parts = [PAUSES.chapter_lead_in, 1.0, 0.3, PAUSES.join_tail_keep]
    parts += [PAUSES.join_head_keep, 1.0, 0.5, PAUSES.chapter_lead_out]
    assert len(samples) == sum(round(s * SR) for s in parts)


def test_quote_split_join_never_trims_more_than_the_silence(tmp_path) -> None:
    # 0.1 s edges are already under both keeps: nothing to trim, bit-identical concatenation.
    a = edged_wav(tmp_path, "a", 0.1)
    b = edged_wav(tmp_path, "b", 0.1)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a),
            seg("ch001_b0001", BlockType.PARAGRAPH, b),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES)
    lead, out = np.zeros(round(PAUSES.chapter_lead_in * SR)), np.zeros(SR)
    expected = np.concatenate([lead, read_clip(tmp_path, a), read_clip(tmp_path, b), out])
    np.testing.assert_array_equal(samples, expected.astype(np.float32))


def test_silent_beat_inside_a_block_keeps_both_facing_edges(tmp_path) -> None:
    # He paused. “……” Then he said — a beat inside the block is a deliberate held silence, so
    # neither join next to it is trimmed.
    a = edged_wav(tmp_path, "a", 0.5)
    beat = "cache/beat.wav"
    AudioFile(samples=np.zeros(round(0.2 * SR), dtype=np.float32)).save(tmp_path / beat)
    c = edged_wav(tmp_path, "c", 0.5)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a),
            seg("ch001_b0001", BlockType.PARAGRAPH, beat),
            seg("ch001_b0001", BlockType.PARAGRAPH, c),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES)
    parts = [PAUSES.chapter_lead_in, 2.0, 0.2, 2.0, PAUSES.chapter_lead_out]
    assert len(samples) == sum(round(s * SR) for s in parts)


def test_joins_between_blocks_are_not_trimmed(tmp_path) -> None:
    a = edged_wav(tmp_path, "a", 0.5)
    b = edged_wav(tmp_path, "b", 0.5)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a),
            seg("ch001_b0002", BlockType.PARAGRAPH, b),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES)
    expected = np.concatenate(
        [
            np.zeros(round(PAUSES.chapter_lead_in * SR)),
            read_clip(tmp_path, a),
            np.zeros(round(PAUSES.paragraph * SR)),
            read_clip(tmp_path, b),
            np.zeros(round(PAUSES.chapter_lead_out * SR)),
        ]
    )
    np.testing.assert_array_equal(samples, expected.astype(np.float32))


def test_join_silence_is_measured_before_voice_gain(tmp_path) -> None:
    # A -56 dBFS tail is silence on the raw clip; a x4 voice gain would lift it over the -50
    # dBFS threshold, so measuring after the gain would wrongly keep it.
    a = edged_wav(tmp_path, "a", 0.5, floor=0.0015)
    b = edged_wav(tmp_path, "b", 0.5)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="loud"),
            seg("ch001_b0001", BlockType.PARAGRAPH, b, voice="narr"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES, "narr", {"loud": 4.0})
    parts = [PAUSES.chapter_lead_in, 0.5, 1.0, PAUSES.join_tail_keep]
    parts += [PAUSES.join_head_keep, 1.0, 0.5, PAUSES.chapter_lead_out]
    assert len(samples) == sum(round(s * SR) for s in parts)


def test_loudnorm_filter_two_pass_includes_measured() -> None:
    target = LoudnessTarget()
    assert _loudnorm_filter(target) == "loudnorm=I=-18.0:TP=-1.5:LRA=11.0"
    measured = {
        "input_i": "-30.0",
        "input_tp": "-9.0",
        "input_lra": "5.0",
        "input_thresh": "-40.0",
        "target_offset": "0.5",
    }
    f = _loudnorm_filter(target, measured)
    assert "measured_I=-30.0" in f and "offset=0.5" in f and "linear=true" in f


def test_measure_loudness_parses_ffmpeg_json(tmp_path) -> None:
    import soundfile as sf

    wav = tmp_path / "tone.wav"
    t = np.arange(CANONICAL_SAMPLE_RATE) / CANONICAL_SAMPLE_RATE
    sf.write(
        str(wav),
        (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"),
        CANONICAL_SAMPLE_RATE,
        subtype="PCM_16",
    )
    measured = _measure_loudness(wav, LoudnessTarget())
    assert "input_i" in measured and "target_offset" in measured


def test_assemble_with_loudness_preserves_duration(rendered_book_dir) -> None:
    result = assemble_book(rendered_book_dir, loudness=LoudnessTarget())
    assert [p.name for p in result.mp3_paths] == ["ch001.mp3", "ch002.mp3"]
    for path in result.mp3_paths:
        assert path.is_file()
    # loudnorm changes gain, not length
    probed = sum(ffprobe_duration(p) for p in result.mp3_paths)
    assert probed == pytest.approx(result.total_seconds, abs=0.2)


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


def test_block_edges_are_untouched_unless_trim_block_edges(tmp_path) -> None:
    # Off (the default), a paragraph join is the clips' own edges plus the gap, as before.
    a = edged_wav(tmp_path, "a", 0.5)
    b = edged_wav(tmp_path, "b", 0.4)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="narr"),
            seg("ch001_b0002", BlockType.PARAGRAPH, b, voice="narr"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, PAUSES, "narr")
    gap = np.diff(np.flatnonzero(samples)).max() - 1
    assert gap == round(0.5 * SR) + round(PAUSES.paragraph * SR) + round(0.4 * SR)


def test_trim_block_edges_cuts_both_edges_then_adds_the_gap(tmp_path) -> None:
    # On: the first clip's tail is cut to block_tail_keep, the next clip's head to
    # block_head_keep, and the paragraph gap sits between them. Speech is never shortened.
    from dataclasses import replace

    pauses = replace(PAUSES, trim_block_edges=True)
    a = edged_wav(tmp_path, "a", 0.5)
    b = edged_wav(tmp_path, "b", 0.4)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="narr"),
            seg("ch001_b0002", BlockType.PARAGRAPH, b, voice="narr"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, pauses, "narr")
    voiced = np.flatnonzero(samples)
    gap = np.diff(voiced).max() - 1
    keep = round(pauses.block_tail_keep * SR) + round(pauses.block_head_keep * SR)
    assert gap == keep + round(pauses.paragraph * SR)
    assert len(voiced) == 2 * SR  # every voiced sample survives
    # the chapter's outer edges (lead-in side of the first clip, tail of the last) are kept
    assert voiced[0] == round((pauses.chapter_lead_in + 0.5) * SR)
    assert len(samples) - 1 - voiced[-1] == round((0.4 + pauses.chapter_lead_out) * SR)


def test_trim_block_edges_leaves_a_silent_beat_whole(tmp_path) -> None:
    # A beat between two spoken clips keeps its full length; the spoken clips facing it
    # are still trimmed.
    from dataclasses import replace

    pauses = replace(PAUSES, trim_block_edges=True)
    a = edged_wav(tmp_path, "a", 0.5)
    b = edged_wav(tmp_path, "b", 0.5)
    beat_rel = "cache/beat.wav"
    AudioFile(samples=np.zeros(round(0.2 * SR), dtype=np.float32)).save(tmp_path / beat_rel)
    chapter = RenderedChapter(
        index=1,
        title="T",
        segments=[
            seg("ch001_b0001", BlockType.PARAGRAPH, a, voice="narr"),
            seg("ch001_b0002", BlockType.PARAGRAPH, beat_rel, voice="narr"),
            seg("ch001_b0003", BlockType.PARAGRAPH, b, voice="narr"),
        ],
    )
    samples = _chapter_samples(chapter, tmp_path, pauses, "narr")
    gap = np.diff(np.flatnonzero(samples)).max() - 1
    expected = [pauses.block_tail_keep, pauses.paragraph, 0.2, pauses.paragraph]
    expected.append(pauses.block_head_keep)
    assert gap == sum(round(s * SR) for s in expected)
