"""Inter-voice loudness matching: one BS.1770 gain per voice per book, before concatenation."""

import math
import os
import subprocess

import numpy as np
import pyloudnorm
import pytest
import soundfile as sf
from click.testing import CliRunner

import seiyuu.assemble
import seiyuu.assemble.pipeline as pipeline
from seiyuu.assemble import PauseProfile, VoiceMatch, assemble_book, master_book
from seiyuu.assemble.voice_match import (
    LOUDNESS_CACHE_NAME,
    SegmentLevel,
    measure_segment,
    plan_voice_gains,
)
from seiyuu.cli import main
from seiyuu.engines import CANONICAL_SAMPLE_RATE
from seiyuu.ingest.models import BlockType
from seiyuu.render import MANIFEST_NAME, RenderedChapter, RenderedSegment, RenderManifest

SR = CANONICAL_SAMPLE_RATE
PAUSES = PauseProfile()


def lufs(samples: np.ndarray) -> float:
    return float(pyloudnorm.Meter(SR).integrated_loudness(samples.astype(np.float64)))


def noise_at(target_lufs: float, seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Seeded white noise scaled to a BS.1770 integrated loudness (scaled on >= 3 s, then
    truncated, so sub-block clips get the same level too)."""
    n = round(seconds * SR)
    x = np.random.default_rng(seed).uniform(-1, 1, max(n, 3 * SR)).astype(np.float64)
    x *= 10 ** ((target_lufs - lufs(x)) / 20)
    return x[:n].astype(np.float32)


def write_wav(book_dir, name: str, samples: np.ndarray) -> str:
    rel = f"cache/{name}.wav"
    (book_dir / "cache").mkdir(parents=True, exist_ok=True)
    sf.write(str(book_dir / rel), samples, SR, subtype="PCM_16")
    return rel


def write_manifest(book_dir, voiced: list[tuple[str, str]], narrator: str | None = "narr"):
    """voiced = [(voice_id, wav_rel), ...] -> one chapter, one block per segment."""
    segments = [
        RenderedSegment(
            block_id=f"ch001_b{i:04d}",
            type=BlockType.PARAGRAPH,
            wav=rel,
            duration_seconds=sf.info(str(book_dir / rel)).duration,
            voice_id=voice,
        )
        for i, (voice, rel) in enumerate(voiced, start=1)
    ]
    manifest = RenderManifest(
        book_id="vm-book",
        book_title="VM",
        chapters=[RenderedChapter(index=1, title="Chapter 1", segments=segments)],
        assignment={"narrator_voice_id": narrator} if narrator else None,
    )
    (book_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(), encoding="utf-8")
    return manifest


@pytest.fixture
def two_voice_dir(tmp_path):
    """Loud narrator (-18 LUFS) / quiet character (-26 LUFS, inside the default 9 dB clamp) /
    narrator, 3 s each."""
    book = tmp_path / "book"
    n1 = write_wav(book, "n1", noise_at(-18.0, seed=1))
    a1 = write_wav(book, "a1", noise_at(-26.0, seed=2))
    n2 = write_wav(book, "n2", noise_at(-18.0, seed=3))
    write_manifest(book, [("narr", n1), ("alice", a1), ("narr", n2)])
    return book


def regions(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The middle 2 s of segment 1 (narrator) and segment 2 (alice) in an assembled chapter."""
    start1 = PAUSES.chapter_lead_in
    start2 = start1 + 3.0 + PAUSES.paragraph

    def mid(start: float) -> np.ndarray:
        return samples[round((start + 0.5) * SR) : round((start + 2.5) * SR)]

    return mid(start1), mid(start2)


def decode_mp3(path, tmp_path) -> np.ndarray:
    out = tmp_path / f"{path.stem}.decoded.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), str(out)],
        check=True,
    )
    return sf.read(str(out), dtype="float32")[0]


def capture_chapters(monkeypatch) -> list[np.ndarray]:
    """Spy on _chapter_samples to see the exact pre-encode samples of each chapter."""
    seen: list[np.ndarray] = []
    real = pipeline._chapter_samples

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(out.copy())
        return out

    monkeypatch.setattr(pipeline, "_chapter_samples", spy)
    return seen


def counting_measure():
    calls: list[str] = []

    def measure(path):
        calls.append(path.name)
        return measure_segment(path)

    return measure, calls


# -- end to end ---------------------------------------------------------------------------


def test_assembled_voices_end_up_within_1db(two_voice_dir, tmp_path) -> None:
    msgs: list[str] = []
    result = assemble_book(two_voice_dir, voice_match=VoiceMatch(), progress=msgs.append)
    narr, alice = regions(decode_mp3(result.mp3_paths[0], tmp_path))
    assert abs(lufs(narr) - lufs(alice)) < 1.0
    assert abs(lufs(narr) - -18.0) < 1.0  # narrator is the reference: untouched
    assert msgs[0] == "matching voice loudness: measuring 3 segments (0 cached)..."
    line = next(m for m in msgs if m.startswith("matching voice loudness: 2 voices"))
    assert "2 voices" in line and "narrator narr" in line and "+8.0" in line


def test_disabled_keeps_the_jump(two_voice_dir, tmp_path) -> None:
    result = assemble_book(two_voice_dir, voice_match=None)
    narr, alice = regions(decode_mp3(result.mp3_paths[0], tmp_path))
    assert lufs(narr) - lufs(alice) > 7.0


def test_disabled_is_exactly_the_old_behaviour(two_voice_dir, monkeypatch) -> None:
    seen = capture_chapters(monkeypatch)
    msgs: list[str] = []
    assemble_book(two_voice_dir, voice_match=None, progress=msgs.append)
    manifest = RenderManifest.model_validate_json(
        (two_voice_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    # Pre-feature call shape (no gain map) produces the same bytes the disabled run used.
    monkeypatch.undo()
    expected = pipeline._chapter_samples(manifest.chapters[0], two_voice_dir, PAUSES, "narr")
    assert np.array_equal(seen[0], expected)
    assert not (two_voice_dir / LOUDNESS_CACHE_NAME).exists()  # nothing measured
    assert not any("matching voice loudness" in m for m in msgs)


def test_master_gets_the_same_gains(two_voice_dir, monkeypatch) -> None:
    seen = capture_chapters(monkeypatch)
    assemble_book(two_voice_dir, voice_match=VoiceMatch())
    master_book(two_voice_dir, voice_match=VoiceMatch())
    assert len(seen) == 2
    assert np.array_equal(seen[0], seen[1])
    narr, alice = regions(seen[1])
    assert abs(lufs(narr) - lufs(alice)) < 0.5


def test_single_voice_is_a_bit_identical_noop(tmp_path, monkeypatch) -> None:
    book = tmp_path / "book"
    a = write_wav(book, "a", noise_at(-30.0, seed=1))
    b = write_wav(book, "b", noise_at(-20.0, seed=2))
    write_manifest(book, [("solo", a), ("solo", b)], narrator=None)
    seen = capture_chapters(monkeypatch)
    msgs: list[str] = []
    assemble_book(book, voice_match=VoiceMatch(), progress=msgs.append)
    assemble_book(book, voice_match=None)
    assert np.array_equal(seen[0], seen[1])
    assert not (book / LOUDNESS_CACHE_NAME).exists()  # single voice: nothing measured
    assert not any("matching voice loudness" in m for m in msgs)


# -- planning -----------------------------------------------------------------------------


def _manifest(book_dir) -> RenderManifest:
    return RenderManifest.model_validate_json(
        (book_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )


def test_narrator_is_the_target_and_never_moves(two_voice_dir) -> None:
    plan = plan_voice_gains(_manifest(two_voice_dir), two_voice_dir, VoiceMatch())
    assert plan is not None
    assert plan.target_lufs == pytest.approx(-18.0, abs=0.2)
    assert plan.gains["narr"].gain_db == 0.0
    assert plan.gains["alice"].gain_db == pytest.approx(8.0, abs=0.3)
    assert "narr" not in plan.factors()  # exact no-op for the reference voice


def test_gain_is_clamped(tmp_path) -> None:
    book = tmp_path / "book"
    n = write_wav(book, "n", noise_at(-16.0, seed=1))
    q = write_wav(book, "q", noise_at(-40.0, seed=2))  # 24 dB quieter
    write_manifest(book, [("narr", n), ("alice", q)])
    plan = plan_voice_gains(_manifest(book), book, VoiceMatch(max_gain_db=6.0))
    g = plan.gains["alice"]
    assert g.clamped and g.gain_db == 6.0 and g.wanted_db == pytest.approx(24.0, abs=0.3)
    assert "clamped: alice (+24.0 -> +6.0 dB)" in plan.summary()


def test_loud_voice_is_turned_down_and_clamped_too(tmp_path) -> None:
    book = tmp_path / "book"
    n = write_wav(book, "n", noise_at(-30.0, seed=1))
    loud = write_wav(book, "l", noise_at(-14.0, seed=2))
    write_manifest(book, [("narr", n), ("bob", loud)])
    plan = plan_voice_gains(_manifest(book), book, VoiceMatch(max_gain_db=9.0))
    assert plan.gains["bob"].gain_db == -9.0 and plan.gains["bob"].clamped


def test_peak_guard_reduces_gain_instead_of_clipping(tmp_path) -> None:
    book = tmp_path / "book"
    n = write_wav(book, "n", noise_at(-18.0, seed=1))
    quiet = noise_at(-26.0, seed=2)
    quiet[SR] = 0.6  # one spike: +8 dB would push it to ~1.5 (hard clip)
    q = write_wav(book, "q", quiet)
    write_manifest(book, [("narr", n), ("alice", q)])
    match = VoiceMatch(peak_ceiling_dbfs=-1.0)
    plan = plan_voice_gains(_manifest(book), book, match)
    g = plan.gains["alice"]
    assert g.peak_limited and 0.0 < g.gain_db < g.wanted_db
    peak = float(np.max(np.abs(sf.read(str(book / q), dtype="float32")[0])))
    assert peak * g.factor == pytest.approx(10 ** (-1.0 / 20), rel=1e-4)
    chapter = _manifest(book).chapters[0]
    out = pipeline._chapter_samples(chapter, book, PAUSES, "narr", plan.factors())
    assert float(np.max(np.abs(out))) <= 10 ** (-1.0 / 20) + 1e-4
    assert "peak-limited: alice" in plan.summary()


def test_median_target_without_narrator(tmp_path) -> None:
    book = tmp_path / "book"
    a = write_wav(book, "a", noise_at(-30.0, seed=1))
    b = write_wav(book, "b", noise_at(-22.0, seed=2))
    c = write_wav(book, "c", noise_at(-16.0, seed=3))
    write_manifest(book, [("a", a), ("b", b), ("c", c)], narrator=None)
    plan = plan_voice_gains(_manifest(book), book, VoiceMatch())
    assert plan.target_source == "median"
    assert plan.target_lufs == pytest.approx(-22.0, abs=0.2)
    assert plan.gains["b"].gain_db == 0.0


def test_voice_loudness_is_duration_weighted_power_mean(tmp_path) -> None:
    book = tmp_path / "book"
    n = write_wav(book, "n", noise_at(-18.0, seed=1))
    a1 = write_wav(book, "a1", noise_at(-20.0, seconds=3.0, seed=2))
    a2 = write_wav(book, "a2", noise_at(-30.0, seconds=1.0, seed=3))
    write_manifest(book, [("narr", n), ("alice", a1), ("alice", a2)])
    plan = plan_voice_gains(_manifest(book), book, VoiceMatch())
    expected = 10 * math.log10((3 * 10 ** (-20 / 10) + 1 * 10 ** (-30 / 10)) / 4)
    assert plan.gains["alice"].loudness == pytest.approx(expected, abs=0.2)


def test_silent_and_short_segments_are_skipped(tmp_path) -> None:
    book = tmp_path / "book"
    assert measure_segment(book / write_wav(book, "z", np.zeros(SR, np.float32))).lufs is None
    short = noise_at(-10.0, seconds=0.3, seed=4)  # below the 0.4 s BS.1770 block
    short_level = measure_segment(book / write_wav(book, "s", short))
    assert short_level.lufs is None and short_level.peak > 0  # still counted for the peak

    n = write_wav(book, "n", noise_at(-18.0, seed=1))
    a = write_wav(book, "a", noise_at(-24.0, seed=2))
    write_manifest(
        book, [("narr", n), ("alice", a), ("alice", "cache/s.wav"), ("alice", "cache/z.wav")]
    )
    plan = plan_voice_gains(_manifest(book), book, VoiceMatch())
    # the loud short + silent segments do not drag alice's measurement
    assert plan.gains["alice"].loudness == pytest.approx(-24.0, abs=0.2)


def test_voice_with_only_unmeasurable_audio_is_left_alone(tmp_path) -> None:
    book = tmp_path / "book"
    n = write_wav(book, "n", noise_at(-18.0, seed=1))
    s = write_wav(book, "s", noise_at(-30.0, seconds=0.2, seed=2))
    write_manifest(book, [("narr", n), ("alice", s)])
    assert plan_voice_gains(_manifest(book), book, VoiceMatch()) is None


# -- measurement cache --------------------------------------------------------------------


def test_cache_is_reused_then_invalidated_by_a_changed_wav(two_voice_dir) -> None:
    manifest = _manifest(two_voice_dir)
    measure, calls = counting_measure()
    first = plan_voice_gains(manifest, two_voice_dir, VoiceMatch(), measure=measure)
    assert sorted(calls) == ["a1.wav", "n1.wav", "n2.wav"]
    assert (two_voice_dir / LOUDNESS_CACHE_NAME).is_file()

    calls.clear()
    second = plan_voice_gains(manifest, two_voice_dir, VoiceMatch(), measure=measure)
    assert calls == []  # assemble -> master: nothing re-measured
    assert second.gains["alice"].gain_db == first.gains["alice"].gain_db

    # a re-rendered segment (new content, new size/mtime) is re-measured
    wav = two_voice_dir / "cache" / "a1.wav"
    sf.write(str(wav), noise_at(-22.0, seconds=2.0, seed=9), SR, subtype="PCM_16")
    st = wav.stat()
    os.utime(wav, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    third = plan_voice_gains(manifest, two_voice_dir, VoiceMatch(), measure=measure)
    assert calls == ["a1.wav"]
    assert third.gains["alice"].gain_db == pytest.approx(4.0, abs=0.3)


def test_assemble_then_master_measures_once(two_voice_dir, monkeypatch) -> None:
    import seiyuu.assemble.voice_match as vm

    measure, calls = counting_measure()
    monkeypatch.setattr(vm, "measure_segment", measure)
    assemble_book(two_voice_dir, voice_match=VoiceMatch())
    assert len(calls) == 3
    master_book(two_voice_dir, voice_match=VoiceMatch())
    assert len(calls) == 3


@pytest.mark.parametrize("junk", ["{not json", '{"version": 999, "segments": {}}', "[1, 2]"])
def test_corrupt_cache_recomputes_without_error(two_voice_dir, junk) -> None:
    (two_voice_dir / LOUDNESS_CACHE_NAME).write_text(junk, encoding="utf-8")
    measure, calls = counting_measure()
    plan = plan_voice_gains(_manifest(two_voice_dir), two_voice_dir, VoiceMatch(), measure=measure)
    assert len(calls) == 3 and plan is not None


def test_cache_entry_with_bad_fields_is_remeasured(two_voice_dir) -> None:
    manifest = _manifest(two_voice_dir)
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch())
    import json

    path = two_voice_dir / LOUDNESS_CACHE_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["segments"]["cache/a1.wav"]["seconds"] = "nope"
    path.write_text(json.dumps(data), encoding="utf-8")
    measure, calls = counting_measure()
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch(), measure=measure)
    assert calls == ["a1.wav"]


def test_cached_measurements_match_fake_levels(tmp_path) -> None:
    """The plan works purely from (cached or fresh) SegmentLevels — no audio decoding needed."""
    book = tmp_path / "book"
    n = write_wav(book, "n", np.full(SR, 0.01, np.float32))
    a = write_wav(book, "a", np.full(SR, 0.01, np.float32))
    write_manifest(book, [("narr", n), ("alice", a)])
    levels = {"n.wav": SegmentLevel(-20.0, 1.0, 0.1), "a.wav": SegmentLevel(-23.0, 1.0, 0.1)}
    plan = plan_voice_gains(_manifest(book), book, VoiceMatch(), measure=lambda p: levels[p.name])
    assert plan.gains["alice"].gain_db == pytest.approx(3.0)


# -- config wiring ------------------------------------------------------------------------


def test_cli_match_voices_flag(tmp_path, monkeypatch) -> None:
    book = tmp_path / "out" / "vm-book"
    book.mkdir(parents=True)
    (book / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    captured: dict = {}

    def fake_assemble(book_dir, **kwargs):
        captured.update(kwargs)
        return seiyuu.assemble.AssembleResult(mp3_paths=[], total_seconds=0.0)

    monkeypatch.setattr(seiyuu.assemble, "assemble_book", fake_assemble)
    runner = CliRunner()
    args = ["assemble", "vm-book", "--output-dir", str(tmp_path / "out")]
    assert runner.invoke(main, args).exit_code == 0
    assert isinstance(captured["voice_match"], VoiceMatch)  # settings default: on
    assert runner.invoke(main, [*args, "--no-match-voices"]).exit_code == 0
    assert captured["voice_match"] is None


def test_api_voice_match_null_semantics(tmp_path) -> None:
    from seiyuu.api.handlers import _voice_match
    from seiyuu.api.schemas import LoudnessWrite
    from test_api_m6b1 import make_settings

    cfg = make_settings(tmp_path)
    assert _voice_match(cfg, None) == VoiceMatch(max_gain_db=cfg.voice_loudness_max_gain_db)
    assert _voice_match(cfg, LoudnessWrite(match_voices=False)) is None
    # independent of chapter normalization
    assert _voice_match(cfg, LoudnessWrite(enabled=False)) is not None
    off = make_settings(tmp_path, voice_loudness_match=False)
    assert _voice_match(off, None) is None
    assert _voice_match(off, LoudnessWrite(match_voices=True)) is not None


# -- review fixes: interrupted passes, errors, progress, bounds ---------------------------


class _Cancelled(Exception):
    pass


def _cancel_on_call(n: int):
    """A check_cancel that raises on its n-th call (1-based)."""
    calls = {"n": 0}

    def check() -> None:
        calls["n"] += 1
        if calls["n"] >= n:
            raise _Cancelled

    return check


def _cache_segments(book_dir) -> dict:
    import json

    return json.loads((book_dir / LOUDNESS_CACHE_NAME).read_text(encoding="utf-8"))["segments"]


def test_cancelled_pass_keeps_every_cached_entry(two_voice_dir) -> None:
    manifest = _manifest(two_voice_dir)
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch())
    before = _cache_segments(two_voice_dir)
    assert len(before) == 3
    with pytest.raises(_Cancelled):
        plan_voice_gains(manifest, two_voice_dir, VoiceMatch(), check_cancel=_cancel_on_call(1))
    assert _cache_segments(two_voice_dir) == before


def test_cancel_mid_measure_keeps_work_done_and_unreached_entries(two_voice_dir) -> None:
    manifest = _manifest(two_voice_dir)
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch())
    before = _cache_segments(two_voice_dir)
    # a1 re-rendered -> needs measuring; cancel at the measuring pass's first check
    wav = two_voice_dir / "cache" / "a1.wav"
    st = wav.stat()
    os.utime(wav, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    with pytest.raises(_Cancelled):
        plan_voice_gains(manifest, two_voice_dir, VoiceMatch(), check_cancel=_cancel_on_call(2))
    after = _cache_segments(two_voice_dir)
    assert set(after) == set(before)  # nothing dropped
    assert after["cache/n1.wav"] == before["cache/n1.wav"]


def test_cache_is_pruned_to_the_manifest(two_voice_dir) -> None:
    import json

    manifest = _manifest(two_voice_dir)
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch())
    path = two_voice_dir / LOUDNESS_CACHE_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["segments"]["cache/gone.wav"] = {"size": 1, "mtime_ns": 1, "lufs": None}
    path.write_text(json.dumps(data), encoding="utf-8")
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch())
    assert "cache/gone.wav" not in _cache_segments(two_voice_dir)


def test_unreadable_segment_names_chapter_and_block(two_voice_dir) -> None:
    from seiyuu.assemble import AssembleError

    manifest = _manifest(two_voice_dir)
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch())
    (two_voice_dir / "cache" / "a1.wav").write_bytes(b"not a wav at all")
    with pytest.raises(AssembleError, match=r"chapter=1 block=ch001_b0002"):
        plan_voice_gains(manifest, two_voice_dir, VoiceMatch())
    # the good entries survive the failed pass
    assert {"cache/n1.wav", "cache/n2.wav"} <= set(_cache_segments(two_voice_dir))


def test_master_cancel_while_measuring_leaves_no_work_dir(two_voice_dir) -> None:
    with pytest.raises(_Cancelled):
        master_book(two_voice_dir, voice_match=VoiceMatch(), check_cancel=_cancel_on_call(1))
    assert not (two_voice_dir / "master").exists()


def test_measuring_pass_announces_itself(two_voice_dir) -> None:
    manifest = _manifest(two_voice_dir)
    msgs: list[str] = []
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch(), progress=msgs.append)
    assert msgs == ["matching voice loudness: measuring 3 segments (0 cached)..."]
    msgs.clear()
    plan_voice_gains(manifest, two_voice_dir, VoiceMatch(), progress=msgs.append)
    assert msgs == []  # fully cached: nothing slow to announce


def test_negative_max_gain_is_rejected(tmp_path) -> None:
    from pydantic import ValidationError

    from test_api_m6b1 import make_settings

    with pytest.raises(ValueError, match="max_gain_db"):
        VoiceMatch(max_gain_db=-3.0)
    with pytest.raises(ValidationError):
        make_settings(tmp_path, voice_loudness_max_gain_db=-3.0)
    assert VoiceMatch(max_gain_db=0.0).max_gain_db == 0.0


def test_measuring_pass_reports_periodic_progress(two_voice_dir, monkeypatch) -> None:
    import seiyuu.assemble.voice_match as vm

    monkeypatch.setattr(vm, "_PROGRESS_EVERY", 1)
    msgs: list[str] = []
    plan_voice_gains(_manifest(two_voice_dir), two_voice_dir, VoiceMatch(), progress=msgs.append)
    assert msgs == [
        "matching voice loudness: measuring 3 segments (0 cached)...",
        "matching voice loudness: measured 1/3 segments",
        "matching voice loudness: measured 2/3 segments",
    ]
