"""Assembly stage: manifest + cached segments → per-chapter MP3s.

Pause logic: scene_break → long, after-heading → extra, dialogue↔dialogue → short beat,
otherwise the paragraph gap. Optional inter-voice loudness matching (`voice_match.py`) first
gives every voice ONE book-wide gain toward the narrator's BS.1770 loudness, applied in memory
before concatenation, so speaker changes don't jump in volume (cached segment WAVs are never
modified; single-voice renders are a no-op). Optional EBU R128 loudness normalization
(two-pass loudnorm) then brings each chapter to a target LUFS. The chaptered .m4b is built by
the `master` stage from these same chapters. ffmpeg (on PATH) is the only way audio containers
are touched.
"""

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from seiyuu.assemble.errors import AssembleError
from seiyuu.assemble.voice_match import VoiceMatch, plan_voice_gains
from seiyuu.engines import CANONICAL_SAMPLE_RATE
from seiyuu.ingest.models import BlockType
from seiyuu.render import MANIFEST_NAME, RenderedChapter, RenderManifest


@dataclass(frozen=True)
class PauseProfile:
    """Silence durations in seconds. Tune freely.

    `dialogue` is the short beat between two consecutive dialogue turns (back-and-forth between
    characters); the longer `paragraph` gap applies to narration and dialogue↔narration
    transitions. Dialogue pacing only kicks in for multi-voice renders (it needs the narrator
    voice id to tell dialogue from narration).

    `join_tail_keep` / `join_head_keep` apply only where a block was split on a quote boundary
    (“Yes,” he said, “I know.” is three clips with no gap between them): each clip's own edge
    silence is trimmed down to these, so the split reads like the comma it replaces (~0.32 s)
    instead of ~0.8 s of Kokoro clip edges mid-sentence. Never exposed in the API/CLI.

    `trim_block_edges` (off by default) does the same at joins BETWEEN blocks, down to
    `block_tail_keep` / `block_head_keep`, before the gap is added. Kokoro clips carry ~0.33 s
    of lead and ~0.5 s of tail silence, so a 0.6 s paragraph gap measures ~1.4 s untrimmed;
    trimmed it is ~0.9 s. Speech itself is untouched. A silent beat keeps its own edges.
    """

    paragraph: float = 0.6
    after_heading: float = 1.2
    scene_break: float = 1.8
    dialogue: float = 0.35
    chapter_lead_in: float = 0.5
    chapter_lead_out: float = 1.0
    join_tail_keep: float = 0.20
    join_head_keep: float = 0.12
    trim_block_edges: bool = False
    block_tail_keep: float = 0.20
    block_head_keep: float = 0.10


@dataclass(frozen=True)
class LoudnessTarget:
    """EBU R128 loudnorm targets. -18 LUFS integrated suits audiobooks (Audible guidance is
    -18 to -20); TP is the true-peak ceiling (dBTP), LRA the allowed loudness range."""

    i: float = -18.0
    tp: float = -1.5
    lra: float = 11.0


@dataclass
class AssembleResult:
    mp3_paths: list[Path]
    total_seconds: float


@dataclass
class MasterResult:
    m4b_path: Path
    total_seconds: float  # final runtime after any target-duration tempo change
    chapters: int
    tempo: float = 1.0  # atempo applied (1.0 == none)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * CANONICAL_SAMPLE_RATE), dtype=np.float32)


# Edge trimming at joins inside one block: 10 ms frames whose RMS is under -50 dBFS count as
# silence (engine clip edges sit near -80 dBFS, speech tails at -35 to -55), and each cut gets
# a 5 ms fade so a cut in the noise floor can't click.
_EDGE_FRAME = CANONICAL_SAMPLE_RATE // 100
_EDGE_SILENCE = 10 ** (-50 / 20)
_EDGE_FADE = CANONICAL_SAMPLE_RATE // 200


def _leading_silence(samples: np.ndarray) -> int:
    """Samples of leading silence, in whole 10 ms frames; ``len(samples)`` for a clip with
    nothing above the threshold (a silent beat)."""
    for start in range(0, len(samples), _EDGE_FRAME):
        frame = samples[start : start + _EDGE_FRAME].astype(np.float64)
        if np.sqrt(np.mean(frame**2)) >= _EDGE_SILENCE:
            return start
    return len(samples)


def _trim_tail(prev: np.ndarray, prev_raw: np.ndarray, keep: float) -> np.ndarray:
    """``prev`` (the placed, possibly gained clip) with its tail silence, measured on
    ``prev_raw``, cut down to ``keep`` seconds and a short fade-out. A silent clip (a beat)
    and a tail already within ``keep`` come back unchanged."""
    tail = _leading_silence(prev_raw[::-1])
    drop = 0 if tail >= len(prev_raw) else max(0, tail - round(keep * CANONICAL_SAMPLE_RATE))
    if not drop:
        return prev
    prev = prev[: len(prev) - drop].copy()
    fade = min(_EDGE_FADE, len(prev))
    prev[len(prev) - fade :] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return prev


def _trim_head(cur_raw: np.ndarray, keep: float) -> np.ndarray:
    """``cur_raw`` with its lead silence cut down to ``keep`` seconds and a short fade-in; a
    silent clip (a beat) and a lead already within ``keep`` come back unchanged."""
    head = _leading_silence(cur_raw)
    drop = 0 if head >= len(cur_raw) else max(0, head - round(keep * CANONICAL_SAMPLE_RATE))
    if not drop:
        return cur_raw
    cur_raw = cur_raw[drop:].copy()
    fade = min(_EDGE_FADE, len(cur_raw))
    cur_raw[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    return cur_raw


def _trim_join(
    prev: np.ndarray, prev_raw: np.ndarray, cur_raw: np.ndarray, pauses: PauseProfile
) -> tuple[np.ndarray, np.ndarray]:
    """Trim the edge silence where two clips of ONE block meet: the previous clip's tail down
    to ``join_tail_keep`` and the next clip's head down to ``join_head_keep`` (never more than
    the silence measured). Silence is measured on the raw clips (before voice gain); ``prev``
    is the previous clip as already placed (possibly gained). A silent beat on either side
    keeps every edge: that held silence is deliberate."""
    if _leading_silence(prev_raw) >= len(prev_raw) or _leading_silence(cur_raw) >= len(cur_raw):
        return prev, cur_raw
    return (
        _trim_tail(prev, prev_raw, pauses.join_tail_keep),
        _trim_head(cur_raw, pauses.join_head_keep),
    )


def _is_dialogue(seg, narrator_voice_id: str | None) -> bool:
    """A segment voiced by a character (not the narrator) — multi-voice only. Single-voice
    renders have no narrator id, so this is always False and dialogue pacing is skipped."""
    return (
        narrator_voice_id is not None
        and seg.voice_id is not None
        and seg.voice_id != narrator_voice_id
    )


def _chapter_samples(
    chapter: RenderedChapter,
    book_dir: Path,
    pauses: PauseProfile,
    narrator_voice_id: str | None = None,
    voice_gains: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Concatenate a chapter's segment audio with pause silences. Clips of one block (a
    quote-boundary split) join with no gap and their facing edge silence trimmed (_trim_join);
    single-voice renders have one clip per block, so they are never trimmed.

    ``voice_gains`` maps voice_id -> linear gain (inter-voice loudness matching); voices not
    in it are copied through untouched, so an empty/None map is bit-identical to no matching.
    """
    parts = [_silence(pauses.chapter_lead_in)]
    prev: BlockType | None = None
    prev_block_id: str | None = None
    prev_raw: np.ndarray | None = None  # the previous clip before voice gain
    prev_dialogue = False
    pending_scene_break = False
    for seg in chapter.segments:
        if seg.type is BlockType.SCENE_BREAK:
            pending_scene_break = True
            continue
        cur_dialogue = _is_dialogue(seg, narrator_voice_id)
        # A multi-voice paragraph yields several segments sharing one block_id; only insert a
        # pause when the BLOCK changes (single-voice has one segment per block, so identical).
        block_join = prev is not None and seg.block_id != prev_block_id
        if block_join and pauses.trim_block_edges and prev_raw is not None:
            parts[-1] = _trim_tail(parts[-1], prev_raw, pauses.block_tail_keep)
        if block_join:
            if pending_scene_break:
                gap = pauses.scene_break
            elif prev is BlockType.HEADING:
                gap = pauses.after_heading
            elif prev_dialogue and cur_dialogue:
                gap = pauses.dialogue  # short beat in a character back-and-forth
            else:
                gap = pauses.paragraph
            parts.append(_silence(gap))
        pending_scene_break = False

        wav_path = book_dir / seg.wav
        if not wav_path.is_file():
            raise AssembleError(
                f"missing segment audio: chapter={chapter.index} block={seg.block_id} "
                f"expected {wav_path}; re-run `seiyuu render`"
            )
        samples, sample_rate = sf.read(str(wav_path), dtype="float32")
        if sample_rate != CANONICAL_SAMPLE_RATE or samples.ndim != 1:
            raise AssembleError(
                f"non-canonical audio reached assembly: {wav_path} "
                f"(rate={sample_rate}, ndim={samples.ndim}, block={seg.block_id})"
            )
        if prev_raw is not None and seg.block_id == prev_block_id:
            # a quote-boundary split inside one block: no gap here, so trim the clip edges
            parts[-1], samples = _trim_join(parts[-1], prev_raw, samples, pauses)
        elif block_join and pauses.trim_block_edges:
            samples = _trim_head(samples, pauses.block_head_keep)
        prev_raw = samples
        factor = (voice_gains or {}).get(seg.voice_id) if seg.voice_id else None
        if factor is not None:
            samples = samples * np.float32(factor)
        parts.append(samples)
        prev = seg.type
        prev_block_id = seg.block_id
        prev_dialogue = cur_dialogue
    parts.append(_silence(pauses.chapter_lead_out))
    return np.concatenate(parts)


def _loudnorm_filter(target: LoudnessTarget, measured: dict | None = None) -> str:
    """The ffmpeg `loudnorm` filter string. Pass 1 (measured=None) just measures; pass 2 feeds
    the measured values back with linear=true for a non-pumping, single correction."""
    base = f"loudnorm=I={target.i}:TP={target.tp}:LRA={target.lra}"
    if measured is None:
        return base
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def _measure_loudness(wav_path: Path, target: LoudnessTarget) -> dict:
    """Pass 1: run loudnorm in analysis mode and parse the JSON it prints to stderr."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(wav_path),
        "-af", f"{_loudnorm_filter(target)}:print_format=json",
        "-f", "null", "-",
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_ffmpeg_timeout(wav_path)
        )
    except subprocess.TimeoutExpired as exc:
        raise AssembleError(f"loudness measurement timed out for {wav_path.name}") from exc
    start, end = proc.stderr.rfind("{"), proc.stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise AssembleError(
            f"loudness measurement failed for {wav_path.name}: {proc.stderr.strip()[-400:]}"
        )
    return json.loads(proc.stderr[start : end + 1])


def _ffmpeg_timeout(*inputs: Path) -> float:
    """Hang guard, not a performance bound: a generous floor plus an ultra-conservative
    per-byte rate, so a legitimate encode can never trip it while a wedged ffmpeg fails
    the job loudly instead of pinning it in 'running' until a server restart. Cancel is
    only checked between chapters, so this is the ONLY exit from a stuck subprocess."""
    total = 0
    for path in inputs:
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return 600.0 + total / 200_000


def _encode_mp3(
    wav_path: Path, mp3_path: Path, title: str, track: int, album: str, af: str | None = None
) -> None:
    # Encode to a .part sibling and replace on success: a failed/killed encode must never
    # leave a truncated file at the final name (the book registry infers `assembled` from
    # chapters/*.mp3 existence). The .part suffix hides the extension, so -f is explicit.
    tmp = mp3_path.with_name(mp3_path.name + ".part")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path)]
    if af:
        cmd += ["-af", af]
    cmd += [
        "-codec:a", "libmp3lame", "-q:a", "4",
        "-metadata", f"title={title}",
        "-metadata", f"track={track}",
        "-metadata", f"album={album}",
        "-f", "mp3",
        str(tmp),
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_ffmpeg_timeout(wav_path)
        )
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise AssembleError(f"ffmpeg timed out encoding {mp3_path.name}") from exc
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise AssembleError(f"ffmpeg failed for {mp3_path.name}: {proc.stderr.strip()}")
    os.replace(tmp, mp3_path)


def _voice_gains(
    manifest: RenderManifest,
    book_output_dir: Path,
    voice_match: VoiceMatch | None,
    say: Callable[[str], None],
    check: Callable[[], None],
) -> dict[str, float]:
    """The per-BOOK voice gain map (computed once per assemble/master call, before the
    chapter loop, so every chapter of both outputs gets the same gains)."""
    if voice_match is None:
        return {}
    plan = plan_voice_gains(
        manifest, book_output_dir, voice_match, check_cancel=check, progress=say
    )
    if plan is None:
        return {}
    say(plan.summary())
    return plan.factors()


def assemble_book(
    book_output_dir: Path,
    *,
    pauses: PauseProfile | None = None,
    loudness: LoudnessTarget | None = None,
    voice_match: VoiceMatch | None = None,
    progress: Callable[[str], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> AssembleResult:
    """Build chapters/ch{NNN}.mp3 for every chapter in the render manifest.

    When `voice_match` is given, each voice first gets one book-wide gain toward the
    narrator's loudness (None = no matching). When `loudness` is given, each chapter is then
    two-pass loudnorm'd to that target before encoding (None = leave levels untouched).
    ``check_cancel`` (when given) is called
    between chapters and may raise to abort cooperatively; finished chapter MP3s remain.
    """
    if shutil.which("ffmpeg") is None:
        raise AssembleError("ffmpeg not found on PATH; install it and retry")
    book_output_dir = Path(book_output_dir)
    manifest_path = book_output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise AssembleError(f"no render manifest at {manifest_path}; run `seiyuu render` first")
    manifest = RenderManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    pauses = pauses or PauseProfile()
    say = progress or (lambda _msg: None)
    check = check_cancel or (lambda: None)
    chapters_dir = book_output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    album = manifest.book_title or manifest.book_id
    # narrator id (multi-voice only) lets the pause logic tell dialogue from narration
    narrator_voice_id = (manifest.assignment or {}).get("narrator_voice_id")

    gains = _voice_gains(manifest, book_output_dir, voice_match, say, check)

    mp3_paths: list[Path] = []
    total_seconds = 0.0
    for chapter in manifest.chapters:
        check()
        samples = _chapter_samples(chapter, book_output_dir, pauses, narrator_voice_id, gains)
        tmp_wav = chapters_dir / f"ch{chapter.index:03d}.tmp.wav"
        mp3_path = chapters_dir / f"ch{chapter.index:03d}.mp3"
        try:
            sf.write(str(tmp_wav), samples, CANONICAL_SAMPLE_RATE, subtype="PCM_16")
            af = None
            if loudness:
                af = _loudnorm_filter(loudness, _measure_loudness(tmp_wav, loudness))
            _encode_mp3(tmp_wav, mp3_path, chapter.title, chapter.index, album, af=af)
        finally:
            tmp_wav.unlink(missing_ok=True)
        seconds = len(samples) / CANONICAL_SAMPLE_RATE
        total_seconds += seconds
        mp3_paths.append(mp3_path)
        say(f"chapter {chapter.index}: {mp3_path.name} ({seconds / 60:.1f} min)")
    return AssembleResult(mp3_paths=mp3_paths, total_seconds=total_seconds)


def _escape_meta(value: str) -> str:
    """Escape the ffmetadata special characters (= ; # \\ and newline)."""
    return re.sub(r"([=;#\\\n])", r"\\\1", value)


def _ffmetadata(title: str | None, chapters: list[tuple[str, int, int]]) -> str:
    """An ffmpeg metadata file: a global title plus one [CHAPTER] block per chapter (ms)."""
    lines = [";FFMETADATA1"]
    if title:
        lines.append(f"title={_escape_meta(title)}")
    for chap_title, start_ms, end_ms in chapters:
        lines += [
            "",
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={_escape_meta(chap_title)}",
        ]
    return "\n".join(lines) + "\n"


def _encode_m4b(
    book_wav: Path,
    ffmeta: Path,
    m4b_path: Path,
    *,
    af: str | None,
    cover: Path | None,
    bitrate: str,
    sample_rate: int,
) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    cmd += ["-i", str(book_wav), "-i", str(ffmeta)]
    if cover:
        cmd += ["-i", str(cover)]
    cmd += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
    if cover:
        cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    if af:
        cmd += ["-af", af]
    # Encode to a .part sibling and replace on success: the registry infers `mastered` from
    # {book_id}.m4b existence, and -y would otherwise truncate a previous good m4b before a
    # re-master that then fails. -f ipod: the MP4/iTunes audiobook muxer (chapters + cover).
    tmp = m4b_path.with_name(m4b_path.name + ".part")
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-ar", str(sample_rate), "-f", "ipod", str(tmp)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_ffmpeg_timeout(book_wav)
        )
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise AssembleError(f"ffmpeg timed out encoding {m4b_path.name}") from exc
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise AssembleError(f"ffmpeg failed for {m4b_path.name}: {proc.stderr.strip()}")
    os.replace(tmp, m4b_path)


def master_book(
    book_output_dir: Path,
    *,
    pauses: PauseProfile | None = None,
    loudness: LoudnessTarget | None = None,
    voice_match: VoiceMatch | None = None,
    cover: Path | None = None,
    bitrate: str = "64k",
    sample_rate: int = 44_100,
    target_seconds: float | None = None,
    tempo_bounds: tuple[float, float] = (0.85, 1.3),
    progress: Callable[[str], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> MasterResult:
    """Build one chaptered .m4b (AAC) audiobook from the render manifest.

    Chapters are streamed into a single working WAV (peak memory = one chapter, so full books
    fit), with the same per-book voice gains as `assemble_book` when `voice_match` is given,
    loudness-normalized over the whole book when `loudness` is given, then encoded to AAC
    at `sample_rate` (the single 24kHz→44.1kHz upsample) with ffmpeg chapter markers and an
    optional cover image. When `target_seconds` is set, a single clamped `atempo` nudges the
    whole book toward that runtime and the chapter markers are scaled to match.

    ``check_cancel`` (when given) is called between chapters during streaming, after
    streaming, and before the encode, and may raise to abort cooperatively; the working
    WAV is removed either way. Only the final ffmpeg passes are uninterruptible, and the
    m4b lands at its final name atomically (encode to .part, replace on success).
    """
    from seiyuu.duration import tempo_for_target

    if shutil.which("ffmpeg") is None:
        raise AssembleError("ffmpeg not found on PATH; install it and retry")
    book_output_dir = Path(book_output_dir)
    manifest_path = book_output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise AssembleError(f"no render manifest at {manifest_path}; run `seiyuu render` first")
    manifest = RenderManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    pauses = pauses or PauseProfile()
    say = progress or (lambda _msg: None)
    check = check_cancel or (lambda: None)
    narrator_voice_id = (manifest.assignment or {}).get("narrator_voice_id")

    # Measure before creating the work dir: a cancel/failure while measuring leaves nothing.
    gains = _voice_gains(manifest, book_output_dir, voice_match, say, check)
    work = book_output_dir / "master"
    work.mkdir(parents=True, exist_ok=True)
    book_wav = work / "book.wav"
    ffmeta = work / "chapters.ffmeta"
    m4b_path = book_output_dir / f"{manifest.book_id}.m4b"

    raw_marks: list[tuple[str, int, int]] = []
    cursor_ms = 0
    # One try around streaming AND encoding: a cancel/failure mid-stream must still remove
    # the working WAV (it is the whole book, potentially many GB).
    try:
        with sf.SoundFile(
            str(book_wav), mode="w", samplerate=CANONICAL_SAMPLE_RATE, channels=1, subtype="PCM_16"
        ) as out:
            for chapter in manifest.chapters:
                check()
                samples = _chapter_samples(
                    chapter, book_output_dir, pauses, narrator_voice_id, gains
                )
                out.write(samples)
                dur_ms = round(len(samples) / CANONICAL_SAMPLE_RATE * 1000)
                raw_marks.append((chapter.title, cursor_ms, cursor_ms + dur_ms))
                cursor_ms += dur_ms
                say(f"chapter {chapter.index}: {chapter.title} (+{dur_ms / 1000 / 60:.1f} min)")

        check()  # last chance before the two full-book ffmpeg passes (measure + encode)
        # target-duration: one clamped atempo over the whole book; scale chapter marks to match.
        tempo = 1.0
        if target_seconds:
            lo, hi = tempo_bounds
            tempo = tempo_for_target(cursor_ms / 1000, target_seconds, lo=lo, hi=hi)
            say(f"target-duration: atempo={tempo:.4f}")
        marks = [(t, round(s / tempo), round(e / tempo)) for t, s, e in raw_marks]
        ffmeta.write_text(
            _ffmetadata(manifest.book_title or manifest.book_id, marks), encoding="utf-8"
        )
        filters: list[str] = []
        if loudness:
            say("loudness: measuring...")
            filters.append(_loudnorm_filter(loudness, _measure_loudness(book_wav, loudness)))
        if abs(tempo - 1.0) > 1e-3:
            filters.append(f"atempo={tempo:.4f}")
        check()  # the encode itself is the one uninterruptible step
        say(f"encoding {m4b_path.name}...")
        _encode_m4b(
            book_wav,
            ffmeta,
            m4b_path,
            af=",".join(filters) or None,
            cover=cover,
            bitrate=bitrate,
            sample_rate=sample_rate,
        )
    finally:
        book_wav.unlink(missing_ok=True)
        ffmeta.unlink(missing_ok=True)
        if not any(work.iterdir()):
            work.rmdir()
    return MasterResult(
        m4b_path=m4b_path,
        total_seconds=cursor_ms / 1000 / tempo,
        chapters=len(raw_marks),
        tempo=tempo,
    )
