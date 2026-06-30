"""Assembly stage: manifest + cached segments → per-chapter MP3s.

Pause logic: scene_break → long, after-heading → extra, dialogue↔dialogue → short beat,
otherwise the paragraph gap. Optional EBU R128 loudness normalization (two-pass loudnorm)
brings each chapter to a target LUFS. The chaptered .m4b is built by the `master` stage from
these same normalized chapters. ffmpeg (on PATH) is the only way audio containers are touched.
"""

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

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
    """

    paragraph: float = 0.6
    after_heading: float = 1.2
    scene_break: float = 1.8
    dialogue: float = 0.35
    chapter_lead_in: float = 0.5
    chapter_lead_out: float = 1.0


@dataclass(frozen=True)
class LoudnessTarget:
    """EBU R128 loudnorm targets. -18 LUFS integrated suits audiobooks (Audible guidance is
    -18 to -20); TP is the true-peak ceiling (dBTP), LRA the allowed loudness range."""

    i: float = -18.0
    tp: float = -1.5
    lra: float = 11.0


class AssembleError(Exception):
    """Loud assembly failure naming the chapter/block involved."""


@dataclass
class AssembleResult:
    mp3_paths: list[Path]
    total_seconds: float


@dataclass
class MasterResult:
    m4b_path: Path
    total_seconds: float
    chapters: int


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * CANONICAL_SAMPLE_RATE), dtype=np.float32)


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
) -> np.ndarray:
    """Concatenate a chapter's segment audio with pause silences."""
    parts = [_silence(pauses.chapter_lead_in)]
    prev: BlockType | None = None
    prev_block_id: str | None = None
    prev_dialogue = False
    pending_scene_break = False
    for seg in chapter.segments:
        if seg.type is BlockType.SCENE_BREAK:
            pending_scene_break = True
            continue
        cur_dialogue = _is_dialogue(seg, narrator_voice_id)
        # A multi-voice paragraph yields several segments sharing one block_id; only insert a
        # pause when the BLOCK changes (single-voice has one segment per block, so identical).
        if prev is not None and seg.block_id != prev_block_id:
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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    start, end = proc.stderr.rfind("{"), proc.stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise AssembleError(
            f"loudness measurement failed for {wav_path.name}: {proc.stderr.strip()[-400:]}"
        )
    return json.loads(proc.stderr[start : end + 1])


def _encode_mp3(
    wav_path: Path, mp3_path: Path, title: str, track: int, album: str, af: str | None = None
) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path)]
    if af:
        cmd += ["-af", af]
    cmd += [
        "-codec:a", "libmp3lame", "-q:a", "4",
        "-metadata", f"title={title}",
        "-metadata", f"track={track}",
        "-metadata", f"album={album}",
        str(mp3_path),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssembleError(f"ffmpeg failed for {mp3_path.name}: {proc.stderr.strip()}")


def assemble_book(
    book_output_dir: Path,
    *,
    pauses: PauseProfile | None = None,
    loudness: LoudnessTarget | None = None,
    progress: Callable[[str], None] | None = None,
) -> AssembleResult:
    """Build chapters/ch{NNN}.mp3 for every chapter in the render manifest.

    When `loudness` is given, each chapter is two-pass loudnorm'd to that target before
    encoding (None = leave levels untouched).
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
    chapters_dir = book_output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    album = manifest.book_title or manifest.book_id
    # narrator id (multi-voice only) lets the pause logic tell dialogue from narration
    narrator_voice_id = (manifest.assignment or {}).get("narrator_voice_id")

    mp3_paths: list[Path] = []
    total_seconds = 0.0
    for chapter in manifest.chapters:
        samples = _chapter_samples(chapter, book_output_dir, pauses, narrator_voice_id)
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
    # -f ipod: the MP4/iTunes audiobook muxer (chapters + cover); .m4b isn't auto-detected.
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-ar", str(sample_rate), "-f", "ipod", str(m4b_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssembleError(f"ffmpeg failed for {m4b_path.name}: {proc.stderr.strip()}")


def master_book(
    book_output_dir: Path,
    *,
    pauses: PauseProfile | None = None,
    loudness: LoudnessTarget | None = None,
    cover: Path | None = None,
    bitrate: str = "64k",
    sample_rate: int = 44_100,
    progress: Callable[[str], None] | None = None,
) -> MasterResult:
    """Build one chaptered .m4b (AAC) audiobook from the render manifest.

    Chapters are streamed into a single working WAV (peak memory = one chapter, so full books
    fit), loudness-normalized over the whole book when `loudness` is given, then encoded to AAC
    at `sample_rate` (the single 24kHz→44.1kHz upsample) with ffmpeg chapter markers and an
    optional cover image.
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
    narrator_voice_id = (manifest.assignment or {}).get("narrator_voice_id")

    work = book_output_dir / "master"
    work.mkdir(parents=True, exist_ok=True)
    book_wav = work / "book.wav"
    ffmeta = work / "chapters.ffmeta"
    m4b_path = book_output_dir / f"{manifest.book_id}.m4b"

    chapter_marks: list[tuple[str, int, int]] = []
    cursor_ms = 0
    with sf.SoundFile(
        str(book_wav), mode="w", samplerate=CANONICAL_SAMPLE_RATE, channels=1, subtype="PCM_16"
    ) as out:
        for chapter in manifest.chapters:
            samples = _chapter_samples(chapter, book_output_dir, pauses, narrator_voice_id)
            out.write(samples)
            dur_ms = round(len(samples) / CANONICAL_SAMPLE_RATE * 1000)
            chapter_marks.append((chapter.title, cursor_ms, cursor_ms + dur_ms))
            cursor_ms += dur_ms
            say(f"chapter {chapter.index}: {chapter.title} (+{dur_ms / 1000 / 60:.1f} min)")

    ffmeta.write_text(
        _ffmetadata(manifest.book_title or manifest.book_id, chapter_marks), encoding="utf-8"
    )
    try:
        af = None
        if loudness:
            say("loudness: measuring...")
            af = _loudnorm_filter(loudness, _measure_loudness(book_wav, loudness))
        say(f"encoding {m4b_path.name}...")
        _encode_m4b(
            book_wav, ffmeta, m4b_path, af=af, cover=cover, bitrate=bitrate, sample_rate=sample_rate
        )
    finally:
        book_wav.unlink(missing_ok=True)
        ffmeta.unlink(missing_ok=True)
        if not any(work.iterdir()):
            work.rmdir()
    return MasterResult(
        m4b_path=m4b_path, total_seconds=cursor_ms / 1000, chapters=len(chapter_marks)
    )
