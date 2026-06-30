"""Assembly stage (M1 minimal): manifest + cached segments → per-chapter MP3s.

Pause logic is the M1 subset (scene_break → long, paragraph → medium, extra
room after headings). Dialogue-aware pauses, loudness normalization, and the
chaptered .m4b arrive in M4. ffmpeg (on PATH) is the only way audio containers
are touched.
"""

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


class AssembleError(Exception):
    """Loud assembly failure naming the chapter/block involved."""


@dataclass
class AssembleResult:
    mp3_paths: list[Path]
    total_seconds: float


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


def _encode_mp3(wav_path: Path, mp3_path: Path, title: str, track: int, album: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav_path),
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
    progress: Callable[[str], None] | None = None,
) -> AssembleResult:
    """Build chapters/ch{NNN}.mp3 for every chapter in the render manifest."""
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
            _encode_mp3(tmp_wav, mp3_path, chapter.title, chapter.index, album)
        finally:
            tmp_wav.unlink(missing_ok=True)
        seconds = len(samples) / CANONICAL_SAMPLE_RATE
        total_seconds += seconds
        mp3_paths.append(mp3_path)
        say(f"chapter {chapter.index}: {mp3_path.name} ({seconds / 60:.1f} min)")
    return AssembleResult(mp3_paths=mp3_paths, total_seconds=total_seconds)
