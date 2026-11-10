"""The transcript a new clone binds, and which engine a new clone gets by default.

Two local engines clone from a clip AND its exact transcript:

- Qwen3-TTS has two modes: from the clip's sound alone (x-vector), or from the clip plus its
  transcript (ICL), which follows the speaker more closely.
- Breeze TTS 2 has only the transcript mode. A Breeze voice without one cannot speak at all.

The transcript is bound in the voice's meta (``settings.<engine>.ref_text_digest``, which rides
the segment cache key) and written beside the clip as ``reference.txt``. The digest is the same
value under either engine (a hash of the clip and the text), so a voice keeps one binding
whichever engine clones it.

At clone time the transcript is the user's own when they typed one, otherwise whisper's (CPU).
Two cases bind nothing, because a wrong or oversized prompt is worse than none:

- the clip runs longer than ``ICL_MAX_SECONDS``: the reference and its transcript share the
  model's window with every line it speaks, so a long prompt cuts long lines short;
- whisper heard nothing, or failed (and the user gave no transcript).

For Qwen that means x-vector mode. For Breeze it means the clone is REFUSED
(:class:`CloneRefused`) rather than saved as a voice that fails on its first line.

Shared by the API clone route, ``seiyuu voice clone``, the design "keep" and the engine switch.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiyuu.engines.breeze_engine import breeze_ready
from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
from seiyuu.repository import atomic_write_text
from seiyuu.validate import ValidationError

QWEN_ENGINE = "qwen3tts"
BREEZE_ENGINE = "breeze"
# Engines that clone from the clip plus its transcript, and the ones with no other mode.
TRANSCRIPT_ENGINES = frozenset({QWEN_ENGINE, BREEZE_ENGINE})
TRANSCRIPT_REQUIRED = frozenset({BREEZE_ENGINE})
ICL_MAX_SECONDS = 30.0
TRANSCRIPT_NAME = "reference.txt"

# (clip path) -> (transcript, duration in seconds); Validator.transcribe_clip fits.
Transcriber = Callable[[Path], tuple[str, float]]


class CloneRefused(ValueError):
    """This engine cannot clone this clip (it needs a transcript and there is none to bind)."""


@dataclass(frozen=True)
class CloneTranscript:
    text: str | None  # None -> x-vector mode (nothing bound); never None for Breeze
    note: str  # one plain sentence saying which mode and why


def default_clone_engine(cfg: Any) -> str:
    """The engine a new clone, and a kept design take, gets when none is named: Breeze where
    it is set up on this machine (worker python, checkout and weights present), else Qwen3-TTS.
    A machine without the Breeze environment therefore behaves exactly as before."""
    return BREEZE_ENGINE if breeze_ready(cfg) else QWEN_ENGINE


def _clip_seconds(reference: Path) -> float | None:
    """The clip's duration from its header (no model); None when it is not readable audio."""
    import soundfile as sf

    try:
        info = sf.info(str(reference))
    except (RuntimeError, OSError):  # libsndfile: not audio, or not there
        return None
    return info.frames / float(info.samplerate) if info.samplerate else None


def _without(engine: str, reason: str, fallback: str) -> CloneTranscript:
    """No transcript to bind: x-vector for Qwen (``fallback`` says so), a refusal for Breeze."""
    if engine in TRANSCRIPT_REQUIRED:
        raise CloneRefused(
            f"{reason}, and Breeze clones only from a clip together with its exact words. Use a "
            f"clip of 5-{ICL_MAX_SECONDS:.0f} s of clear speech (typing its transcript helps), "
            f"or clone on {QWEN_ENGINE}"
        )
    return CloneTranscript(None, f"{reason}, {fallback}")


def choose_transcript(
    reference: Path,
    *,
    provided: str | None,
    transcribe: Transcriber,
    engine: str = QWEN_ENGINE,
) -> CloneTranscript:
    """Decide the transcript to bind for a clone of ``reference`` on ``engine``.

    Qwen: the text, or None for x-vector mode. Breeze: the text, or :class:`CloneRefused`."""
    typed = " ".join((provided or "").split())
    alone = "so it clones from the sound alone"
    try:
        heard, seconds = transcribe(reference)
    except ValidationError as exc:
        if typed and engine in TRANSCRIPT_REQUIRED:
            # Whisper also measures the clip. Breeze has no fallback mode, so the length rule
            # is checked another way before a typed transcript is bound to it.
            seconds = _clip_seconds(reference)
            if seconds is None:
                return _without(engine, f"the clip could not be read as audio ({exc})", alone)
            if seconds > ICL_MAX_SECONDS:
                return _without(
                    engine, f"the clip is {seconds:.0f} s, over {ICL_MAX_SECONDS:.0f} s", alone
                )
        if typed:
            return CloneTranscript(typed, "cloned from the clip and your transcript")
        return _without(engine, f"couldn't transcribe the clip ({exc})", alone)
    if seconds > ICL_MAX_SECONDS:
        return _without(
            engine,
            f"the clip is {seconds:.0f} s, over {ICL_MAX_SECONDS:.0f} s",
            f"{alone} (a shorter clip of 5-30 s clones more closely)",
        )
    if typed:
        return CloneTranscript(typed, "cloned from the clip and your transcript")
    heard = " ".join(heard.split())
    if not heard:
        return _without(engine, "no speech was heard", alone)
    return CloneTranscript(heard, "cloned from the clip and its transcript")


def bind_transcript(
    reference: Path, reference_sha256: str, text: str | None, *, engine: str = QWEN_ENGINE
) -> dict[str, dict[str, float]]:
    """Write (or remove) ``reference.txt`` beside ``reference`` and return the meta settings
    that bind it under ``engine``; ``{}`` (and no file) when there is no text.
    ``reference_sha256`` must be the hash of the PUBLISHED clip, the same bytes the consent
    attestation covers."""
    path = reference.with_name(TRANSCRIPT_NAME)
    if text is None:
        path.unlink(missing_ok=True)
        return {}
    atomic_write_text(path, text + "\n")
    return {engine: {REF_TEXT_DIGEST_KEY: reference_text_digest(reference_sha256, text)}}
