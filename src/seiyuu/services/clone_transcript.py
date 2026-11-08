"""The transcript a new Qwen3-TTS clone binds, so Qwen clones in its stronger ICL mode.

Qwen3-TTS clones two ways: from the clip's sound alone (x-vector), or from the clip plus its
exact transcript (ICL), which follows the speaker more closely. ICL needs the transcript bound
in the voice's meta (``settings.qwen3tts.ref_text_digest``, which rides the segment cache key)
and written beside the clip as ``reference.txt``; see ``engines/qwen3tts_engine.py``.

At clone time the transcript is the user's own when they typed one, otherwise whisper's (CPU).
Two cases fall back to x-vector instead of binding a transcript, because ICL with a wrong or
oversized prompt is worse than none:

- the clip runs longer than ``ICL_MAX_SECONDS``: the reference and its transcript share the
  generation budget with every line Qwen speaks, so a long prompt cuts long lines short;
- whisper heard nothing, or failed (and the user gave no transcript).

Shared by the API clone route and ``seiyuu voices clone``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
from seiyuu.repository import atomic_write_text
from seiyuu.validate import ValidationError

QWEN_ENGINE = "qwen3tts"
ICL_MAX_SECONDS = 30.0
TRANSCRIPT_NAME = "reference.txt"

# (clip path) -> (transcript, duration in seconds); Validator.transcribe_clip fits.
Transcriber = Callable[[Path], tuple[str, float]]


@dataclass(frozen=True)
class CloneTranscript:
    text: str | None  # None -> x-vector mode (nothing bound)
    note: str  # one plain sentence saying which mode and why


def choose_transcript(
    reference: Path, *, provided: str | None, transcribe: Transcriber
) -> CloneTranscript:
    """Decide the transcript to bind for a Qwen clone of ``reference`` (or None for x-vector)."""
    typed = " ".join((provided or "").split())
    try:
        heard, seconds = transcribe(reference)
    except ValidationError as exc:
        if typed:
            return CloneTranscript(typed, "cloned from the clip and your transcript")
        return CloneTranscript(
            None, f"couldn't transcribe the clip ({exc}), so it clones from the sound alone"
        )
    if seconds > ICL_MAX_SECONDS:
        return CloneTranscript(
            None,
            f"the clip is {seconds:.0f} s, over {ICL_MAX_SECONDS:.0f} s, so it clones from the "
            "sound alone (a shorter clip of 5-30 s clones more closely)",
        )
    if typed:
        return CloneTranscript(typed, "cloned from the clip and your transcript")
    heard = " ".join(heard.split())
    if not heard:
        return CloneTranscript(None, "no speech was heard, so it clones from the sound alone")
    return CloneTranscript(heard, "cloned from the clip and its transcript")


def bind_transcript(
    reference: Path, reference_sha256: str, text: str | None
) -> dict[str, dict[str, float]]:
    """Write (or remove) ``reference.txt`` beside ``reference`` and return the meta settings
    that bind it; ``{}`` (and no file) for x-vector mode. ``reference_sha256`` must be the hash
    of the PUBLISHED clip, the same bytes the consent attestation covers."""
    path = reference.with_name(TRANSCRIPT_NAME)
    if text is None:
        path.unlink(missing_ok=True)
        return {}
    atomic_write_text(path, text + "\n")
    return {QWEN_ENGINE: {REF_TEXT_DIGEST_KEY: reference_text_digest(reference_sha256, text)}}
