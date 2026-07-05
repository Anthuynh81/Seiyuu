"""Lazy forced alignment (F2): compute-or-load a segment's word timings from its cached wav.

The counterpart to the Chatterbox/Fish render-loop piggyback. Engines that don't transcribe
during render (Kokoro, ElevenLabs) have no words at manifest-write time, so the read-along
endpoint aligns them on first request against the on-disk wav — and only then. This path is
deliberately CPU-only and lives OFF the GPU resource manager, OFF the HeavyWorkGate, and OFF the
JobRunner: it must never become a second resident GPU model or queue behind a multi-hour render.

faster-whisper / CTranslate2 is not safe under concurrent ``transcribe``, so every alignment
serializes on one process-shared lock supplied by the caller.
"""

import threading
from pathlib import Path

import soundfile as sf
from pydantic import ValidationError

from seiyuu.render.cache import words_sidecar_for_wav
from seiyuu.repository import atomic_write_text
from seiyuu.validate import SegmentWords, Validator


def ensure_words(wav_path: Path, validator: Validator, lock: threading.Lock) -> SegmentWords:
    """Return the wav's cached `{key_hash}.words.json`, computing + caching it on the first miss.

    Cheap hit path is lock-free (a bare file read). On a miss the lock is taken and existence is
    re-checked before transcribing, so a burst of concurrent Listen requests for the same clip
    computes once. The sidecar is written crash-atomically (temp + fsync + replace), so a killed
    process never leaves a truncated file that would poison the clip. A re-render mints a new
    key_hash (new wav, new sidecar name), so this never returns stale timings.
    """
    wav_path = Path(wav_path)
    sidecar = words_sidecar_for_wav(wav_path)
    cached = _read(sidecar)
    if cached is not None:
        return cached
    with lock:
        cached = _read(sidecar)  # another thread may have computed it while we waited
        if cached is not None:
            return cached
        words = validator.transcribe_words(wav_path)
        segment_words = SegmentWords(
            words=words, audio_duration=float(sf.info(str(wav_path)).duration)
        )
        atomic_write_text(sidecar, segment_words.model_dump_json(indent=2))
        return segment_words


def _read(sidecar: Path) -> SegmentWords | None:
    if not sidecar.is_file():
        return None
    try:
        return SegmentWords.model_validate_json(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        # A torn/corrupt sidecar poisons nothing: treat it as a miss so ensure_words recomputes
        # and atomically overwrites it (atomic writes make this unreachable in practice).
        return None
