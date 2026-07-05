"""Validation payloads: the documented result of transcribing a rendered segment.

`ValidationResult` is the validator's output for one segment; the render loop folds it (plus an
attempt count) into the manifest so reviewers and the M6 UI can see which segments failed
whisper and why. Additive — non-validated engines (Kokoro) never produce one.

`WordTiming`/`SegmentWords` (F2) are the forced-alignment payload: per-word (start, end, word)
spans from a whisper `word_timestamps` pass, cached per rendered segment in a
`{key_hash}.words.json` sidecar and served to the read-along UI. Additive — they never touch
`ValidationResult` or the frozen SegmentKey/cache-key format.
"""

from pydantic import BaseModel


class ValidationResult(BaseModel):
    ok: bool  # score >= the configured minimum ratio
    score: float  # 0..1 fuzzy ratio of transcript vs the normalized text
    transcript: str  # what whisper heard
    expected: str  # the normalized text the engine was asked to speak


class WordTiming(BaseModel):
    start: float  # seconds into the segment audio where the word begins
    end: float  # seconds where it ends
    word: str  # the spoken token as whisper heard it (leading space kept as whisper emits it)


class SegmentWords(BaseModel):
    words: list[WordTiming]  # in spoken order; empty when whisper heard nothing
    audio_duration: float  # length of the aligned wav, so the client can clamp the last word
    source: str = "whisper"  # provenance tag; only whisper today, but recorded for future sources
