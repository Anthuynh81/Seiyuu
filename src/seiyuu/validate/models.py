"""Validation payloads: the documented result of transcribing a rendered segment.

`ValidationResult` is the validator's output for one segment; the render loop folds it (plus an
attempt count) into the manifest so reviewers and the M6 UI can see which segments failed
whisper and why. Additive — non-validated engines (Kokoro) never produce one.
"""

from pydantic import BaseModel


class ValidationResult(BaseModel):
    ok: bool  # score >= the configured minimum ratio
    score: float  # 0..1 fuzzy ratio of transcript vs the normalized text
    transcript: str  # what whisper heard
    expected: str  # the normalized text the engine was asked to speak
