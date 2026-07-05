"""Validation stage: transcribe rendered segments (faster-whisper, CPU) and fuzzy-match
against normalized text. Mandatory for LLM-style TTS (Chatterbox/Fish) before assembly.
"""

from seiyuu.validate.models import SegmentWords, ValidationResult, WordTiming
from seiyuu.validate.validator import ValidationError, Validator, match_ratio

__all__ = [
    "SegmentWords",
    "ValidationError",
    "ValidationResult",
    "Validator",
    "WordTiming",
    "match_ratio",
]
