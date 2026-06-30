"""Validation stage: transcribe rendered segments (faster-whisper, CPU) and fuzzy-match
against normalized text. Mandatory for LLM-style TTS (Chatterbox/Fish) before assembly.
"""

from seiyuu.validate.models import ValidationResult
from seiyuu.validate.validator import ValidationError, Validator, match_ratio

__all__ = [
    "ValidationError",
    "ValidationResult",
    "Validator",
    "match_ratio",
]
