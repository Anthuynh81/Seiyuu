"""Text normalization stage (pure function, no I/O) — M3.

Turns written prose into speakable text: numbers, currency, ordinals, roman numerals,
abbreviations, unicode cleanup, and engine-aware punctuation. Deterministic and
fixture-tested. The TTS segment cache key hashes this output, so when normalization changes
the hashes change and stale cache entries are naturally invalidated (bump
``settings.normalization_version`` for debuggability — it is NOT part of the key).

Number/word expansion is identical across engines so there is one spoken-word reference for
M4 whisper validation; only punctuation handling differs per profile.
"""

from seiyuu.normalize.profiles import apply_profile, profile_for


def normalize_text(text: str, *, profile: str = "default") -> str:
    """Normalize `text` to speakable words under an engine profile ('kokoro'|'chatterbox')."""
    return apply_profile(text, profile)


__all__ = ["normalize_text", "profile_for"]
