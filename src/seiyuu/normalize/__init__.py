"""Text normalization stage (pure function, no I/O) — M3.

Turns written prose into speakable text: numbers, currency, ordinals, roman numerals,
abbreviations, unicode cleanup, and engine-aware punctuation. Deterministic and
fixture-tested. The TTS segment cache key hashes this output, so when normalization changes
the hashes change and stale cache entries are naturally invalidated (bump
``settings.normalization_version`` for debuggability — it is NOT part of the key).

Number/word expansion is identical across engines so there is one spoken-word reference for
M4 whisper validation; only punctuation handling differs per profile.
"""

from typing import TYPE_CHECKING

from seiyuu.normalize.profiles import apply_profile, profile_for

if TYPE_CHECKING:
    from seiyuu.normalize.lexicon import CompiledLexicon


def normalize_text(
    text: str, *, profile: str = "default", lexicon: "CompiledLexicon | None" = None
) -> str:
    """Normalize `text` to speakable words under an engine profile ('kokoro'|'chatterbox').

    ``lexicon`` (F3), when given, is a pre-compiled per-book pronunciation matcher applied as
    a respell pass right after unicode cleanup. It is passed IN (never read from disk here) so
    this stays a pure, deterministic, fixture-testable function.
    """
    return apply_profile(text, profile, lexicon=lexicon)


__all__ = ["normalize_text", "profile_for"]
