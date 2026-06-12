"""Text normalization stage (pure function, no I/O).

M1 ships an explicit identity normalizer: the real engine-aware normalization
(numbers, abbreviations, roman numerals, ...) lands in M3. The TTS segment
cache key hashes THIS function's output, so the key format is already final —
when M3 changes normalization, hashes change and stale cache entries are
naturally invalidated.
"""


def normalize_text(text: str) -> str:
    return text
