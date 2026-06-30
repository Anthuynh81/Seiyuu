"""Ordered, deterministic normalization transforms + per-engine profiles.

Number/word expansion is shared across engines (one spoken-word reference for M4 whisper
validation); profiles differ ONLY in punctuation handling, since Chatterbox's own punc_norm
already folds em-dashes/ellipses/curly quotes while Kokoro does not.
"""

import re
import unicodedata

from seiyuu.normalize.numbers import (
    expand_currency,
    expand_decades,
    expand_numbers,
    expand_ordinals,
    expand_percent,
    expand_roman_numerals,
)

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
_WHITESPACE = re.compile(r"[ \t ]+")
_NEWLINES = re.compile(r"\s*\n\s*")

# Abbreviation/honorific expansion. Order matters (longer/qualified patterns first).
_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bMrs\.", re.I), "Missus"),
    (re.compile(r"\bMr\.", re.I), "Mister"),
    (re.compile(r"\bMs\.", re.I), "Miss"),
    (re.compile(r"\bDr\.", re.I), "Doctor"),
    (re.compile(r"\bProf\.", re.I), "Professor"),
    (re.compile(r"\bMt\.", re.I), "Mount"),
    (re.compile(r"\bRev\.", re.I), "Reverend"),
    (re.compile(r"\bCapt\.", re.I), "Captain"),
    (re.compile(r"\bGen\.", re.I), "General"),
    (re.compile(r"\bSgt\.", re.I), "Sergeant"),
    (re.compile(r"\bLt\.", re.I), "Lieutenant"),
    (re.compile(r"\be\.g\.", re.I), "for example"),
    (re.compile(r"\bi\.e\.", re.I), "that is"),
    (re.compile(r"\betc\.", re.I), "et cetera"),
    (re.compile(r"\bvs\.?", re.I), "versus"),
    (re.compile(r"\bNo\.\s*(?=\d)", re.I), "Number "),
    (re.compile(r"&"), " and "),
]

# "St." is Street when it follows a capitalized word (Baker St.), else Saint (St. Paul).
_STREET_RE = re.compile(r"\b([A-Z][a-zA-Z]+)\s+St\.")
_SAINT_RE = re.compile(r"\bSt\.")


def _unicode_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = "".join(c for c in text if c == "\n" or unicodedata.category(c)[0] != "C")
    return text


def _abbreviations(text: str) -> str:
    text = _STREET_RE.sub(r"\1 Street", text)
    text = _SAINT_RE.sub("Saint", text)
    for pattern, repl in _ABBREVIATIONS:
        text = pattern.sub(repl, text)
    return text


def _punctuation(text: str, *, fold_dashes: bool) -> str:
    if fold_dashes:
        # A pause the engine will honor; Chatterbox does this itself, so its profile skips it.
        text = re.sub(r"\s*[—–]\s*", ", ", text)  # em/en dash
        text = re.sub(r"\s*\.{3,}\s*|\s*…\s*", ", ", text)  # ellipsis
    return text


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", _NEWLINES.sub(" ", text)).strip()


# profile -> options
_PROFILES = {
    "default": {"fold_dashes": True},
    "kokoro": {"fold_dashes": True},
    "chatterbox": {"fold_dashes": False},  # the engine's punc_norm folds these itself
}


def profile_for(engine_id: str | None) -> str:
    return engine_id if engine_id in _PROFILES else "default"


def apply_profile(text: str, profile: str) -> str:
    opts = _PROFILES.get(profile, _PROFILES["default"])
    text = _unicode_clean(text)
    text = _abbreviations(text)
    text = expand_roman_numerals(text)
    text = expand_percent(text)
    text = expand_currency(text)
    text = expand_decades(text)
    text = expand_ordinals(text)
    text = expand_numbers(text)
    text = _punctuation(text, fold_dashes=opts["fold_dashes"])
    return _collapse_whitespace(text)
