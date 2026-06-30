"""Number, currency, ordinal and roman-numeral → spoken-word helpers.

Pure and deterministic (no I/O). Everything resolves to plain words so there is ONE
unambiguous spoken form that both TTS engines and (M4) whisper validation can agree on.
"""

import re

from num2words import num2words

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# A strict roman-numeral grammar so we never mistake an ordinary word for a numeral.
_ROMAN_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.IGNORECASE)


def roman_to_int(token: str) -> int | None:
    """Strict roman→int; returns None if `token` is not a well-formed numeral."""
    if not token or not _ROMAN_RE.match(token):
        return None
    total, prev = 0, 0
    for ch in reversed(token.upper()):
        value = _ROMAN_VALUES[ch]
        total += -value if value < prev else value
        prev = max(prev, value)
    return total


def cardinal(n: int) -> str:
    return num2words(n)


def ordinal_words(n: int) -> str:
    return num2words(n, to="ordinal")


def _ordinal_digits(match: re.Match) -> str:
    return ordinal_words(int(match.group(1)))


_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_SCALES = "hundred|thousand|million|billion|trillion"
# $ / £ / € with optional decimals and an optional scale word ("$5 million").
_CURRENCY_RE = re.compile(rf"([$£€])\s*(\d[\d,]*)(?:\.(\d+))?(?:\s+({_SCALES}))?", re.IGNORECASE)
# Bare numbers: not preceded by a digit (so we never re-split a number already touched).
_NUMBER_RE = re.compile(r"(?<!\d)\d[\d,]*(?:\.\d+)?")
# Decades: 1990s -> nineteen nineties; '90s/40s -> nineties/forties.
_DECADE4_RE = re.compile(r"(?<!\d)([12]\d{3})s\b")
_DECADE2_RE = re.compile(r"(?<!\d)'?(\d0)s\b")

_CURRENCY_UNITS = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}


def expand_ordinals(text: str) -> str:
    return _ORDINAL_RE.sub(_ordinal_digits, text)


def expand_percent(text: str) -> str:
    return _PERCENT_RE.sub(r"\1 percent", text)


def _say_digits_after_point(frac: str) -> str:
    return " ".join(cardinal(int(d)) for d in frac)


def _say_amount(whole: str, frac: str | None) -> str:
    if frac:
        return f"{cardinal(int(whole))} point {_say_digits_after_point(frac)}"
    return cardinal(int(whole))


def _pluralize(word: str) -> str:
    return word[:-1] + "ies" if word.endswith("y") else word + "s"


def _say_currency(match: re.Match) -> str:
    symbol, whole, frac, scale = (
        match.group(1),
        match.group(2).replace(",", ""),
        match.group(3),
        match.group(4),
    )
    major, minor = _CURRENCY_UNITS[symbol]
    if scale:  # "$5 million" -> "five million dollars"
        return f"{_say_amount(whole, frac)} {scale.lower()} {_pluralize(major)}"
    if frac and len(frac) == 2 and int(frac):  # "$5.50" -> dollars and cents
        n, c = int(whole), int(frac)
        cents = "pence" if minor == "pence" else (minor if c == 1 else _pluralize(minor))
        return f"{cardinal(n)} {major if n == 1 else _pluralize(major)} and {cardinal(c)} {cents}"
    if frac:  # other decimal, no scale: "$5.5" -> "five point five dollars"
        return f"{_say_amount(whole, frac)} {_pluralize(major)}"
    n = int(whole)
    return f"{cardinal(n)} {major if n == 1 else _pluralize(major)}"


def expand_currency(text: str) -> str:
    return _CURRENCY_RE.sub(_say_currency, text)


def expand_decades(text: str) -> str:
    text = _DECADE4_RE.sub(lambda m: _pluralize(num2words(int(m.group(1)), to="year")), text)
    return _DECADE2_RE.sub(lambda m: _pluralize(cardinal(int(m.group(1)))), text)


def _say_number(match: re.Match) -> str:
    raw = match.group(0).replace(",", "")
    whole, _, frac = raw.partition(".")
    return _say_amount(whole, frac or None)


def expand_numbers(text: str) -> str:
    return _NUMBER_RE.sub(_say_number, text)


# Headings count ("Chapter IV" -> "Chapter four"); a proper name is regnal ("Henry VIII"
# -> "Henry the eighth"). Bare romans (esp. "I") are left alone — too ambiguous.
_HEADING_WORDS = r"chapter|part|book|act|scene|volume|canto|section|appendix"
_HEADING_ROMAN_RE = re.compile(rf"\b({_HEADING_WORDS})\s+([IVXLCDM]+)\b", re.IGNORECASE)
_REGNAL_ROMAN_RE = re.compile(r"\b([A-Z][a-z]+)\s+([IVXLCDM]+)\b")


def expand_roman_numerals(text: str) -> str:
    def heading(m: re.Match) -> str:
        # Require a CAPITALIZED heading word: real headings are "Part IV"/"CHAPTER II", while
        # lowercase "the part I played" is the pronoun "I" and must be left untouched.
        n = roman_to_int(m.group(2))
        return f"{m.group(1)} {cardinal(n)}" if n and m.group(1)[:1].isupper() else m.group(0)

    def regnal(m: re.Match) -> str:
        # Require >=2 letters: a lone "I"/"V"/"X" after a name is usually a pronoun/initial,
        # not a regnal numeral. Heading context (handled above) still converts single letters.
        n = roman_to_int(m.group(2))
        if n and len(m.group(2)) >= 2:
            return f"{m.group(1)} the {ordinal_words(n)}"
        return m.group(0)

    return _REGNAL_ROMAN_RE.sub(regnal, _HEADING_ROMAN_RE.sub(heading, text))
