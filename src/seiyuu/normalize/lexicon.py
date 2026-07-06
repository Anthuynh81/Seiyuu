"""Per-book pronunciation lexicon (F3): term → grapheme respelling (+ optional Kokoro-only IPA).

Stored at ``books/{id}/lexicon.json`` as user INPUT (a sibling of ``normalized.json``,
purged with the book by ``delete_book_trees``). It is NOT a pipeline-stage marker, so nothing
is added to ``BookStatus``.

The entries are compiled ONCE into a :class:`CompiledLexicon` matcher and threaded INTO the
PURE :func:`seiyuu.normalize.normalize_text` as an argument, so normalization stays
deterministic, I/O-free, and fixture-testable. A grapheme respelling is spoken as written AND
survives the whisper validator's fold (lowercase + strip punctuation), so it is
validation-consistent on every engine. An optional per-entry IPA is applied ONLY on the
non-validated Kokoro profile: an IPA/phoneme string is not spoken literally, so on a validated
engine (Chatterbox/ElevenLabs) it would corrupt the whisper ``expected`` reference and break
match ratios — there the respelling is always used.

Matching mirrors the ``_ABBREVIATIONS`` discipline in ``profiles.py``: word-boundary anchored,
case-insensitive by default (per-entry ``case_sensitive`` override), and longest-term-first so
a longer term wins over a shorter one that is a prefix of it and no partial rewrite lands
inside a longer word. All terms are matched in a SINGLE regex pass so a replacement's output
is never re-scanned (no cascading rewrites).
"""

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from seiyuu.repository.atomic import atomic_write_text

LEXICON_NAME = "lexicon.json"
LEXICON_SCHEMA_VERSION = 1


class LexiconEntry(BaseModel):
    """One pronunciation override. ``respelling`` is the primary, engine-agnostic,
    whisper-safe field; ``ipa`` is optional and honored ONLY on the Kokoro profile."""

    term: str
    respelling: str
    ipa: str | None = None
    note: str | None = None
    case_sensitive: bool = False

    @field_validator("term", "respelling")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("term and respelling must be non-empty")
        return stripped

    @field_validator("ipa", "note")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class BookLexicon(BaseModel):
    """The per-book pronunciation dictionary persisted at ``books/{id}/lexicon.json``."""

    schema_version: int = LEXICON_SCHEMA_VERSION
    book_id: str
    entries: list[LexiconEntry] = Field(default_factory=list)


class CompiledLexicon:
    """An immutable, pre-compiled respell matcher — built ONCE from a :class:`BookLexicon`
    and passed into ``normalize_text`` so the pure function does no compilation or I/O per
    call. Empty lexicons compile to a no-op (``bool(compiled) is False``)."""

    __slots__ = ("_regex", "_respellings", "_ipas")

    def __init__(self, entries: Iterable[LexiconEntry]) -> None:
        # Longest term first: regex alternation is ordered (first alternative that matches at
        # a position wins), so this yields longest-match preference at every start position.
        ordered = sorted(entries, key=lambda e: len(e.term), reverse=True)
        parts: list[str] = []
        self._respellings: list[str] = []
        self._ipas: list[str | None] = []
        for i, entry in enumerate(ordered):
            inner = re.escape(entry.term)
            if not entry.case_sensitive:
                inner = f"(?i:{inner})"  # scoped flag: mixes with case-sensitive terms
            parts.append(f"(?P<g{i}>{inner})")
            self._respellings.append(entry.respelling)
            self._ipas.append(entry.ipa)
        self._regex = re.compile(rf"\b(?:{'|'.join(parts)})\b") if parts else None

    def __bool__(self) -> bool:
        return self._regex is not None

    def apply(self, text: str, *, profile: str) -> str:
        """Respell every lexicon term in ``text``. On the Kokoro profile an entry's IPA (if
        present) replaces its respelling; every other profile always uses the respelling."""
        if self._regex is None:
            return text
        use_ipa = profile == "kokoro"

        def _repl(match: re.Match) -> str:
            i = int(match.lastgroup[1:])  # 'gN' -> N
            if use_ipa and self._ipas[i] is not None:
                return self._ipas[i]
            return self._respellings[i]

        return self._regex.sub(_repl, text)


def compile_lexicon(lexicon: BookLexicon) -> CompiledLexicon:
    return CompiledLexicon(lexicon.entries)


# -- persistence (mirrors ingest.write_normalized) ----------------------------------------


def lexicon_path(book_dir: Path) -> Path:
    return Path(book_dir) / LEXICON_NAME


def load_lexicon(book_dir: Path, *, book_id: str | None = None) -> BookLexicon:
    """Load ``books/{id}/lexicon.json``; an absent file is an empty lexicon (not an error).
    A corrupt file raises loudly (pydantic ValidationError)."""
    path = lexicon_path(book_dir)
    if not path.is_file():
        return BookLexicon(book_id=book_id or Path(book_dir).name, entries=[])
    return BookLexicon.model_validate_json(path.read_text(encoding="utf-8"))


def save_lexicon(book_dir: Path, lexicon: BookLexicon) -> Path:
    return atomic_write_text(lexicon_path(book_dir), lexicon.model_dump_json(indent=2))


def load_compiled_lexicon(book_dir: Path) -> CompiledLexicon:
    """Convenience for the render/estimate entry points: load + compile in one call so the
    SAME compiled object can be threaded through every SegmentKey site (key parity)."""
    return compile_lexicon(load_lexicon(book_dir))


# -- deterministic auto-suggest -----------------------------------------------------------


class SuggestedTerm(BaseModel):
    term: str
    count: int  # occurrences capitalized mid-sentence (the proper-noun signal)
    sample: str  # a short surrounding excerpt for context


# Common capitalized words that are NOT names: sentence-openers, honorifics already handled
# by _ABBREVIATIONS, days/months. Kept small and deterministic — precision over recall.
_SUGGEST_STOPWORDS = frozenset(
    {
        "I",
        "The",
        "A",
        "An",
        "And",
        "But",
        "Or",
        "So",
        "Yet",
        "For",
        "Nor",
        "If",
        "Then",
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Mister",
        "Missus",
        "Miss",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
)

# A capitalized, mostly-alphabetic token (allows internal caps/apostrophes: McGonagall,
# O'Brien) of length >= 3.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’-]{2,}")
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*\s+$|[\n]\s*$")


def suggest_terms(
    texts: Iterable[str],
    *,
    existing_terms: Iterable[str] = (),
    min_count: int = 2,
    limit: int = 50,
) -> list[SuggestedTerm]:
    """Deterministic candidate surfacer: capitalized proper-noun-shaped tokens that appear
    MID-sentence (not just as a sentence opener) at least ``min_count`` times across the book
    — the classic 'hard name' signal. Free, no I/O, no paid path; the user fills the
    respelling. NOT auto-applied.
    """
    already = {t.strip().casefold() for t in existing_terms}
    mid_counts: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for text in texts:
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            if not token[0].isupper():
                continue
            preceding = text[: match.start()]
            sentence_start = not preceding.strip() or bool(_SENTENCE_END.search(preceding))
            if sentence_start:
                continue  # a mid-sentence capital is the proper-noun signal
            if token in _SUGGEST_STOPWORDS or token.casefold() in already:
                continue
            mid_counts[token] += 1
            if token not in samples:
                lo = max(0, match.start() - 20)
                hi = min(len(text), match.end() + 20)
                samples[token] = text[lo:hi].strip()
    candidates = [
        SuggestedTerm(term=term, count=count, sample=samples[term])
        for term, count in mid_counts.items()
        if count >= min_count
    ]
    # Deterministic order: most frequent first, then alphabetical.
    candidates.sort(key=lambda s: (-s.count, s.term))
    return candidates[:limit]
