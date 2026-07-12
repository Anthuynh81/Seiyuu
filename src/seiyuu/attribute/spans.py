"""Deterministic span splitting — the foundation of reconstruction-by-construction.

The LLM is unreliable at reproducing prose verbatim (it adds/drops quotes, paraphrases).
So instead of trusting it to echo segment text, we split each block into spans HERE and the
model only labels each span (narration/dialogue/thought + speaker). Segments are then built
from the source span text, so concatenating a block's segments always reproduces the block
exactly — the reconstruction invariant cannot be violated by the model.

Splitting is on double-quote boundaries by default: a quoted run (dialogue) vs the prose
around it. Straight and curly double quotes are recognised; single quotes/apostrophes are
left alone (they are apostrophes far more often than dialogue in English prose). A block
with no double quotes is a single span. UK-convention books (all dialogue in ‘single curly
quotes’) opt into :data:`DialogueConvention.SINGLE_CURLY`, detected once per book by
:func:`detect_dialogue_convention` and threaded down by the attribution pipeline; the
single-curly pattern is guarded so apostrophes (don’t, o’clock, the Bennets’) never open or
close a run. ``"".join(split_block_spans(text)) == text`` always holds in every mode.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

# A quoted run: an opening double quote, any non-double-quote chars, a closing double quote.
# Matching straight (") and curly (“ ”) forms; the inner class excludes all three so a run
# never swallows a second quoted span.
_QUOTED_RUN = re.compile(r'[“"][^“”"]*[”"]')


_OPEN_QUOTES = '"“”'


class DialogueConvention(StrEnum):
    """Which quote glyphs mark dialogue in a book (detected once per book).

    Only SINGLE_CURLY ever switches the splitter: U+2018/U+2019 pairs can be guarded against
    apostrophes. SINGLE_STRAIGHT and UNKNOWN are warn-only verdicts — straight singles are
    glyph-identical to apostrophes, so splitting on them would mis-slice real prose.
    """

    DOUBLE = "double"
    SINGLE_CURLY = "single_curly"
    SINGLE_STRAIGHT = "single_straight"
    UNKNOWN = "unknown"


# A single-curly (UK) dialogue run, guarded because U+2019 doubles as the apostrophe:
# - opener: U+2018 preceded by nothing, whitespace, or opening punctuation (U+2018 is
#   near-unambiguous — apostrophes render as U+2019 — but the guard keeps a stray mid-word
#   glyph from opening a run);
# - closer: U+2019 preceded by punctuation — never a word character, so possessives like
#   "the Bennets’ house" stay inside the quote — and followed by end/whitespace/closing
#   punctuation, never intra-word (don’t, o’clock, it’s).
# Content is lazy and excludes a second opener, so an unclosed quote degrades to prose
# exactly like the double-quote pattern; nested doubles (UK style: ‘… “inner” …’) stay part
# of the run. A punctuation-less closer (‘Yes’ said Tom) is deliberately missed — precision
# over recall, the span stays narration rather than risking an apostrophe mis-close.
_SINGLE_CURLY_RUN = re.compile(
    r"(?<![^\s\(\[\{\-–—“‘\"«])"
    r"‘"
    r"[^‘]*?"
    r"(?<=[.,;:!?…—–\-”\"])’"
    r"(?=[\s\)\]\}\-–—.,;:!?…”\"»]|$)"
)

# The straight-single analogue, used ONLY for detection counting (never for splitting):
# same boundary guards, content confined to one line because a straight quote carries no
# open/close distinction at all.
_SINGLE_STRAIGHT_RUN = re.compile(
    r"(?<![^\s\(\[\{\-–—“\"«])"
    r"'"
    r"[^\n]*?"
    r"(?<=[.,;:!?…—–\-”\"])'"
    r"(?=[\s\)\]\}\-–—.,;:!?…”\"»]|$)"
)

# Conservative thresholds (precision over recall): a book stays DOUBLE unless single-quote
# runs clearly dominate — more than the ratio times the double count AND above an absolute
# floor — so a handful of decorative singles can never flip a real double-quote book.
_SINGLE_DOMINANCE_RATIO = 10
_SINGLE_FLOOR = 20


@dataclass(frozen=True)
class ConventionDetection:
    """Book-level dialogue-convention verdict plus the run counts that justify it."""

    convention: DialogueConvention
    double_runs: int
    single_curly_runs: int
    single_straight_runs: int


def detect_dialogue_convention(text: str) -> ConventionDetection:
    """Classify a book's dialogue convention from its normalized text (pure, deterministic).

    DOUBLE is the default and wins any ambiguity, including a book with no dialogue at all.
    SINGLE_CURLY / SINGLE_STRAIGHT require guarded single runs to dominate doubles by
    ``_SINGLE_DOMINANCE_RATIO`` and clear ``_SINGLE_FLOOR``. UNKNOWN marks a book with zero
    double runs and a single-quote mix too scattered to classify either way.
    """
    doubles = sum(1 for _ in _QUOTED_RUN.finditer(text))
    curly = sum(1 for _ in _SINGLE_CURLY_RUN.finditer(text))
    straight = sum(1 for _ in _SINGLE_STRAIGHT_RUN.finditer(text))
    if curly >= _SINGLE_FLOOR and curly > _SINGLE_DOMINANCE_RATIO * doubles:
        verdict = DialogueConvention.SINGLE_CURLY
    elif straight >= _SINGLE_FLOOR and straight > _SINGLE_DOMINANCE_RATIO * doubles:
        verdict = DialogueConvention.SINGLE_STRAIGHT
    elif doubles == 0 and curly + straight >= _SINGLE_FLOOR:
        verdict = DialogueConvention.UNKNOWN
    else:
        verdict = DialogueConvention.DOUBLE
    return ConventionDetection(verdict, doubles, curly, straight)


def _run_pattern(convention: DialogueConvention) -> re.Pattern[str]:
    """The quoted-run pattern for a convention. Only SINGLE_CURLY leaves the double pattern;
    SINGLE_STRAIGHT/UNKNOWN books are too apostrophe-ambiguous to split (warn-only)."""
    if convention is DialogueConvention.SINGLE_CURLY:
        return _SINGLE_CURLY_RUN
    return _QUOTED_RUN


def split_block_spans(
    text: str, convention: DialogueConvention = DialogueConvention.DOUBLE
) -> list[str]:
    """Split a block into alternating prose / quoted spans. Concatenation reproduces text."""
    spans: list[str] = []
    cursor = 0
    for match in _run_pattern(convention).finditer(text):
        if match.start() > cursor:
            spans.append(text[cursor : match.start()])
        spans.append(match.group())
        cursor = match.end()
    if cursor < len(text):
        spans.append(text[cursor:])
    return spans or [text]


def is_quoted_span(span: str) -> bool:
    """True if a span is a quoted run (dialogue) — it opens with a double-quote glyph."""
    return bool(span) and span[0] in _OPEN_QUOTES


@dataclass(frozen=True)
class Span:
    """One span of a block, carrying its source offset and role.

    ``text`` is a verbatim ``block_text[start:start+len(text)]`` slice — never rewritten — so
    concatenating a block's spans reproduces the block exactly. ``candidate_id`` is set only
    on a substantial italic PROSE run that is a THOUGHT candidate the LLM must confirm.
    """

    text: str
    start: int  # code-point offset into the block text
    quoted: bool
    candidate_id: str | None = None


def quoted_ordinals(spans: list[Span]) -> list[tuple[int, Span]]:
    """``(ordinal, span)`` for each QUOTED span, where ``ordinal`` is its 0-based position
    AMONG THE QUOTED SPANS ONLY (not the interleaved prose+quote list).

    This is the single shared definition of a quote's index (F1). BOTH the prompt indexer
    (which injects ``⟦Q{ordinal}⟧`` markers) and ``_assemble_segments`` (which looks the
    label up) must key on THIS ordinal, or a multi-quote block mis-aligns. Keeping it here,
    over the same ``Span`` list both sides already build, makes drift impossible.
    """
    out: list[tuple[int, Span]] = []
    ordinal = 0
    for span in spans:
        if span.quoted:
            out.append((ordinal, span))
            ordinal += 1
    return out


# A thought candidate must be a multi-word italic run — single italic words, titles, and
# foreign phrases are emphasis, not interior monologue, and are excluded here (the primary
# emphasis-as-thought guard alongside the prose-only rule and the LLM confirm step).
_MIN_THOUGHT_WORDS = 3
# "whole/near-whole sentence": ends at a sentence terminator (after dropping a trailing
# closing quote/apostrophe) OR the run covers nearly the entire prose span.
_SENTENCE_ENDERS = (".", "!", "?", "…")
_TRAILING_QUOTES = "\"”’'"
_NEAR_WHOLE_RATIO = 0.9


def is_italic_thought_candidate(run_text: str, prose_text: str) -> bool:
    """True if an italic run inside ``prose_text`` is a substantial thought candidate (D5).

    Deterministic and conservative: ``>=3`` words AND (ends a sentence OR is nearly the whole
    prose span). It only NOMINATES; the LLM still has to confirm genuine interior monologue,
    so an over-nominated emphasis run degrades to narration rather than becoming a thought.
    """
    stripped = run_text.strip()
    if len(stripped.split()) < _MIN_THOUGHT_WORDS:
        return False
    core = stripped.rstrip(_TRAILING_QUOTES)
    ends_sentence = core.endswith(_SENTENCE_ENDERS)
    prose_stripped = prose_text.strip()
    near_whole = bool(prose_stripped) and len(stripped) >= _NEAR_WHOLE_RATIO * len(prose_stripped)
    return ends_sentence or near_whole


def _quote_regions(
    text: str, convention: DialogueConvention = DialogueConvention.DOUBLE
) -> list[tuple[int, int, bool]]:
    """The block's prose/quoted regions as ``(start, end, quoted)`` — the offset-aware form
    of :func:`split_block_spans` (same partition, same boundaries)."""
    regions: list[tuple[int, int, bool]] = []
    cursor = 0
    for match in _run_pattern(convention).finditer(text):
        if match.start() > cursor:
            regions.append((cursor, match.start(), False))
        regions.append((match.start(), match.end(), True))
        cursor = match.end()
    if cursor < len(text):
        regions.append((cursor, len(text), False))
    return regions or [(0, len(text), False)]


def thought_candidate_spans(
    block_id: str,
    text: str,
    italic_spans: list[tuple[int, int]],
    convention: DialogueConvention = DialogueConvention.DOUBLE,
) -> list[Span]:
    """Quote-split, then sub-split PROSE spans at SUBSTANTIAL italic runs into thought candidates.

    Quoted spans are NEVER sub-split, so italic emphasis inside dialogue stays part of the
    dialogue span — making the cardinal emphasis-as-thought failure mechanically impossible.
    An italic run that straddles a quote boundary (not wholly within one prose region) is
    ignored. Passing ``italic_spans=[]`` reproduces :func:`split_block_spans` exactly, so the
    thought-off path is byte-identical. ``"".join(s.text for s in result) == text`` always
    holds — sub-splitting only subdivides an existing prose slice.
    """
    spans: list[Span] = []
    for rstart, rend, quoted in _quote_regions(text, convention):
        if quoted:
            spans.append(Span(text[rstart:rend], rstart, True))
            continue
        region = text[rstart:rend]
        candidates = [
            (istart, iend)
            for istart, iend in italic_spans
            if rstart <= istart < iend <= rend
            and is_italic_thought_candidate(text[istart:iend], region)
        ]
        if not candidates:
            spans.append(Span(region, rstart, False))
            continue
        cursor = rstart
        for istart, iend in candidates:  # italic_spans are sorted + non-overlapping
            if istart > cursor:
                spans.append(Span(text[cursor:istart], cursor, False))
            spans.append(
                Span(text[istart:iend], istart, False, candidate_id=f"{block_id}:{istart}")
            )
            cursor = iend
        if cursor < rend:
            spans.append(Span(text[cursor:rend], cursor, False))
    return spans
