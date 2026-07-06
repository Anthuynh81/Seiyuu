"""Deterministic span splitting — the foundation of reconstruction-by-construction.

The LLM is unreliable at reproducing prose verbatim (it adds/drops quotes, paraphrases).
So instead of trusting it to echo segment text, we split each block into spans HERE and the
model only labels each span (narration/dialogue/thought + speaker). Segments are then built
from the source span text, so concatenating a block's segments always reproduces the block
exactly — the reconstruction invariant cannot be violated by the model.

Splitting is on double-quote boundaries: a quoted run (dialogue) vs the prose around it.
Straight and curly double quotes are recognised; single quotes/apostrophes are left alone
(they are apostrophes far more often than dialogue in English prose). A block with no
double quotes is a single span. ``"".join(split_block_spans(text)) == text`` always holds.
"""

import re
from dataclasses import dataclass

# A quoted run: an opening double quote, any non-double-quote chars, a closing double quote.
# Matching straight (") and curly (“ ”) forms; the inner class excludes all three so a run
# never swallows a second quoted span.
_QUOTED_RUN = re.compile(r'[“"][^“”"]*[”"]')


_OPEN_QUOTES = '"“”'


def split_block_spans(text: str) -> list[str]:
    """Split a block into alternating prose / quoted spans. Concatenation reproduces text."""
    spans: list[str] = []
    cursor = 0
    for match in _QUOTED_RUN.finditer(text):
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


def _quote_regions(text: str) -> list[tuple[int, int, bool]]:
    """The block's prose/quoted regions as ``(start, end, quoted)`` — the offset-aware form
    of :func:`split_block_spans` (same partition, same boundaries)."""
    regions: list[tuple[int, int, bool]] = []
    cursor = 0
    for match in _QUOTED_RUN.finditer(text):
        if match.start() > cursor:
            regions.append((cursor, match.start(), False))
        regions.append((match.start(), match.end(), True))
        cursor = match.end()
    if cursor < len(text):
        regions.append((cursor, len(text), False))
    return regions or [(0, len(text), False)]


def thought_candidate_spans(
    block_id: str, text: str, italic_spans: list[tuple[int, int]]
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
    for rstart, rend, quoted in _quote_regions(text):
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
