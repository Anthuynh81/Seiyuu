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
