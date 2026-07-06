"""The reconstruction invariant — attribution's hard guard against the LLM rewriting prose.

Pure functions, no I/O (fixture-tested with adversarial cases). SPEC stage 2: per block,
concatenating its segment texts in order must reproduce the source block text exactly,
*modulo whitespace at segment seams*. This catches paraphrase, dropped/added sentences,
reordered dialogue, and punctuation edits (e.g. straight-for-curly quotes) — the failure
modes small local models fall into. Only whitespace differences and Unicode canonical
form (NFC) are tolerated; everything else rejects the chunk.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from seiyuu.attribute.models import Segment
from seiyuu.ingest.models import Block

_WHITESPACE = re.compile(r"\s+")

# Fold typographic quote/apostrophe glyphs to ASCII for comparison only. LLMs routinely
# emit straight quotes for curly source quotes (observed with Qwen3 on real prose); that
# is a cosmetic punctuation change, not a paraphrase, so rejecting it would discard
# otherwise-perfect attribution. Folding is applied to BOTH sides, so any real word- or
# punctuation-level change still fails the check — only the glyph variants below collapse.
# Dashes and ellipses are deliberately NOT folded (they can carry meaning); revisit if a
# real chapter shows a safe, recurring need.
_QUOTE_FOLD = str.maketrans(
    {
        "“": '"',  # " left double
        "”": '"',  # " right double
        "„": '"',  # „ low double
        "‟": '"',  # ‟ high-reversed double
        "‘": "'",  # ' left single
        "’": "'",  # ' right single (also typographic apostrophe)
        "‚": "'",  # ‚ low single
        "‛": "'",  # ‛ high-reversed single
    }
)


def _fold(text: str) -> str:
    """NFC + typographic-quote fold — shared by the readable compare form and the strict one."""
    return unicodedata.normalize("NFC", text).translate(_QUOTE_FOLD)


def normalize_ws(text: str) -> str:
    """Readable compare form: NFC, typographic quotes folded, whitespace collapsed, trimmed.

    Kept for the ReconstructionFailure diagnostics (expected/got) — a single-spaced,
    trimmed rendering a human can diff. The reconstruction VERDICT uses ``_cmp`` below.
    """
    return _WHITESPACE.sub(" ", _fold(text)).strip()


def _cmp(text: str) -> str:
    """Whitespace-INSENSITIVE compare form: reconstruction tolerates ANY whitespace at
    segment seams. A quote abutting a non-space char (em-dash-attached dialogue, a
    space-less ``said,"Hello."`` tag) injects a seam space the collapse form cannot remove,
    which would wrongly reject a split that reproduced the text exactly. Removing ALL
    whitespace only ever reduces false-NEGATIVES: a real word/punctuation change (paraphrase,
    dropped/added word, reordered dialogue) still differs after stripping, because non-space
    characters are untouched.
    """
    return _WHITESPACE.sub("", _fold(text))


def reconstructs_block(block_text: str, segment_texts: Sequence[str]) -> bool:
    """True if the segments, joined in order, reproduce ``block_text`` (whitespace-insensitive)."""
    return _cmp("".join(segment_texts)) == _cmp(block_text)


@dataclass(frozen=True)
class ReconstructionFailure:
    block_id: str
    reason: str
    expected: str = ""  # normalized source text
    got: str = ""  # normalized reconstruction


def find_reconstruction_failures(
    owned_blocks: Sequence[Block], segments: Sequence[Segment]
) -> list[ReconstructionFailure]:
    """Check every owned block reconstructs from its segments (segments kept in order).

    ``segments`` must already be filtered to this chunk's owned blocks; segments for any
    other block_id are ignored here (the pipeline drops non-owned segments first).
    """
    by_block: dict[str, list[str]] = {}
    for seg in segments:
        by_block.setdefault(seg.block_id, []).append(seg.text)

    failures: list[ReconstructionFailure] = []
    for block in owned_blocks:
        texts = by_block.get(block.id)
        if not texts:
            failures.append(ReconstructionFailure(block.id, "no segments produced for block"))
            continue
        if not reconstructs_block(block.text, texts):
            failures.append(
                ReconstructionFailure(
                    block.id,
                    "segment texts do not reconstruct the source block",
                    expected=normalize_ws(block.text),
                    got=normalize_ws(" ".join(texts)),
                )
            )
    return failures
