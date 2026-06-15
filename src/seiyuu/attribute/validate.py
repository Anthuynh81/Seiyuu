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


def normalize_ws(text: str) -> str:
    """Canonical form for comparison: NFC, whitespace runs collapsed to one space, trimmed."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def reconstructs_block(block_text: str, segment_texts: Sequence[str]) -> bool:
    """True if the segment texts, joined in order, reproduce ``block_text`` (whitespace-modulo)."""
    return normalize_ws(" ".join(segment_texts)) == normalize_ws(block_text)


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
