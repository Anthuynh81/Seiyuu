"""Chunk a chapter's blocks for the LLM, with non-overlapping block ownership.

SPEC stage 2: chunks overlap (~2–4k tokens) for context, but **each block is owned by
exactly one chunk**. Neighbouring blocks are included as read-only context so the model
can resolve speakers across chunk seams; segments the model produces for non-owned
context blocks are discarded by the pipeline (the "overlap merge" policy). The owned
windows partition the blocks, so coverage is exact and duplicate-free by construction.
"""

import hashlib
import json
from dataclasses import dataclass

from seiyuu.ingest.models import Block

# Tokenless heuristic: ~4 characters per token is close enough for budgeting and keeps
# attribution free of a tokenizer dependency. Budgeting never needs to be exact.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Chunk:
    index: int
    blocks: list[Block]  # owned blocks plus leading/trailing context blocks, in order
    owned_ids: frozenset[str]  # blocks this chunk is responsible for (no overlap)

    @property
    def owned_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.id in self.owned_ids]

    @property
    def content_hash(self) -> str:
        """Hash of the owned blocks (id + text), in order — the cache's ``chunk_hash``."""
        payload = json.dumps([(b.id, b.text) for b in self.owned_blocks], separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_blocks(
    blocks: list[Block],
    *,
    budget_tokens: int = 3000,
    overlap_blocks: int = 2,
) -> list[Chunk]:
    """Split ``blocks`` into context-overlapping chunks with exclusive ownership.

    A single block larger than ``budget_tokens`` still becomes its own owned chunk.
    """
    if overlap_blocks < 0:
        raise ValueError(f"overlap_blocks must be >= 0, got {overlap_blocks}")
    if budget_tokens < 1:
        raise ValueError(f"budget_tokens must be >= 1, got {budget_tokens}")

    n = len(blocks)
    chunks: list[Chunk] = []
    i = 0
    while i < n:
        tokens = 0
        j = i
        while j < n:
            cost = estimate_tokens(blocks[j].text)
            if j > i and tokens + cost > budget_tokens:
                break
            tokens += cost
            j += 1
        start = max(0, i - overlap_blocks)
        end = min(n, j + overlap_blocks)
        chunks.append(
            Chunk(
                index=len(chunks),
                blocks=blocks[start:end],
                owned_ids=frozenset(b.id for b in blocks[i:j]),
            )
        )
        i = j
    return chunks
