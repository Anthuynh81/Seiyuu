"""Chunking: exclusive block ownership, context overlap, stable hashes."""

from seiyuu.attribute.chunking import chunk_blocks, estimate_tokens
from seiyuu.ingest.models import Block, BlockType


def _blocks(n: int, words_each: int = 50) -> list[Block]:
    return [
        Block(id=f"ch001_b{i:04d}", type=BlockType.PARAGRAPH, text=" ".join(["word"] * words_each))
        for i in range(1, n + 1)
    ]


def test_owned_ids_partition_all_blocks_exactly_once():
    blocks = _blocks(20)
    budget = estimate_tokens(blocks[0].text) * 3  # ~3 blocks per chunk
    chunks = chunk_blocks(blocks, budget_tokens=budget, overlap_blocks=2)

    owned: list[str] = []
    for c in chunks:
        owned.extend(b.id for b in c.owned_blocks)
    assert owned == [b.id for b in blocks]  # in order, no gaps, no dupes
    assert len(set(owned)) == len(blocks)


def test_chunks_carry_context_beyond_owned():
    blocks = _blocks(20)
    budget = estimate_tokens(blocks[0].text) * 3
    chunks = chunk_blocks(blocks, budget_tokens=budget, overlap_blocks=2)
    # A middle chunk sees more blocks (context) than it owns.
    middle = chunks[1]
    assert len(middle.blocks) > len(middle.owned_blocks)
    assert middle.owned_ids.issubset({b.id for b in middle.blocks})


def test_oversized_block_owns_itself():
    blocks = _blocks(3, words_each=5000)
    chunks = chunk_blocks(blocks, budget_tokens=10, overlap_blocks=0)
    assert len(chunks) == 3
    assert all(len(c.owned_ids) == 1 for c in chunks)


def test_content_hash_depends_only_on_owned_blocks():
    blocks = _blocks(6)
    a = chunk_blocks(blocks, budget_tokens=estimate_tokens(blocks[0].text) * 2, overlap_blocks=0)
    b = chunk_blocks(blocks, budget_tokens=estimate_tokens(blocks[0].text) * 2, overlap_blocks=2)
    # Same ownership windows despite different overlap → identical chunk hashes.
    assert [c.content_hash for c in a] == [c.content_hash for c in b]
