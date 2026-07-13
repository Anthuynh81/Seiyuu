"""Adversarial reconstruction-invariant suite — the main guard against local-model drift.

Each fixture is a source block plus a set of segment texts a model might return; the
honest split must pass and every tampered split (paraphrase, dropped/added sentence,
reordered dialogue, punctuation edit) must be rejected.
"""

import pytest

from seiyuu.attribute.models import Segment, SegmentType
from seiyuu.attribute.validate import (
    find_reconstruction_failures,
    normalize_ws,
    reconstructs_block,
)
from seiyuu.ingest.models import Block, BlockType

BLOCK = '"I won\'t go," she said, trembling. "Not after everything."'

# Honest narration/dialogue split of BLOCK.
HONEST = ['"I won\'t go,"', " she said, trembling. ", '"Not after everything."']


def test_honest_split_reconstructs():
    assert reconstructs_block(BLOCK, HONEST)


@pytest.mark.parametrize(
    "segments",
    [
        pytest.param(
            ['"I will not go,"', " she said, trembling. ", '"Not after everything."'],
            id="paraphrase",
        ),
        pytest.param(
            ['"I won\'t go,"', " she said. ", '"Not after everything."'], id="dropped-word"
        ),
        pytest.param(
            ['"Not after everything."', " she said, trembling. ", '"I won\'t go,"'],
            id="reordered-dialogue",
        ),
        pytest.param(['"I won\'t go,"', " she said, trembling. "], id="dropped-sentence"),
        pytest.param(
            ['"I won\'t go,"', " she said, trembling, sobbing. ", '"Not after everything."'],
            id="added-word",
        ),
    ],
)
def test_tampered_splits_rejected(segments):
    assert not reconstructs_block(BLOCK, segments)


def test_typographic_quote_folding_tolerated():
    # The model emitting straight quotes for curly source quotes is cosmetic, not a
    # paraphrase — it must NOT be rejected (real-world Qwen3 behavior on prose).
    curly = "“I won’t go,” she said, trembling. “Not after everything.”"
    straight = ['"I won\'t go,"', " she said, trembling. ", '"Not after everything."']
    assert reconstructs_block(curly, straight)


def test_word_change_still_rejected_under_quote_folding():
    # Folding quotes must not mask an actual word substitution.
    curly = "“I won’t go,” she said."
    assert not reconstructs_block(curly, ['"I will not go,"', " she said."])


def test_seam_whitespace_tolerated():
    # Boundary whitespace may move/duplicate without failing.
    assert reconstructs_block(
        BLOCK, ['"I won\'t go,"  ', "she said, trembling.", '  "Not after everything."']
    )


def test_abutting_emdash_dialogue_reconstructs():
    # An em-dash-attached quote has NO seam whitespace between prose and dialogue; the
    # collapse form would inject a space here and wrongly fail. Whitespace-insensitive
    # comparison reconstructs it exactly. (#1)
    block = '"Stop!"—but he didn\'t.'
    assert reconstructs_block(block, ['"Stop!"', "—but he didn't."])


def test_spaceless_tag_reconstructs():
    # A space-less speech tag: prose abuts the quote with no separating space.
    block = 'He said,"Hello."'
    assert reconstructs_block(block, ["He said,", '"Hello."'])


def test_dropped_seam_space_reconstructs():
    # The pre-existing dropped-whitespace-seam case: '"A""B"' splits into '"A"' + '"B"'
    # with no space between; must reconstruct.
    assert reconstructs_block('"A""B"', ['"A"', '"B"'])


def test_real_change_still_rejected_whitespace_insensitive():
    # Whitespace-insensitivity must ONLY reduce false-negatives: a genuine word change /
    # dropped word (whitespace aside) still fails. (#1)
    assert not reconstructs_block('He said,"Hello."', ["He said,", '"Goodbye."'])
    assert not reconstructs_block('"Stop!"—but he didn\'t.', ['"Stop!"', "—but he did."])


# UK single-quote convention: the honest single-curly split must pass and tampered
# variants must still be rejected (the quote fold collapses ‘/’ to ', never words).
UK_BLOCK = "‘I won’t go,’ she said, trembling. ‘Not after everything.’"
UK_HONEST = ["‘I won’t go,’", " she said, trembling. ", "‘Not after everything.’"]


def test_single_curly_honest_split_reconstructs():
    assert reconstructs_block(UK_BLOCK, UK_HONEST)


@pytest.mark.parametrize(
    "segments",
    [
        pytest.param(
            ["‘I will not go,’", " she said, trembling. ", "‘Not after everything.’"],
            id="uk-paraphrase",
        ),
        pytest.param(["‘I won’t go,’", " she said, trembling. "], id="uk-dropped-sentence"),
        pytest.param(
            ["‘Not after everything.’", " she said, trembling. ", "‘I won’t go,’"],
            id="uk-reordered-dialogue",
        ),
    ],
)
def test_single_curly_tampered_splits_rejected(segments):
    assert not reconstructs_block(UK_BLOCK, segments)


def test_nfc_canonical_equivalence_tolerated():
    # Composed vs decomposed accent is the same text.
    assert reconstructs_block("café", ["café"])


def test_normalize_ws_collapses_and_trims():
    assert normalize_ws("  a\t b\n\n c  ") == "a b c"


def _seg(block_id: str, text: str, type_=SegmentType.NARRATION, speaker=None) -> Segment:
    return Segment(block_id=block_id, type=type_, text=text, speaker=speaker)


def test_find_failures_flags_missing_and_mismatch():
    blocks = [
        Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="Hello there."),
        Block(id="ch001_b0002", type=BlockType.PARAGRAPH, text="Goodbye now."),
    ]
    segments = [_seg("ch001_b0001", "Hello there.")]  # b0002 has nothing
    failures = find_reconstruction_failures(blocks, segments)
    assert [f.block_id for f in failures] == ["ch001_b0002"]
    assert "no segments" in failures[0].reason


def test_find_failures_clean_when_all_reconstruct():
    blocks = [Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="Hello there.")]
    segments = [_seg("ch001_b0001", "Hello "), _seg("ch001_b0001", "there.")]
    assert find_reconstruction_failures(blocks, segments) == []
