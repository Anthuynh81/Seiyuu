"""Deterministic span splitting — the guarantee that underpins reconstruction-by-design."""

import pytest

from seiyuu.attribute.spans import split_block_spans

CASES = [
    ("plain narration with no quotes", ["plain narration with no quotes"]),
    ('He said "yes" then left.', ["He said ", '"yes"', " then left."]),
    ('"Hello," she said.', ['"Hello,"', " she said."]),
    ('"A" "B"', ['"A"', " ", '"B"']),
    # Curly quotes
    ("“My dear,” said she.", ["“My dear,”", " said she."]),
    # Apostrophes are NOT split (they are not dialogue)
    ("Bennet's house wasn't far.", ["Bennet's house wasn't far."]),
    # Unclosed quote -> single span (graceful)
    ('"He never finished the', ['"He never finished the']),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_split_cases(text, expected):
    assert split_block_spans(text) == expected


@pytest.mark.parametrize("text,_", CASES)
def test_concatenation_always_reproduces_source(text, _):
    # The core invariant: joining the spans must reproduce the input byte-for-byte.
    assert "".join(split_block_spans(text)) == text


def test_empty_text_is_single_span():
    assert split_block_spans("") == [""]
