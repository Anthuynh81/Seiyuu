"""Text normalization: pure-function fixtures, idempotency, and the tricky fiction cases."""

import pytest

from seiyuu.normalize import normalize_text, profile_for
from seiyuu.normalize.numbers import roman_to_int


@pytest.mark.parametrize(
    "raw,expected",
    [
        # numbers / decimals / percent
        ("I have 3 cats.", "I have three cats."),
        ("It cost 1234 coins.", "It cost one thousand, two hundred and thirty-four coins."),
        ("5,000 men marched.", "five thousand men marched."),
        ("It was 3.5 miles.", "It was three point five miles."),
        ("Down 50% today.", "Down fifty percent today."),
        # ordinals
        ("the 1st time and the 22nd day", "the first time and the twenty-second day"),
        # currency
        ("It cost $5.50.", "It cost five dollars and fifty cents."),
        ("Just $1 left.", "Just one dollar left."),
        ("paid £20", "paid twenty pounds"),
        ("worth $5 million", "worth five million dollars"),
        ("a $5.5 billion deal", "a five point five billion dollars deal"),
        # zero-cents currency reads as whole dollars, never "point zero zero" (#6)
        ("It cost $5.00.", "It cost five dollars."),
        ("Just $1.00 left.", "Just one dollar left."),
        ("It cost $5.0.", "It cost five dollars."),
        # vs abbreviation expands only as a standalone token, not word-initial (#7)
        ("cats vs dogs", "cats versus dogs"),
        ("cats vs. dogs", "cats versus dogs"),
        ("the vsync test", "the vsync test"),
        ("open vscode now", "open vscode now"),
        # decades (must not produce "ninetys")
        ("the 1990s were wild", "the nineteen nineties were wild"),
        ("back in the '90s", "back in the nineties"),
        ("the 1840s", "the eighteen forties"),
        # abbreviations / honorifics
        ("Mr. and Mrs. Bennet saw Dr. Jones.", "Mister and Missus Bennet saw Doctor Jones."),
        ("Meet at St. Paul.", "Meet at Saint Paul."),
        # the abbreviation dot doubles as the sentence period at end-of-line (consumed)
        ("He lives on Baker St.", "He lives on Baker Street"),
        ("cats & dogs", "cats and dogs"),
        # roman numerals — regnal vs heading vs ambiguous bare 'I'
        ("King Henry VIII reigned.", "King Henry the eighth reigned."),
        ("Chapter IV", "Chapter four"),
        ("Henry I think you are wrong.", "Henry I think you are wrong."),  # bare I left alone
        ("Part IV begins", "Part four begins"),  # capitalized heading converts
        ("the part I played", "the part I played"),  # lowercase heading word + pronoun I: leave
    ],
)
def test_normalize_cases(raw, expected):
    assert normalize_text(raw) == expected


def test_idempotent():
    samples = [
        "Mr. Smith paid $5.50 on the 1st.",
        "King George III and Chapter IV.",
        "cats & dogs, 50% off",
    ]
    for s in samples:
        once = normalize_text(s)
        assert normalize_text(once) == once


def test_unicode_cleanup_and_whitespace():
    # zero-width + smart quotes (NFKC) + collapsed whitespace
    assert normalize_text("a​b   c\n d") == "ab c d"


def test_chatterbox_profile_leaves_dashes_kokoro_folds():
    text = "Wait—stop."
    assert normalize_text(text, profile="kokoro") == "Wait, stop."
    assert normalize_text(text, profile="chatterbox") == "Wait—stop."


def test_profile_for_falls_back_to_default():
    assert profile_for("kokoro") == "kokoro"
    assert profile_for("chatterbox") == "chatterbox"
    assert profile_for("eleven") == "default"
    assert profile_for(None) == "default"


def test_roman_parser_rejects_non_numerals():
    assert roman_to_int("VIII") == 8
    assert roman_to_int("MCMLXXX4") is None
    assert roman_to_int("hello") is None
    assert roman_to_int("") is None


def test_memoization_keys_on_lexicon_content() -> None:
    # normalize_text is memoized; the lexicon participates by CONTENT fingerprint, so a
    # recompile of the same entries hits while different entries can never collide.
    from seiyuu.normalize.lexicon import BookLexicon, LexiconEntry, compile_lexicon

    def lex(respelling: str):
        return compile_lexicon(
            BookLexicon(book_id="b", entries=[LexiconEntry(term="Xy", respelling=respelling)])
        )

    a, a2, b = lex("zed"), lex("zed"), lex("kew")
    assert a == a2 and hash(a) == hash(a2)
    assert a != b
    assert "zed" in normalize_text("Xy said.", lexicon=a)
    assert "kew" in normalize_text("Xy said.", lexicon=b)  # no cross-lexicon collision
    assert "zed" in normalize_text("Xy said.", lexicon=a2)  # equal recompile still respells
    # an empty lexicon is byte-identical to no lexicon
    empty = compile_lexicon(BookLexicon(book_id="b", entries=[]))
    assert normalize_text("Xy said.", lexicon=empty) == normalize_text("Xy said.")
