"""PDNC gold import: text-based, monotonic, conservative mapping of PDNC quotations onto the
book's own quoted spans. Tiny synthetic PDNC-shaped CSVs + a tiny synthetic book; no real
corpus files are read."""

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from seiyuu.attribute.evaluate import book_quotes, load_gold, match_strength, normalize_name
from seiyuu.attribute.pdnc import (
    PdncCharacter,
    _speaker_names,
    align_quotes,
    coverage_report,
    fold_tokens,
    import_pdnc,
    load_pdnc_characters,
    load_pdnc_quotes,
)
from seiyuu.cli import main
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook

BOOK_ID = "pdnc-book-00000000"

LONG_A = "I have been thinking about the ball at Netherfield all the morning long"
LONG_B = "You must not expect such good fortune twice in one season, my dear"
LONG_C = "Then we shall all go together to Meryton tomorrow and hear the news"


def _p(cid: int, bid: int, text: str) -> Block:
    return Block(id=f"ch{cid:03d}_b{bid:04d}", type=BlockType.PARAGRAPH, text=text)


def _book() -> NormalizedBook:
    """Curly quotes and curly apostrophes throughout (PDNC uses straight ones)."""
    return NormalizedBook(
        book_meta=BookMeta(
            book_id=BOOK_ID, title="PDNC Book", source_path="p.epub", source_sha256="0" * 64
        ),
        chapters=[
            Chapter(
                title="Preface",
                blocks=[_p(1, 1, "The critic wrote of “loving by allowance” in his essay.")],
            ),
            Chapter(
                title="Chapter I",
                blocks=[
                    Block(id="ch002_b0001", type=BlockType.HEADING, text="Chapter I"),
                    # PDNC: ONE quote, two sub-quotations around the speech tag
                    _p(2, 2, "“My dear Mr. Bennet,” said his lady, “have you heard the news?”"),
                    # PDNC: ONE sub-quotation covering BOTH of our spans
                    _p(2, 3, "“I am glad,” “very glad indeed, and I don’t mind who knows it.”"),
                    # PDNC: TWO sub-quotations inside ONE of our spans
                    _p(2, 4, "“No. I cannot say that I ever heard of such a thing in my life.”"),
                    _p(2, 5, f"“{LONG_A}.”"),
                    _p(2, 6, "“Yes.”"),  # not annotated in PDNC: must stay unlabeled
                    _p(2, 7, f"“{LONG_B}.”"),
                    _p(2, 8, "“Yes.”"),  # PDNC's "Yes." belongs HERE (after LONG_B)
                    _p(2, 9, f"“{LONG_C}.”"),
                ],
            ),
            Chapter(
                title="Chapter II",
                blocks=[
                    # one span holding two DIFFERENT speakers' quotes: never labeled
                    _p(3, 1, "“How hard it is to be believed! And how impossible in others!”"),
                    # spelling variant (mamma / mama): near-identical, matched
                    _p(3, 2, "“Indeed, mamma, you are quite mistaken about him,” cried she."),
                    # the stranger's line: speaker missing from character_info
                    _p(3, 3, "“Pray, which is the road to Longbourn?”"),
                ],
            ),
        ],
    )


# (quoteID, sub-quotes, speaker, type)
QUOTES = [
    ("Q0", ["My dear Mr. Bennet,", "have you heard the news?"], "Mrs. Bennet", "Anaphoric"),
    ("Q1", ["I am glad, very glad indeed, and I don't mind who knows it."], "Jane", "Implicit"),
    ("Q2", ["No.", "I cannot say that I ever heard of such a thing in my life."], "Mr. Bennet",
     "Implicit"),
    ("Q3", [LONG_A + "."], "Lydia", "Explicit"),
    ("Q4", [LONG_B + "."], "Mrs. Bennet", "Implicit"),
    ("Q5", ["Yes."], "Lydia", "Implicit"),
    ("Q6", ["_This_ sentence was never in our edition of the book at all."], "Jane", "Explicit"),
    ("Q7", [LONG_C + "."], "Kitty", "Implicit"),
    ("Q8", ["How hard it is to be believed!"], "Jane", "Implicit"),
    ("Q9", ["And how impossible in others!"], "Elizabeth", "Implicit"),
    ("Q10", ["Indeed, mama, you are quite mistaken about him,"], "Elizabeth", "Explicit"),
    ("Q11", ["Pray, which is the road to Longbourn?"], "Stranger", "Implicit"),
]  # fmt: skip

CHARACTERS = [
    ("0", "Mrs. Bennet", "{'Mrs. Bennet', 'Bennet'}"),
    ("1", "Mr. Bennet", "['Mr. Bennet']"),
    ("2", "Jane", "{'Jane', 'Miss Bennet'}"),
    ("3", "Elizabeth", "{'Elizabeth', 'Lizzy', 'Jane'}"),  # 'Jane' is another main name
    ("4", "Lydia", "{'Lydia'}"),
    ("5", "Kitty", "{'Kitty'}"),
]


def _write_csvs(tmp_path: Path) -> tuple[Path, Path]:
    quotes = tmp_path / "quotation_info.csv"
    with quotes.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quoteID", "quoteText", "subQuotationList", "quoteByteSpans", "speaker",
                    "addressees", "quoteType", "referringExpression"])  # fmt: skip
        for qid, subs, speaker, qtype in QUOTES:
            w.writerow([qid, " ".join(subs), repr(subs), "[[0, 1]]", speaker, "[]", qtype, ""])
    chars = tmp_path / "character_info.csv"
    with chars.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Character ID", "Main Name", "Aliases", "Gender", "Category"])
        for cid, name, aliases in CHARACTERS:
            w.writerow([cid, name, aliases, "F", "major"])
    return quotes, chars


@pytest.fixture
def imported(tmp_path):
    quotes, chars = _write_csvs(tmp_path)
    return import_pdnc(_book(), load_pdnc_quotes(quotes), load_pdnc_characters(chars))


def _labels(result) -> dict[tuple[str, int], str]:
    return {(r.block_id, r.quote_index): r.speaker for r in result.rows}


def test_fold_tokens_ignores_glyphs_case_italics_and_apostrophes():
    assert (
        fold_tokens("“I don’t _know_,”") == fold_tokens('"i dont know,"') == ["i", "dont", "know"]
    )


def test_loaders_parse_python_literals(tmp_path):
    quotes, chars = _write_csvs(tmp_path)
    q = load_pdnc_quotes(quotes)
    assert q[0].sub_quotes == ["My dear Mr. Bennet,", "have you heard the news?"]
    assert q[0].quote_type is not None and q[0].quote_type.value == "anaphoric"
    c = load_pdnc_characters(chars)
    assert c["Mrs. Bennet"].aliases == ["Bennet", "Mrs. Bennet"]  # a set literal, sorted


def test_loader_rejects_a_bad_literal(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text(
        "quoteID,subQuotationList,speaker,quoteType\nQ0,__import__('os'),Jane,Implicit\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a python literal"):
        load_pdnc_quotes(path)


def test_split_sub_quotations_each_map_to_their_span(imported):
    got = _labels(imported)
    assert got[("ch002_b0002", 0)] == got[("ch002_b0002", 1)] == "Mrs. Bennet"


def test_one_pdnc_sub_quote_spanning_two_of_our_spans(imported):
    got = _labels(imported)
    assert got[("ch002_b0003", 0)] == got[("ch002_b0003", 1)] == "Jane"


def test_two_pdnc_sub_quotes_inside_one_of_our_spans(imported):
    assert _labels(imported)[("ch002_b0004", 0)] == "Mr. Bennet"


def test_repeated_short_quote_maps_to_the_right_occurrence(imported):
    got = _labels(imported)
    assert ("ch002_b0006", 0) not in got  # the earlier "Yes." PDNC never annotated
    assert got[("ch002_b0008", 0)] == "Lydia"
    assert got[("ch002_b0009", 0)] == "Kitty"  # alignment continues after it


def test_quote_missing_from_the_book_is_reported_unmatched(imported):
    unmatched = {u.quote_id: u for u in imported.unmatched}
    assert "Q6" in unmatched and not unmatched["Q6"].partial
    assert "sentence was never" in unmatched["Q6"].text


def test_span_joining_two_speakers_is_never_guessed(imported):
    assert ("ch003_b0001", 0) not in _labels(imported)
    assert {"Q8", "Q9"} <= {u.quote_id for u in imported.unmatched}


def test_spelling_variant_matches(imported):
    assert _labels(imported)[("ch003_b0002", 0)] == "Elizabeth"


def test_unknown_speaker_is_labeled_without_aliases(imported):
    row = next(r for r in imported.rows if r.block_id == "ch003_b0003")
    assert row.speaker == "Stranger" and row.aliases == []
    assert imported.unknown_speakers == ["Stranger"]


def test_rows_carry_aliases_types_and_the_pipeline_quote_identity(imported):
    by_key = {(r.block_id, r.quote_index): r for r in imported.rows}
    row = by_key[("ch002_b0002", 0)]
    assert row.aliases == ["Mrs. Bennet"]  # bare 'Bennet' dropped: Mr. Bennet's surname
    assert row.quote_type is not None and row.quote_type.value == "anaphoric"
    eliza = by_key[("ch003_b0002", 0)]
    assert eliza.aliases == ["Elizabeth", "Lizzy"]  # 'Jane' dropped: another main name
    assert {(d.character, d.alias) for d in imported.dropped_aliases} == {
        ("Elizabeth", "Jane"),
        ("Mrs. Bennet", "Bennet"),
    }
    # every row IS one of the pipeline's own quoted spans, verbatim, at its offsets
    spans = {(q.block_id, q.quote_index): q for q in book_quotes(_book())}
    blocks = {b.id: b.text for ch in _book().chapters for b in ch.blocks}
    for r in imported.rows:
        q = spans[(r.block_id, r.quote_index)]
        assert r.text == q.span.text == blocks[r.block_id][r.char_start : r.char_end]
        assert r.chapter_index == q.chapter_index


def test_coverage_counts(imported):
    r = imported
    assert r.pdnc_quotes == len(QUOTES)
    # Q6 missing, Q8 + Q9 unlabeled (their one span joins two speakers)
    assert r.pdnc_quotes_matched == r.pdnc_quotes_fully_matched == len(QUOTES) - 3
    assert r.spans == 14 and r.spans_labeled == 11  # preface quote, first "Yes.", ch003_b0001
    per = {c.chapter_index: (c.labeled, c.spans) for c in r.chapters}
    assert per == {1: (0, 1), 2: (9, 10), 3: (2, 3)}
    text = "\n".join(coverage_report(r))
    assert "PDNC quotes matched to the book (whole book): 9/12" in text
    assert "Q6 [Jane]" in text and "Stranger" in text


def test_chapter_filter_restricts_rows_not_alignment(tmp_path):
    quotes, chars = _write_csvs(tmp_path)
    q, c = load_pdnc_quotes(quotes), load_pdnc_characters(chars)
    only3 = import_pdnc(_book(), q, c, (3,))
    assert {r.chapter_index for r in only3.rows} == {3}
    assert only3.spans == 3 and [ch.chapter_index for ch in only3.chapters] == [3]
    with pytest.raises(ValueError, match="no such chapter"):
        import_pdnc(_book(), q, c, (9,))


def test_align_quotes_needs_coverage_on_both_sides():
    """Our span being a fragment of a PDNC quote whose rest is nowhere is not enough."""
    got = align_quotes(
        [LONG_A, "Upon my word, I never heard such nonsense in all my life", LONG_B],
        ["Jane", "Lydia", "Kitty"],
        [LONG_A, "Upon my word,", LONG_B],
    )
    assert got == {0: [0], 2: [2]}


def test_cli_writes_gold_and_reports_coverage(tmp_path):
    quotes, chars = _write_csvs(tmp_path)
    books = tmp_path / "books"
    (books / BOOK_ID).mkdir(parents=True)
    (books / BOOK_ID / "normalized.json").write_text(_book().model_dump_json(), encoding="utf-8")
    out = tmp_path / "gold.jsonl"
    res = CliRunner().invoke(
        main,
        ["eval", "gold-import-pdnc", BOOK_ID, "--quotes", str(quotes), "--characters",
         str(chars), "-o", str(out), "--books-dir", str(books)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "gold: 11 quotes" in res.stdout and "wrote:" in res.stdout
    assert "PDNC quotes matched to the book (whole book): 9/12" in res.stderr
    assert "Our quoted spans labeled in all chapters: 11/14" in res.stderr
    assert "spans labeled in all chapters; 9/12 PDNC quotes matched in the whole book" in (
        res.stdout
    )
    # Mrs. Bennet's bare 'Bennet' is dropped, so Jane ('Miss Bennet') no longer collides
    assert "Dropped alias 'Bennet' of 'Mrs. Bennet'" in res.stderr
    assert "WARNING: gold speakers" not in res.stderr
    rows = load_gold(out)
    assert len(rows) == 11 and {r.book_id for r in rows} == {BOOK_ID}


# ------------------------------------------------------------------ regression guards


def test_full_expected_mapping(imported):
    """Every labeled span and its speaker, exactly: count-only checks are blind to two
    speakers being swapped (e.g. LONG_A <-> LONG_B)."""
    assert _labels(imported) == {
        ("ch002_b0002", 0): "Mrs. Bennet",
        ("ch002_b0002", 1): "Mrs. Bennet",
        ("ch002_b0003", 0): "Jane",
        ("ch002_b0003", 1): "Jane",
        ("ch002_b0004", 0): "Mr. Bennet",
        ("ch002_b0005", 0): "Lydia",
        ("ch002_b0007", 0): "Mrs. Bennet",
        ("ch002_b0008", 0): "Lydia",
        ("ch002_b0009", 0): "Kitty",
        ("ch003_b0002", 0): "Elizabeth",
        ("ch003_b0003", 0): "Stranger",
    }


def test_repeated_pdnc_quote_with_two_speakers_is_ambiguous():
    """PDNC has Lydia's AND Kitty's "Yes." where the book has one: whose is it? Unlabeled."""
    got = align_quotes(
        [LONG_A, "Yes.", "Yes.", LONG_B],
        ["Jane", "Lydia", "Kitty", "Jane"],
        [LONG_A, "Yes.", LONG_B],
    )
    assert got == {0: [0], 2: [3]}


def test_repeated_pdnc_quote_with_one_speaker_still_labels():
    got = align_quotes(
        [LONG_A, "Yes.", "Yes.", LONG_B],
        ["Jane", "Lydia", "Lydia", "Jane"],
        [LONG_A, "Yes.", LONG_B],
    )
    assert set(got) == {0, 1, 2} and got[1] in ([1], [2])


def test_repeated_book_span_is_ambiguous():
    """The book has an extra "Lizzy!" PDNC never annotated: neither copy may be labeled."""
    got = align_quotes(
        [LONG_A, "Lizzy!", LONG_B], ["Jane", "Mrs. Bennet", "Jane"],
        [LONG_A, "“Lizzy!”", "“Lizzy!”", LONG_B],
    )  # fmt: skip
    assert got == {0: [0], 3: [2]}


def test_ambiguous_repeat_is_not_relabeled_by_later_passes():
    """A multi-word duplicate must not sneak back in through the token or near-identical
    passes once its anchor is found ambiguous."""
    line = "Oh dear me, what shall we all do now"
    got = align_quotes(
        [LONG_A, line, line, LONG_B], ["Jane", "Jane", "Lydia", "Jane"], [LONG_A, line, LONG_B]
    )
    assert got == {0: [0], 2: [3]}


def test_near_identical_pass_needs_a_unique_candidate():
    got = align_quotes(
        [LONG_A, "Indeed, mama, you are quite mistaken", LONG_B],
        ["Jane", "Elizabeth", "Jane"],
        [LONG_A, "Indeed, mamma, you are quite mistaken", "Indeed, mammma, you are quite mistaken",
         LONG_B],
    )  # fmt: skip
    assert got == {0: [0], 3: [2]}


LONG_X = (
    "I am sure I never saw anybody so pleased in all my life as she was when she heard "
    "that the regiment would stay in Meryton the whole of the winter and the spring"
)


def test_span_with_a_small_second_speaker_share_is_never_labeled():
    """X (Jane, 31 words) + "Oh dear me!" (Lydia) in ONE span: Jane holds > 90% of the
    words, but the span still has two speakers."""
    got = align_quotes(
        [LONG_A, LONG_X, "Oh dear me!", LONG_B],
        ["Kitty", "Jane", "Lydia", "Kitty"],
        [LONG_A, f"“{LONG_X}, oh dear me!”", LONG_B],
    )
    assert got == {0: [0], 2: [3]}


def test_near_identical_pass_respects_one_speaker():
    """13 words of Jane + 3 of Lydia: the token pass rejects it (81%), and the character
    similarity of Jane's piece alone (> 0.9) must not label it either."""
    jane = "You must not expect such good fortune twice in one season, my love"
    got = align_quotes(
        [LONG_A, jane, "Oh dear me!", LONG_B],
        ["Kitty", "Jane", "Lydia", "Kitty"],
        [LONG_A, f"“{jane} oh dear me!”", LONG_B],
    )
    assert got == {0: [0], 2: [3]}


def test_bare_surname_alias_goes_to_the_male_titled_character():
    """PDNC lists 'Bennet' under Mrs. Bennet; a predicted {Mr. Bennet, Bennet} must then
    match Mr. Bennet only. 'Darcy' under Mr. Darcy (nobody else is 'Mr. Darcy') stays."""
    names, dropped = _speaker_names(
        {
            "Mrs. Bennet": PdncCharacter(name="Mrs. Bennet", aliases=["Bennet", "Mrs. Bennet"]),
            "Mr. Bennet": PdncCharacter(name="Mr. Bennet", aliases=["Mr. Bennet"]),
            "Mr. Darcy": PdncCharacter(name="Mr. Darcy", aliases=["Darcy", "Mr. Darcy"]),
            "Miss Darcy": PdncCharacter(name="Miss Darcy", aliases=["Miss Darcy"]),
            "Denny": PdncCharacter(name="Denny", aliases=["Denny"]),
            "Mr. Denney": PdncCharacter(name="Mr. Denney", aliases=["Mr. Denny"]),
        }
    )
    assert names["Mrs. Bennet"] == ["Mrs. Bennet"]
    assert names["Mr. Darcy"] == ["Darcy", "Mr. Darcy"]
    assert names["Denny"] == ["Denny"]  # a main name is never dropped, nor reported
    assert [(d.character, d.alias) for d in dropped] == [("Mrs. Bennet", "Bennet")]
    predicted = {normalize_name(n) for n in ("Mr. Bennet", "Bennet")}
    gold = {k: {normalize_name(n) for n in v} for k, v in names.items()}
    assert match_strength(predicted, gold["Mr. Bennet"]) > match_strength(
        predicted, gold["Mrs. Bennet"]
    )


def test_chapter_filter_scopes_the_report_and_says_what_is_whole_book(tmp_path):
    quotes, chars = _write_csvs(tmp_path)
    q, c = load_pdnc_quotes(quotes), load_pdnc_characters(chars)
    whole = {u.quote_id: u.near_chapters for u in import_pdnc(_book(), q, c).unmatched}
    assert whole == {"Q6": [2], "Q8": [2, 3], "Q9": [2, 3]}  # Q8/Q9 sit at the ch 2/3 seam
    only3 = import_pdnc(_book(), q, c, (3,))
    assert {u.quote_id for u in only3.unmatched} == {"Q8", "Q9"}  # not Q6 (chapter 2)
    text = "\n".join(coverage_report(only3))
    assert "PDNC quotes matched to the book (whole book): 9/12" in text
    assert "Our quoted spans labeled in the chosen chapters: 2/3" in text
    assert "PDNC quotes not (fully) matched near the chosen chapters: 2" in text
    assert "Q6" not in text and "Q8 [Jane] ~ch 2/3" in text
