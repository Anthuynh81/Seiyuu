"""Deterministic gender back-fill (attribute/genders.py): titles, name nouns, pronoun evidence.

Precision over recall: a gender the model gave is never overwritten, and thin or mixed evidence
leaves the gender empty (casting's warning then tells the user) rather than guessing.
"""

import json

import pytest
from click.testing import CliRunner

from factories import make_book
from fake_provider import FakeProvider
from seiyuu.attribute.genders import count_pronoun_votes, infer_genders
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterMention,
    CharacterRegistry,
    ChunkAttribution,
    Segment,
    SegmentType,
)
from seiyuu.cli import main
from seiyuu.services import ServiceError, fill_genders


def _chapter(*narration: str, dialogue: tuple[str, str] | None = None) -> AttributedChapter:
    segments = [
        Segment(block_id=f"ch001_b{i + 1:04d}", type=SegmentType.NARRATION, text=text)
        for i, text in enumerate(narration)
    ]
    if dialogue:
        speaker, text = dialogue
        segments.append(
            Segment(block_id="ch001_b0999", type=SegmentType.DIALOGUE, speaker=speaker, text=text)
        )
    return AttributedChapter(index=1, title="One", segments=segments)


def _registry(*chars: Character) -> CharacterRegistry:
    return CharacterRegistry(characters=list(chars))


def _char(cid: str, name: str, gender: str | None = None, aliases=()) -> Character:
    return Character(id=cid, canonical_name=name, gender=gender, aliases=list(aliases))


THREE_HE = (
    "Kwak Jaekang nodded. He looked tired.",
    "Kwak Jaekang sat down; he sighed.",
    "Kwak Jaekang left and he did not look back.",
)


# -- pronoun evidence --------------------------------------------------------------------------


def test_three_agreeing_pronouns_fill_an_empty_gender():
    reg = _registry(_char("kwak", "Kwak Jaekang"))
    notes = infer_genders(reg, [_chapter(*THREE_HE)])
    assert reg.get("kwak").gender == "male"
    assert notes == ["gender inferred for 'Kwak Jaekang': male (3 of 3 pronouns after the name)"]


def test_too_few_votes_leave_the_gender_unknown():
    reg = _registry(_char("kwak", "Kwak Jaekang"))
    assert infer_genders(reg, [_chapter(*THREE_HE[:2])]) == []
    assert reg.get("kwak").gender is None


def test_mixed_evidence_below_the_majority_stays_unknown():
    reg = _registry(_char("lee", "Lee Jaheon"))
    text = ["Lee Jaheon spoke. He paused."] * 4 + ["Lee Jaheon turned. She smiled."] * 2
    assert infer_genders(reg, [_chapter(*text)]) == []  # 4 of 6 = 67% < 80%
    assert reg.get("lee").gender is None


def test_a_given_gender_is_never_overwritten():
    reg = _registry(_char("kwak", "Kwak Jaekang", gender="female"))
    assert infer_genders(reg, [_chapter(*THREE_HE)]) == []
    assert reg.get("kwak").gender == "female"


def test_a_sentence_naming_two_characters_casts_no_vote():
    reg = _registry(_char("kwak", "Kwak Jaekang"), _char("go", "Go Yeongeun"))
    votes = count_pronoun_votes(reg, [_chapter("Kwak Jaekang looked at Go Yeongeun. She smiled.")])
    assert sum(votes["kwak"].values()) == 0 and sum(votes["go"].values()) == 0


def test_the_next_sentence_counts_only_when_it_names_no_one():
    reg = _registry(_char("kwak", "Kwak Jaekang"), _char("go", "Go Yeongeun"))
    votes = count_pronoun_votes(
        reg, [_chapter("Kwak Jaekang waited. Go Yeongeun arrived and she sat.")]
    )
    assert sum(votes["kwak"].values()) == 0  # "she" belongs to the sentence naming Go
    assert votes["go"]["female"] == 1


def test_only_pronouns_after_the_name_count():
    reg = _registry(_char("kwak", "Kwak Jaekang"))
    votes = count_pronoun_votes(reg, [_chapter("She handed it to Kwak Jaekang.")])
    assert sum(votes["kwak"].values()) == 0


def test_dialogue_is_not_evidence():
    reg = _registry(_char("kwak", "Kwak Jaekang"))
    chapter = _chapter(dialogue=("kwak", "Kwak Jaekang? She said he was gone."))
    assert sum(count_pronoun_votes(reg, [chapter])["kwak"].values()) == 0


def test_a_role_name_the_book_uses_as_a_common_word_is_not_matched():
    reg = _registry(_char("host", "Host"))
    text = ["The host waved. He grinned.", "Host of the night, he bowed."] * 3
    assert "host" not in count_pronoun_votes(reg, [_chapter(*text)])
    assert reg.get("host").gender is None


def test_aliases_count_toward_the_same_character():
    reg = _registry(_char("kim", "Kim Soleum", aliases=["Soleum"]))
    text = ["Soleum froze. He swallowed.", "Kim Soleum ran. He fell.", "Soleum got up; he left."]
    infer_genders(reg, [_chapter(*text)])
    assert reg.get("kim").gender == "male"


def test_the_narrator_record_is_left_alone():
    reg = _registry(_char("narrator", "Narrator"))
    assert infer_genders(reg, [_chapter("Narrator. He said.") for _ in range(3)]) == []
    assert reg.get("narrator").gender is None


# -- titles and name nouns ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "gender"),
    [("Mr. Bennet", "male"), ("Mrs. Bennet", "female"), ("Miss Lucas", "female")],
)
def test_a_gendered_title_fills_the_gender(name, gender):
    reg = _registry(_char("x", name))
    notes = infer_genders(reg, [_chapter("Nothing here.")])
    assert reg.get("x").gender == gender
    assert "from the title" in notes[0]


def test_a_child_title_is_not_a_gender():
    reg = _registry(_char("x", "Master Lucas"))
    assert infer_genders(reg, [_chapter("Nothing here.")]) == []


@pytest.mark.parametrize(
    ("name", "gender"),
    [
        ("pointy hat woman with the painted face", "female"),
        ("Audrey Hall's maidservant", "female"),
        ("the king's daughter", "female"),  # the possessor's noun is someone else
        ("woman with a man's coat", "female"),
        ("Old Man Kim", "male"),
        ("gray-eyed police inspector", None),
        ("Mother and son", None),  # disagreeing nouns: no guess
    ],
)
def test_a_gendered_noun_in_a_descriptive_name(name, gender):
    reg = _registry(_char("x", name))
    infer_genders(reg, [_chapter("Nothing here.")])
    assert reg.get("x").gender == gender


def test_no_empty_genders_means_no_scan():
    reg = _registry(_char("a", "Alice", gender="female"))
    assert infer_genders(reg, []) == []


# -- back-fill service + CLI -------------------------------------------------------------------


def _write_report(book_dir, registry, chapters):
    book_dir.mkdir(parents=True, exist_ok=True)
    report = AttributionReport(
        book_id=book_dir.name,
        provider_id="fake",
        model_id="fake",
        prompt_version="v1",
        registry=registry,
        chapters=chapters,
        registry_notes=["an earlier note"],
    )
    (book_dir / "attribution.json").write_text(report.model_dump_json(), encoding="utf-8")
    return book_dir / "attribution.json"


def test_fill_genders_rewrites_only_when_something_was_filled(tmp_path):
    book_dir = tmp_path / "book-00000000"
    path = _write_report(book_dir, _registry(_char("kwak", "Kwak Jaekang")), [_chapter(*THREE_HE)])
    report, notes = fill_genders(book_dir)
    assert report.registry.get("kwak").gender == "male" and len(notes) == 1
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["registry"]["characters"][0]["gender"] == "male"
    assert saved["registry_notes"] == ["an earlier note", notes[0]]

    before = path.stat().st_mtime_ns
    assert fill_genders(book_dir)[1] == []  # nothing left to fill: no rewrite
    assert path.stat().st_mtime_ns == before


def test_fill_genders_without_attribution_is_a_service_error(tmp_path):
    with pytest.raises(ServiceError, match="no attribution"):
        fill_genders(tmp_path)


def test_infer_genders_cli(tmp_path):
    books = tmp_path / "books"
    _write_report(
        books / "ghost-00000000",
        _registry(_char("kwak", "Kwak Jaekang"), _char("lee", "Lee Jaheon")),
        [_chapter(*THREE_HE)],
    )
    result = CliRunner().invoke(main, ["infer-genders", "ghost", "--books-dir", str(books)])
    assert result.exit_code == 0, result.output
    assert "'Kwak Jaekang': male" in result.output
    assert "filled 1; still unknown: 1" in result.output


# -- wired into attribution --------------------------------------------------------------------


def _bob(chunk, registry, attempt):
    return ChunkAttribution(
        segments=[
            Segment(block_id=b.id, type=SegmentType.DIALOGUE, text=b.text, speaker="Bob")
            if b.text.startswith('"')
            else Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text)
            for b in chunk.owned_blocks
        ],
        characters=[CharacterMention(name="Bob")],  # the model gave no gender
    )


def test_attribution_fills_genders_after_the_alias_pass(tmp_path, monkeypatch):
    import seiyuu.attribute.providers

    book = make_book()
    book.chapters[0].blocks[1].text = '"Hello."'
    book.chapters[0].blocks[3].text = "Bob nodded. He smiled. Bob sat and he ate. Bob left; he ran."
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    (book_dir / "normalized.json").write_text(book.model_dump_json(), encoding="utf-8")
    provider = FakeProvider(_bob)
    monkeypatch.setattr(seiyuu.attribute.providers, "get_provider", lambda *a, **k: provider)

    books = str(tmp_path / "books")
    result = CliRunner().invoke(main, ["attribute", "test-book", "--books-dir", books])
    assert result.exit_code == 0, result.output
    report = AttributionReport.model_validate_json(
        (book_dir / "attribution.json").read_text(encoding="utf-8")
    )
    bob = next(c for c in report.registry.characters if c.canonical_name == "Bob")
    assert bob.gender == "male"
    assert any("gender inferred for 'Bob': male" in n for n in report.registry_notes)
