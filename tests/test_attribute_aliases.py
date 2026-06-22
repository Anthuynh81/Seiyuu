"""Deterministic alias resolution: correct merges + the adversarial must-stay-separate cases.

Over-merging two distinct characters is the worst failure, so the guards (gender/generation,
ambiguous-surname) get the most coverage.
"""

from seiyuu.attribute.aliases import resolve_registry_aliases
from seiyuu.attribute.models import (
    AttributedChapter,
    Character,
    CharacterRegistry,
    Segment,
    SegmentType,
)


def _reg(*chars: Character) -> CharacterRegistry:
    return CharacterRegistry(characters=list(chars))


def _ch(*speakers: str) -> list[AttributedChapter]:
    # One dialogue segment per speaker id, so each has >=1 attributed line.
    segs = [
        Segment(block_id="ch001_b0001", type=SegmentType.DIALOGUE, text='"x"', speaker=s)
        for s in speakers
    ]
    return [AttributedChapter(index=1, title="C", segments=segs)]


def _ids(reg: CharacterRegistry) -> set[str]:
    return {c.id for c in reg.characters}


# --- correct merges ---


def test_honorific_variant_auto_merges():
    reg = _reg(
        Character(
            id="mr_darcy", canonical_name="Mr. Darcy", gender="male", first_appearance="ch004_b0001"
        ),
        Character(
            id="darcy", canonical_name="Darcy", gender="male", first_appearance="ch004_b0007"
        ),
    )
    remap, notes = resolve_registry_aliases(reg, _ch("mr_darcy", "darcy"))
    assert remap == {"darcy": "mr_darcy"}  # earlier first_appearance wins
    assert _ids(reg) == {"mr_darcy"}
    assert "Darcy" in reg.get("mr_darcy").aliases


def test_subset_alias_consolidates():
    reg = _reg(
        Character(
            id="elizabeth_bennet",
            canonical_name="Elizabeth Bennet",
            aliases=["Elizabeth"],
            gender="female",
            first_appearance="ch003_b0002",
        ),
        Character(
            id="elizabeth",
            canonical_name="Elizabeth",
            gender="female",
            first_appearance="ch003_b0009",
        ),
    )
    remap, _ = resolve_registry_aliases(reg, _ch("elizabeth_bennet", "elizabeth"))
    assert remap == {"elizabeth": "elizabeth_bennet"}
    assert _ids(reg) == {"elizabeth_bennet"}


def test_transitive_chain_collapses_to_one_survivor():
    reg = _reg(
        Character(
            id="darcy", canonical_name="Darcy", gender="male", first_appearance="ch004_b0009"
        ),
        Character(
            id="mr_darcy", canonical_name="Mr. Darcy", gender="male", first_appearance="ch004_b0001"
        ),
        Character(
            id="fitzwilliam_darcy",
            canonical_name="Fitzwilliam Darcy",
            aliases=["Mr. Darcy", "Darcy"],
            gender="male",
            first_appearance="ch010_b0001",
        ),
    )
    remap, _ = resolve_registry_aliases(reg, _ch("darcy", "mr_darcy", "fitzwilliam_darcy"))
    survivors = _ids(reg)
    assert len(survivors) == 1
    # every remap target is the single survivor (no dangling loser ids)
    assert set(remap.values()) <= survivors


# --- adversarial: must stay separate ---


def test_bennet_gender_conflict_never_merges():
    reg = _reg(
        Character(id="mr_bennet", canonical_name="Mr. Bennet", gender="male"),
        Character(id="mrs_bennet", canonical_name="Mrs. Bennet", gender="female"),
    )
    remap, notes = resolve_registry_aliases(reg, _ch("mr_bennet", "mrs_bennet"))
    assert remap == {}
    assert _ids(reg) == {"mr_bennet", "mrs_bennet"}
    assert any("ambiguous" in n and "bennet" in n.lower() for n in notes)


def test_lucas_family_never_over_merges():
    reg = _reg(
        Character(id="charlotte_lucas", canonical_name="Charlotte Lucas", gender="female"),
        Character(id="miss_lucas", canonical_name="Miss Lucas", gender="female"),
        Character(id="lady_lucas", canonical_name="Lady Lucas", gender="female"),
        Character(id="young_lucas", canonical_name="Young Lucas"),
    )
    remap, _ = resolve_registry_aliases(
        reg, _ch("charlotte_lucas", "miss_lucas", "lady_lucas", "young_lucas")
    )
    # Miss (unmarried) vs Lady (adult) is a generation conflict; the others have distinct
    # stripped names — nothing merges.
    assert remap == {}
    assert _ids(reg) == {"charlotte_lucas", "miss_lucas", "lady_lucas", "young_lucas"}


def test_first_name_and_nickname_never_auto_merge():
    reg = _reg(
        Character(id="elizabeth_bennet", canonical_name="Elizabeth Bennet", gender="female"),
        Character(id="elizabeth", canonical_name="Elizabeth", gender="female"),
        Character(id="lizzy", canonical_name="Lizzy", gender="female"),
        Character(id="miss_bennet", canonical_name="Miss Bennet", gender="female"),
    )
    remap, _ = resolve_registry_aliases(
        reg, _ch("elizabeth_bennet", "elizabeth", "lizzy", "miss_bennet")
    )
    # No record's name set subsets another and no stripped names coincide -> no merge.
    assert remap == {}
    assert len(reg.characters) == 4


def test_same_given_name_distinct_people_not_merged():
    reg = _reg(
        Character(id="elizabeth_bennet", canonical_name="Elizabeth Bennet", gender="female"),
        Character(id="elizabeth_lucas", canonical_name="Elizabeth Lucas", gender="female"),
    )
    remap, _ = resolve_registry_aliases(reg, _ch("elizabeth_bennet", "elizabeth_lucas"))
    assert remap == {}
    assert len(reg.characters) == 2


# --- segment remap + flags ---


def test_segments_remapped_to_survivor():
    reg = _reg(
        Character(
            id="mr_darcy", canonical_name="Mr. Darcy", gender="male", first_appearance="ch004_b0001"
        ),
        Character(
            id="darcy", canonical_name="Darcy", gender="male", first_appearance="ch004_b0007"
        ),
    )
    chapters = [
        AttributedChapter(
            index=4,
            title="III",
            segments=[
                Segment(
                    block_id="ch004_b0001",
                    type=SegmentType.DIALOGUE,
                    text='"a"',
                    speaker="mr_darcy",
                ),
                Segment(
                    block_id="ch004_b0007", type=SegmentType.DIALOGUE, text='"b"', speaker="darcy"
                ),
            ],
        )
    ]
    remap, _ = resolve_registry_aliases(reg, chapters)
    # caller applies remap; emulate it and assert no segment points at a removed id
    for chx in chapters:
        chx.segments = [
            s.model_copy(update={"speaker": remap[s.speaker]}) if s.speaker in remap else s
            for s in chx.segments
        ]
    assert {s.speaker for s in chapters[0].segments} == {"mr_darcy"}


def test_low_evidence_record_flagged_not_removed():
    reg = _reg(Character(id="ghost", canonical_name="Ghost"))  # null metadata, no segments
    remap, notes = resolve_registry_aliases(reg, _ch())  # no attributed speakers
    assert remap == {}
    assert _ids(reg) == {"ghost"}  # never deleted
    assert any("low-evidence" in n and "ghost" in n for n in notes)


def test_no_op_on_clean_registry():
    reg = _reg(
        Character(id="alice", canonical_name="Alice", gender="female"),
        Character(id="bob", canonical_name="Bob", gender="male"),
    )
    remap, notes = resolve_registry_aliases(reg, _ch("alice", "bob"))
    assert remap == {} and notes == [] and len(reg.characters) == 2
