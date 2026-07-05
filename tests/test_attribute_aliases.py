"""Deterministic alias resolution: correct merges + the adversarial must-stay-separate cases.

Over-merging two distinct characters is the worst failure, so the guards (gender/generation,
ambiguous-surname) get the most coverage.
"""

from collections import Counter
from pathlib import Path

from fake_alias_resolver import (
    FakeAliasResolver,
    ScriptedAdjudicatorProvider,
    approve_all,
)
from seiyuu.attribute.adjudicate import LLMAdjudicator
from seiyuu.attribute.aliases import _generate_candidates, resolve_registry_aliases
from seiyuu.attribute.cache import AttributionCache
from seiyuu.attribute.models import (
    AttributedChapter,
    Character,
    CharacterRegistry,
    Segment,
    SegmentType,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


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


# --- opt-in LLM adjudication: generator units ---


def _elizabeth_and_full() -> CharacterRegistry:
    return _reg(
        Character(
            id="elizabeth_bennet",
            canonical_name="Elizabeth Bennet",
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


def test_g1_containment_emits_single_candidate():
    cands, flags = _generate_candidates(
        _elizabeth_and_full(), Counter(), cap=40, use_nicknames=True
    )
    assert len(cands) == 1
    assert cands[0].generator == "G1"
    assert cands[0].pair_id == "elizabeth::elizabeth_bennet"
    assert flags == []


def test_g1_multi_leader_flags_not_candidate():
    reg = _reg(
        Character(id="elizabeth", canonical_name="Elizabeth", gender="female"),
        Character(id="elizabeth_bennet", canonical_name="Elizabeth Bennet", gender="female"),
        Character(id="elizabeth_lucas", canonical_name="Elizabeth Lucas", gender="female"),
    )
    cands, flags = _generate_candidates(reg, Counter(), cap=40, use_nicknames=True)
    assert cands == []
    assert any("ambiguous given name 'elizabeth'" in n for n in flags)


def test_g2_emits_title_surname_but_not_given_given():
    title_reg = _reg(
        Character(id="miss_bennet", canonical_name="Miss Bennet", gender="female"),
        Character(id="elizabeth_bennet", canonical_name="Elizabeth Bennet", gender="female"),
    )
    cands, _ = _generate_candidates(title_reg, Counter(), cap=40, use_nicknames=False)
    assert [c.generator for c in cands] == ["G2"]

    siblings = _reg(
        Character(id="elizabeth_bennet", canonical_name="Elizabeth Bennet", gender="female"),
        Character(id="jane_bennet", canonical_name="Jane Bennet", gender="female"),
    )
    cands2, _ = _generate_candidates(siblings, Counter(), cap=40, use_nicknames=False)
    assert cands2 == []  # two given-name records sharing a surname are NEVER a candidate


def test_g2_title_matching_two_sisters_is_flagged_not_candidate():
    reg = _reg(
        Character(id="miss_bennet", canonical_name="Miss Bennet", gender="female"),
        Character(id="elizabeth_bennet", canonical_name="Elizabeth Bennet", gender="female"),
        Character(id="jane_bennet", canonical_name="Jane Bennet", gender="female"),
    )
    cands, flags = _generate_candidates(reg, Counter(), cap=40, use_nicknames=False)
    assert cands == []
    assert any("ambiguous title 'Miss Bennet'" in n for n in flags)


def test_generator_prefilters_conflict_pairs():
    reg = _reg(
        Character(id="miss_bennet", canonical_name="Miss Bennet", gender="female"),
        Character(id="john_bennet", canonical_name="John Bennet", gender="male"),
    )
    cands, _ = _generate_candidates(reg, Counter(), cap=40, use_nicknames=False)
    assert cands == []  # gender clash -> dropped before it can be adjudicated


def test_g3_nickname_only_from_table():
    linked = _reg(
        Character(id="lizzy", canonical_name="Lizzy", gender="female"),
        Character(id="elizabeth", canonical_name="Elizabeth", gender="female"),
    )
    cands, _ = _generate_candidates(linked, Counter(), cap=40, use_nicknames=True)
    assert [c.generator for c in cands] == ["G3"]

    # A name NOT in the curated table produces no candidate (fuzzy/edit-distance stays OFF).
    unlisted = _reg(
        Character(id="lizette", canonical_name="Lizette", gender="female"),
        Character(id="lizetta", canonical_name="Lizetta", gender="female"),
    )
    cands2, _ = _generate_candidates(unlisted, Counter(), cap=40, use_nicknames=True)
    assert cands2 == []


def test_candidate_cap_truncates_and_flags():
    reg = _reg(
        Character(id="anna", canonical_name="Anna", gender="female"),
        Character(id="anna_smith", canonical_name="Anna Smith", gender="female"),
        Character(id="bella", canonical_name="Bella", gender="female"),
        Character(id="bella_jones", canonical_name="Bella Jones", gender="female"),
        Character(id="cara", canonical_name="Cara", gender="female"),
        Character(id="cara_lee", canonical_name="Cara Lee", gender="female"),
    )
    cands, flags = _generate_candidates(reg, Counter(), cap=2, use_nicknames=True)
    assert len(cands) == 2
    assert any("over the cap of 2" in n for n in flags)


# --- opt-in LLM adjudication: merge behavior + guards ---


def test_resolver_none_is_byte_identical_default():
    # A registry that WOULD produce a G1 candidate, but with no resolver nothing merges.
    reg = _elizabeth_and_full()
    remap, notes = resolve_registry_aliases(reg, _ch("elizabeth_bennet", "elizabeth"))
    assert remap == {}
    assert len(reg.characters) == 2
    assert notes == []


def test_true_positive_first_name_merges_and_remaps_segments():
    reg = _elizabeth_and_full()
    chapters = [
        AttributedChapter(
            index=3,
            title="C",
            segments=[
                Segment(
                    block_id="ch003_b0002",
                    type=SegmentType.DIALOGUE,
                    text='"How so?"',
                    speaker="elizabeth_bennet",
                ),
                Segment(
                    block_id="ch003_b0009",
                    type=SegmentType.DIALOGUE,
                    text='"I do not cough for my own amusement."',
                    speaker="elizabeth",
                ),
            ],
        )
    ]
    remap, notes = resolve_registry_aliases(
        reg, chapters, resolver=FakeAliasResolver(approve_all(0.95))
    )
    assert remap == {"elizabeth": "elizabeth_bennet"}
    assert _ids(reg) == {"elizabeth_bennet"}
    assert "Elizabeth" in reg.get("elizabeth_bennet").aliases
    texts_before = [s.text for chx in chapters for s in chx.segments]
    for chx in chapters:
        chx.segments = [
            s.model_copy(update={"speaker": remap[s.speaker]}) if s.speaker in remap else s
            for s in chx.segments
        ]
    assert {s.speaker for chx in chapters for s in chx.segments} == {"elizabeth_bennet"}
    assert [s.text for chx in chapters for s in chx.segments] == texts_before  # text untouched
    assert any("merged 'Elizabeth' -> 'Elizabeth Bennet'" in n for n in notes)


def test_nickname_merges_via_curated_table():
    reg = _reg(
        Character(
            id="elizabeth", canonical_name="Elizabeth", gender="female",
            first_appearance="ch001_b0001",
        ),
        Character(
            id="lizzy", canonical_name="Lizzy", gender="female", first_appearance="ch001_b0009"
        ),
    )  # fmt: skip
    remap, _ = resolve_registry_aliases(
        reg, _ch("elizabeth", "lizzy"), resolver=FakeAliasResolver(approve_all(0.9))
    )
    assert len(reg.characters) == 1
    assert len(remap) == 1


def test_below_threshold_approval_is_flagged_not_merged():
    reg = _elizabeth_and_full()
    remap, notes = resolve_registry_aliases(
        reg,
        _ch("elizabeth_bennet", "elizabeth"),
        resolver=FakeAliasResolver(lambda _c: (True, 0.5)),
        confidence_threshold=0.85,
    )
    assert remap == {}
    assert len(reg.characters) == 2
    assert any("flagged, not merged" in n for n in notes)


def test_adversarial_gender_clash_never_merges_even_when_approved():
    reg = _reg(
        Character(id="mr_bennet", canonical_name="Mr. Bennet", gender="male"),
        Character(id="mrs_bennet", canonical_name="Mrs. Bennet", gender="female"),
    )
    remap, _ = resolve_registry_aliases(
        reg, _ch("mr_bennet", "mrs_bennet"), resolver=FakeAliasResolver(approve_all(1.0))
    )
    assert remap == {}
    assert _ids(reg) == {"mr_bennet", "mrs_bennet"}


def test_adversarial_siblings_never_merge_even_when_approved():
    reg = _reg(
        Character(id="elizabeth_bennet", canonical_name="Elizabeth Bennet", gender="female"),
        Character(id="jane_bennet", canonical_name="Jane Bennet", gender="female"),
    )
    remap, _ = resolve_registry_aliases(
        reg, _ch("elizabeth_bennet", "jane_bennet"), resolver=FakeAliasResolver(approve_all(1.0))
    )
    assert remap == {}  # G2 never emits distinct-given-name same-surname pairs
    assert _ids(reg) == {"elizabeth_bennet", "jane_bennet"}


def test_adversarial_multi_leader_never_merges_even_when_approved():
    reg = _reg(
        Character(id="elizabeth", canonical_name="Elizabeth", gender="female"),
        Character(id="elizabeth_bennet", canonical_name="Elizabeth Bennet", gender="female"),
        Character(id="elizabeth_lucas", canonical_name="Elizabeth Lucas", gender="female"),
    )
    remap, notes = resolve_registry_aliases(
        reg,
        _ch("elizabeth", "elizabeth_bennet", "elizabeth_lucas"),
        resolver=FakeAliasResolver(approve_all(1.0)),
    )
    assert remap == {}  # bare 'Elizabeth' leads two full names -> flag only, no candidate
    assert len(reg.characters) == 3
    assert any("ambiguous given name 'elizabeth'" in n for n in notes)


def test_rejected_pair_stays_flagged():
    reg = _elizabeth_and_full()
    remap, notes = resolve_registry_aliases(
        reg,
        _ch("elizabeth_bennet", "elizabeth"),
        resolver=FakeAliasResolver(lambda _c: (False, 0.0)),
    )
    assert remap == {}
    assert any("adjudicator rejected" in n for n in notes)


# --- determinism + per-book cache ---


def test_cache_fires_llm_once_across_reruns(tmp_path):
    provider = ScriptedAdjudicatorProvider(confidence=1.0)
    chapters = _ch("elizabeth_bennet", "elizabeth")
    with AttributionCache(tmp_path / "attribution.db") as cache:
        adjudicator = LLMAdjudicator(
            provider, cache=cache, book_id="book", prompt_version="v1", prompts_dir=_PROMPTS_DIR
        )
        reg1 = _elizabeth_and_full()
        remap1, _ = resolve_registry_aliases(reg1, chapters, resolver=adjudicator)
        reg2 = _elizabeth_and_full()
        remap2, _ = resolve_registry_aliases(reg2, chapters, resolver=adjudicator)
    # The candidate set is unchanged, so the "LLM" is consulted exactly once; the second run
    # replays the cached verdicts and produces a byte-identical merge.
    assert provider.calls == 1
    assert remap1 == remap2 == {"elizabeth": "elizabeth_bennet"}
    assert _ids(reg1) == _ids(reg2) == {"elizabeth_bennet"}
