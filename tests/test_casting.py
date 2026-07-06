"""Smart auto-casting (F4): the pure caster's determinism / collision-freeness / blend
fallback / narrator exclusion, plus the draft_assignment recast + reserve wiring.

Torch-free by construction — no engine/GPU import, so it runs in the default suite."""

import pytest

from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    Segment,
    SegmentType,
)
from seiyuu.services import draft_assignment, suggest_assignment
from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceMeta
from seiyuu.voices.blends import _POOLS
from seiyuu.voices.casting import _DEEP, _YOUNG, cast_book


def _chars(n: int, gender: str = "female") -> list[Character]:
    g = gender[0]  # namespace ids by gender so f/m casts don't share character ids
    return [Character(id=f"{g}{i:03d}", canonical_name=f"N{i}", gender=gender) for i in range(n)]


def _sig(recipe) -> tuple:
    return tuple(recipe)


# --- pure caster: the two hard guarantees ------------------------------------------------


def test_cast_book_is_deterministic_regardless_of_input_order():
    chars = _chars(10, "female") + _chars(10, "male")
    a = cast_book(chars, narrator_preset="af_heart", accent="a")
    b = cast_book(list(reversed(chars)), narrator_preset="af_heart", accent="a")
    assert a == b  # order-stable -> stable settings_hash -> no silent re-render


def test_cast_book_is_collision_free():
    chars = _chars(11, "female") + _chars(11, "male")
    cast = cast_book(chars, narrator_preset="af_heart", accent="a")
    sigs = [_sig(r) for r in cast.values()]
    assert len(sigs) == len(set(sigs)) == len(chars)  # no two characters share a voice


def test_narrator_preset_never_handed_to_a_character():
    cast = cast_book(_chars(6, "female"), narrator_preset="af_heart", accent="a")
    assert all(r != [("af_heart", 1.0)] for r in cast.values())


def test_pool_exhaustion_falls_to_distinct_blends_still_collision_free():
    # American-female bucket has 6 presets; minus the narrator that's 5 distinct singles.
    # 12 female characters MUST overflow into 2-preset blends, still with no collisions.
    cast = cast_book(_chars(12, "female"), narrator_preset="af_heart", accent="a")
    singles = [r for r in cast.values() if len(r) == 1]
    blends = [r for r in cast.values() if len(r) == 2]
    assert len(singles) == 5  # every distinct non-narrator single used
    assert len(blends) == 7  # the rest are distinct blends
    sigs = [_sig(r) for r in cast.values()]
    assert len(set(sigs)) == 12
    # blends are same-family (VoiceMeta blend validator requires one accent family)
    for b in blends:
        assert len({p[:2] for p, _ in b}) == 1


def test_taken_presets_are_reserved_out_of_singles():
    cast = cast_book(
        _chars(3, "female"), narrator_preset="af_heart", accent="a", taken={"af_bella"}
    )
    assert all(r != [("af_bella", 1.0)] for r in cast.values())


def test_family_derives_from_gender_unknown_defaults_female():
    cast = cast_book(
        [
            Character(id="m", canonical_name="M", gender="male"),
            Character(id="f", canonical_name="F", gender="female"),
            Character(id="u", canonical_name="U", gender=None),  # unknown -> female
        ],
        narrator_preset="am_adam",
        accent="a",
    )
    assert cast["m"][0][0].startswith("am")
    assert cast["f"][0][0].startswith("af")
    assert cast["u"][0][0].startswith("af")  # defaulted to the female pool


def test_description_keyword_bias_is_a_tiebreaker_only():
    # A youthful character should land a young-tagged voice; collision-freeness is unaffected.
    chars = [
        Character(id="a_adult", canonical_name="Adult", gender="female", description="stern woman"),
        Character(id="z_kid", canonical_name="Kid", gender="female", description="a young girl"),
    ]
    cast = cast_book(chars, narrator_preset="af_heart", accent="a")
    assert cast["z_kid"][0][0] in _YOUNG
    assert _sig(cast["a_adult"]) != _sig(cast["z_kid"])


def test_trait_tables_are_a_subset_of_the_pools():
    # Guard against drift from kokoro_engine._DESCRIPTIONS: every tagged id must be castable.
    all_presets = {p for pool in _POOLS.values() for p in pool}
    assert _YOUNG <= all_presets
    assert _DEEP <= all_presets


# --- draft_assignment wiring: recast + reserve semantics ---------------------------------


def _report(chars: list[Character]) -> AttributionReport:
    return AttributionReport(
        book_id="book-0000000000000000",
        provider_id="local",
        model_id="m",
        prompt_version="v3",
        registry=CharacterRegistry(characters=chars),
        chapters=[
            AttributedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    Segment(block_id="ch001_b0001", type=SegmentType.NARRATION, text="Narration."),
                ],
            )
        ],
    )


def test_smart_strategy_gives_distinct_voices_where_hash_could_collide(tmp_path):
    report = _report(_chars(8, "female") + _chars(8, "male"))
    lib = VoiceLibrary(tmp_path / "voices")
    draft_assignment(report, lib, default_preset="af_heart", strategy="smart")
    # Each auto voice on disk carries a distinct recipe -> distinct rendered identity.
    recipes = []
    for char in report.registry.characters:
        meta = lib.load(f"{char.id}_auto")
        if meta.kind is VoiceKind.PRESET:
            recipes.append(("preset", meta.preset_id))
        else:
            recipes.append(("blend", tuple((c.preset_id, c.weight) for c in meta.blend)))
    assert len(recipes) == len(set(recipes))  # collision-free at the persisted-voice level


def test_recast_off_skips_existing_but_recast_on_overwrites(tmp_path):
    report = _report(_chars(3, "female"))
    lib = VoiceLibrary(tmp_path / "voices")
    # First draft with the legacy hash strategy so the auto voices are BLENDS.
    draft_assignment(report, lib, default_preset="af_heart", strategy="hash")
    before = lib.load("f000_auto").model_dump()

    # Smart WITHOUT recast: skip-if-exists -> the existing voice is untouched.
    draft_assignment(report, lib, default_preset="af_heart", strategy="smart")
    assert lib.load("f000_auto").model_dump() == before

    # Smart WITH recast: overwrite -> the voice now reflects the smart caster.
    draft_assignment(report, lib, default_preset="af_heart", strategy="smart", recast=True)
    after = lib.load("f000_auto")
    assert after.source == "auto_cast"
    assert after.model_dump() != before


def test_overridden_characters_are_reserved_from_the_caster(tmp_path):
    report = _report(_chars(3, "female"))
    lib = VoiceLibrary(tmp_path / "voices")
    # A pre-existing library voice to override one character onto (a plain preset voice).
    lib.save(
        VoiceMeta(
            voice_id="cloud_x",
            name="Cloud X",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="af_bella",
        )
    )
    assignment = draft_assignment(
        report,
        lib,
        default_preset="af_heart",
        strategy="smart",
        overrides={"f001": "cloud_x"},
    )
    # The override wins for f001; the caster never created f001_auto (budget not spent).
    assert assignment.assignments["f001"] == "cloud_x"
    assert not lib.meta_path("f001_auto").is_file()
    assert lib.meta_path("f000_auto").is_file()  # the other characters still got cast


def test_suggest_assignment_previews_without_writing(tmp_path):
    report = _report(_chars(3, "female"))
    lib = VoiceLibrary(tmp_path / "voices")
    preview = suggest_assignment(report, lib, default_preset="af_heart")
    assert set(preview.assignment.assignments) == {"f000", "f001", "f002"}
    assert sorted(preview.would_create) == ["f000_auto", "f001_auto", "f002_auto"]
    assert preview.would_recast == []
    # Nothing was written to the library.
    assert not lib.meta_path("f000_auto").is_file()

    # After a real draft, a re-preview reports the same voices as would_recast, not create.
    draft_assignment(report, lib, default_preset="af_heart", strategy="smart")
    preview2 = suggest_assignment(report, lib, default_preset="af_heart")
    assert sorted(preview2.would_recast) == ["f000_auto", "f001_auto", "f002_auto"]
    assert preview2.would_create == []


def test_pool_exhaustion_raises_a_loud_error(monkeypatch):
    # A degenerate 1-preset pool yields 1 single and 0 blends: asking for 2 distinct voices
    # must fail LOUDLY rather than silently collide two characters onto one voice.
    from seiyuu.voices import casting

    monkeypatch.setattr(casting, "_POOLS", {("a", "f"): ("af_bella",)})
    with pytest.raises(ValueError, match="exhausted"):
        cast_book(_chars(2, "female"), narrator_preset="none", accent="a")
