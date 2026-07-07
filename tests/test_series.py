"""F5 — series / library voice consistency.

Covers the global series.json round-trip, the pure inheritance resolver (identity match,
deleted-voice skip, case-insensitivity), the draft path inheriting via overrides, the F4
reserved-set + _auto-orphan-skip coupling, precision (no global name match), and book-delete
membership drop.
"""

from seiyuu.attribute.models import (
    AttributionReport,
    Character,
    CharacterRegistry,
)
from seiyuu.services import draft_assignment
from seiyuu.voices import (
    Series,
    SeriesRegistry,
    VoiceAssignment,
    VoiceKind,
    VoiceLibrary,
    VoiceMeta,
    drop_book,
    identity_key,
    load_registry,
    resolve_series_overrides,
    save_cast_to_series,
    save_registry,
    seed_voice_links,
    suggest_links,
)


def _report(book_id: str, chars: list[Character]) -> AttributionReport:
    return AttributionReport(
        book_id=book_id,
        provider_id="local",
        model_id="m",
        prompt_version="v5",
        registry=CharacterRegistry(characters=chars),
        chapters=[],
    )


def _lib_with_preset(tmp_path, voice_id: str, preset_id: str) -> VoiceLibrary:
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(
            voice_id=voice_id,
            name=voice_id,
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id=preset_id,
        )
    )
    return lib


# -- persistence --------------------------------------------------------------------------


def test_series_json_round_trip(tmp_path):
    reg = SeriesRegistry(
        series=[
            Series(
                series_id="holmes_abc123",
                name="Sherlock Holmes",
                book_ids=["book-1"],
                voice_links={"sherlock holmes": "voice_a", "watson": "voice_b"},
            )
        ]
    )
    path = save_registry(tmp_path, reg)
    assert path.is_file() and path.name == "series.json"
    assert load_registry(tmp_path) == reg
    # absent file -> empty registry (not an error)
    assert load_registry(tmp_path / "nowhere").series == []


def test_seed_voice_links_from_assignment():
    report = _report(
        "b1",
        [
            Character(id="sh", canonical_name="Sherlock"),
            Character(id="wat", canonical_name="Watson"),
        ],
    )
    assignment = VoiceAssignment(
        book_id="b1", narrator_voice_id="narr", assignments={"sh": "v_sh", "wat": "v_wat"}
    )
    links = seed_voice_links(report, assignment)
    assert links == {"sherlock": "v_sh", "watson": "v_wat"}


# -- resolve_series_overrides (the core) --------------------------------------------------


def test_resolve_matches_by_casefold_canonical_name(tmp_path):
    lib = _lib_with_preset(tmp_path, "v_sh", "af_bella")
    series = Series(series_id="s1", name="S", voice_links={"sherlock": "v_sh"})
    # New book: same character, DIFFERENT registry id, differently-cased name.
    report = _report("b2", [Character(id="detective", canonical_name="SHERLOCK")])
    assert resolve_series_overrides(report, series, lib) == {"detective": "v_sh"}


def test_resolve_skips_deleted_linked_voice(tmp_path):
    # The linked voice was never saved / was deleted from the library.
    lib = VoiceLibrary(tmp_path / "voices")
    series = Series(series_id="s1", name="S", voice_links={"sherlock": "gone"})
    report = _report("b2", [Character(id="detective", canonical_name="Sherlock")])
    # Degrades to a fresh cast (empty overrides) rather than raising.
    assert resolve_series_overrides(report, series, lib) == {}


def test_resolve_is_alias_aware(tmp_path):
    lib = _lib_with_preset(tmp_path, "v_sh", "af_bella")
    series = Series(series_id="s1", name="S", voice_links={"sherlock": "v_sh"})
    # Canonical name differs, but an alias matches the link.
    report = _report(
        "b2", [Character(id="detective", canonical_name="Mr. Holmes", aliases=["Sherlock"])]
    )
    assert resolve_series_overrides(report, series, lib) == {"detective": "v_sh"}


# -- the draft path inherits via overrides ------------------------------------------------


def test_draft_inherits_series_voice_and_skips_auto_orphan(tmp_path):
    lib = _lib_with_preset(tmp_path, "v_sh", "af_bella")
    series = Series(series_id="s1", name="S", voice_links={"sherlock": "v_sh"})
    report = _report("b2", [Character(id="detective", canonical_name="Sherlock")])
    overrides = resolve_series_overrides(report, series, lib)

    assignment = draft_assignment(report, lib, default_preset="af_heart", overrides=overrides)
    # The linked voice wins over the auto-cast...
    assert assignment.assignments["detective"] == "v_sh"
    # ...and NO throwaway {id}_auto voice was created for the overridden character.
    assert not lib.meta_path("detective_auto").is_file()


def test_hash_strategy_also_skips_auto_orphan_for_overrides(tmp_path):
    lib = _lib_with_preset(tmp_path, "v_sh", "af_bella")
    report = _report("b2", [Character(id="detective", canonical_name="Sherlock")])
    draft_assignment(
        report, lib, default_preset="af_heart", overrides={"detective": "v_sh"}, strategy="hash"
    )
    assert not lib.meta_path("detective_auto").is_file()


# -- F4 coupling: inherited voice's preset is reserved out of the smart caster ------------


def test_smart_cast_reserves_inherited_preset(tmp_path):
    # Inherited voice uses preset af_bella; narrator uses af_heart (the default).
    lib = _lib_with_preset(tmp_path, "v_sh", "af_bella")
    report = _report(
        "b2",
        [
            Character(id="detective", canonical_name="Sherlock", gender="male"),
            Character(id="rival", canonical_name="Moriarty", gender="female"),
        ],
    )
    assignment = draft_assignment(
        report,
        lib,
        default_preset="af_heart",
        overrides={"detective": "v_sh"},
        strategy="smart",
    )
    assert assignment.assignments["detective"] == "v_sh"  # inherited
    # The new character was cast onto a DISTINCT preset — never the inherited af_bella
    # (nor the narrator's af_heart).
    rival_meta = lib.load(assignment.assignments["rival"])
    rival_presets = (
        {rival_meta.preset_id}
        if rival_meta.kind is VoiceKind.PRESET
        else {c.preset_id for c in rival_meta.blend}
    )
    assert "af_bella" not in rival_presets
    assert "af_heart" not in rival_presets


# -- precision: no global name match across unrelated series ------------------------------


def test_precision_no_cross_series_link(tmp_path):
    """Two distinct same-named characters in UNRELATED books are never auto-linked: matching is
    scoped to a DECLARED series' voice_links, not a global name pool."""
    lib = VoiceLibrary(tmp_path / "voices")
    for vid, preset in (("v_hero_john", "af_bella"), ("v_villain_john", "af_nicole")):
        lib.save(
            VoiceMeta(
                voice_id=vid, name=vid, kind=VoiceKind.PRESET, engine="kokoro", preset_id=preset
            )
        )
    # Series 1 (book-1) links its own "John" to v_hero_john.
    s1 = Series(
        series_id="s1", name="Alpha", book_ids=["book-1"], voice_links={"john": "v_hero_john"}
    )
    # Series 2 (book-2) is a DIFFERENT series with its own "John".
    s2 = Series(series_id="s2", name="Beta", book_ids=["book-2"])

    report2 = _report("book-2", [Character(id="villain", canonical_name="John")])
    # book-2's John inherits nothing from s2 (no link yet) — and crucially never picks up
    # s1's v_hero_john, because resolution is scoped to the series it is asked about.
    assert resolve_series_overrides(report2, s2, lib) == {}
    assert suggest_links(report2, s2, lib) == []
    # The two series keep separate link namespaces in one registry (no global pool).
    reg = SeriesRegistry(series=[s1, s2])
    save_registry(tmp_path, reg)
    reloaded = load_registry(tmp_path)
    assert reloaded.get("s1").voice_links == {"john": "v_hero_john"}
    assert reloaded.get("s2").voice_links == {}


def test_suggest_links_surfaces_matches_for_confirmation(tmp_path):
    lib = _lib_with_preset(tmp_path, "v_sh", "af_bella")
    series = Series(series_id="s1", name="S", voice_links={"sherlock": "v_sh"})
    report = _report(
        "b2",
        [
            Character(id="detective", canonical_name="Sherlock"),
            Character(id="stranger", canonical_name="Nobody"),
        ],
    )
    suggestions = suggest_links(report, series, lib)
    assert len(suggestions) == 1
    assert suggestions[0].character_id == "detective"
    assert suggestions[0].voice_id == "v_sh"
    assert suggestions[0].voice_exists is True


# -- write-back + membership drop ---------------------------------------------------------


def test_save_cast_to_series_writes_back():
    series = Series(series_id="s1", name="S", voice_links={"sherlock": "old"})
    report = _report(
        "b2",
        [
            Character(id="sh", canonical_name="Sherlock"),
            Character(id="wat", canonical_name="Watson"),
        ],
    )
    assignment = VoiceAssignment(
        book_id="b2", narrator_voice_id="narr", assignments={"sh": "new_sh", "wat": "v_wat"}
    )
    changed = save_cast_to_series(series, report, assignment)
    assert set(changed) == {"sherlock", "watson"}  # sherlock updated, watson added
    assert series.voice_links == {"sherlock": "new_sh", "watson": "v_wat"}


def test_drop_book_removes_membership_everywhere(tmp_path):
    reg = SeriesRegistry(
        series=[
            Series(series_id="s1", name="A", book_ids=["book-1", "book-2"]),
            Series(series_id="s2", name="B", book_ids=["book-3"]),
        ]
    )
    affected = drop_book(reg, "book-1")
    assert affected == ["s1"]
    assert reg.get("s1").book_ids == ["book-2"]
    assert reg.get("s2").book_ids == ["book-3"]
    # links (name-keyed) are untouched; only membership drops
    save_registry(tmp_path, reg)
    assert load_registry(tmp_path).get("s1").book_ids == ["book-2"]


def test_identity_key_normalizes():
    assert identity_key("  Sherlock Holmes ") == "sherlock holmes"
    assert identity_key("SHERLOCK") == identity_key("sherlock")
