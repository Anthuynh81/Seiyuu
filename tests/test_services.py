"""M6a-6 service layer: edits overlay, attribution lifecycle, characters aggregate,
assignment drafting, voice deletion guard."""

import pytest

from factories import make_book
from fake_provider import FakeProvider
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
from seiyuu.gpu import GpuResourceManager
from seiyuu.services import (
    MergeCharacters,
    ReassignSegment,
    RenameCharacter,
    ServiceError,
    append_edit,
    apply_edits,
    characters_overview,
    delete_voice,
    draft_assignment,
    load_edits,
    load_report,
    pop_edit,
    run_attribution,
    save_assignment,
    voice_references,
)
from seiyuu.services.edits import EditLog
from seiyuu.settings import get_settings
from seiyuu.voices import VoiceAssignment, VoiceKind, VoiceLibrary, VoiceMeta


def _report() -> AttributionReport:
    return AttributionReport(
        book_id="test-book-00000000",
        provider_id="local",
        model_id="m",
        prompt_version="v3",
        registry=CharacterRegistry(
            characters=[
                Character(id="alice", canonical_name="Alice", gender="female"),
                Character(id="al", canonical_name="Al", aliases=["Big Al"]),
            ]
        ),
        chapters=[
            AttributedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    Segment(block_id="ch001_b0001", type=SegmentType.NARRATION, text="Narration."),
                    Segment(
                        block_id="ch001_b0002",
                        type=SegmentType.DIALOGUE,
                        text="Hi.",
                        speaker="alice",
                    ),  # fmt: skip
                    Segment(block_id="ch001_b0002", type=SegmentType.NARRATION, text="she said."),
                    Segment(
                        block_id="ch001_b0003", type=SegmentType.DIALOGUE, text="Yo.", speaker="al"
                    ),
                ],
            )
        ],
    )


# --- edits overlay semantics ---


def test_rename_keeps_old_name_as_alias():
    edited, warnings = apply_edits(
        _report(), EditLog(ops=[RenameCharacter(character_id="alice", new_name="Alicia")])
    )
    assert warnings == []
    char = edited.registry.get("alice")
    assert char.canonical_name == "Alicia" and "Alice" in char.aliases


def test_merge_remaps_segments_and_absorbs_names():
    edited, warnings = apply_edits(
        _report(), EditLog(ops=[MergeCharacters(loser_id="al", winner_id="alice")])
    )
    assert warnings == []
    assert edited.registry.get("al") is None
    alice = edited.registry.get("alice")
    assert "Al" in alice.aliases and "Big Al" in alice.aliases
    speakers = [s.speaker for s in edited.chapters[0].segments]
    assert speakers == [None, "alice", None, "alice"]  # al's line moved to alice


def test_reassign_flips_types_to_keep_the_segment_invariant():
    ops = [
        # narration -> dialogue, then dialogue -> narration
        ReassignSegment(block_id="ch001_b0002", segment_index=1, speaker="al"),
        ReassignSegment(block_id="ch001_b0003", segment_index=0, speaker=None),
    ]
    edited, warnings = apply_edits(_report(), EditLog(ops=ops))
    assert warnings == []
    b2_second = edited.chapters[0].segments[2]
    assert b2_second.speaker == "al" and b2_second.type is SegmentType.DIALOGUE
    b3 = edited.chapters[0].segments[3]
    assert b3.speaker is None and b3.type is SegmentType.NARRATION


def test_stale_ops_warn_and_skip_never_crash():
    ops = [
        RenameCharacter(character_id="ghost", new_name="X"),
        MergeCharacters(loser_id="ghost", winner_id="alice"),
        ReassignSegment(block_id="ch009_b0009", segment_index=0, speaker="alice"),
        ReassignSegment(block_id="ch001_b0002", segment_index=9, speaker="alice"),
        ReassignSegment(block_id="ch001_b0001", segment_index=0, speaker="ghost"),
    ]
    edited, warnings = apply_edits(_report(), EditLog(ops=ops))
    assert len(warnings) == 5  # every op skipped, with a reason
    assert edited.registry.get("alice").canonical_name == "Alice"  # nothing changed


def test_apply_does_not_mutate_the_raw_report():
    raw = _report()
    apply_edits(raw, EditLog(ops=[MergeCharacters(loser_id="al", winner_id="alice")]))
    assert raw.registry.get("al") is not None  # raw untouched


def test_edit_log_roundtrip_append_and_undo(tmp_path):
    op = RenameCharacter(character_id="alice", new_name="Alicia")
    append_edit(tmp_path, op)
    append_edit(tmp_path, MergeCharacters(loser_id="al", winner_id="alice"))
    assert len(load_edits(tmp_path).ops) == 2
    popped = pop_edit(tmp_path)
    assert isinstance(popped, MergeCharacters)
    assert load_edits(tmp_path).ops == [op]
    pop_edit(tmp_path)
    assert pop_edit(tmp_path) is None


def test_reassign_anchor_blocks_silent_retargeting():
    """THE review find: a block flagged in run A (one fallback segment) gets reassigned;
    run B re-splits it — the op must SKIP with a warning, never retarget the new text."""
    op = ReassignSegment(
        block_id="ch001_b0002", segment_index=0, speaker="al", text_anchor="Original text"
    )
    edited, warnings = apply_edits(_report(), EditLog(ops=[op]))
    assert len(warnings) == 1 and "different text" in warnings[0]
    assert edited.chapters[0].segments[1].speaker == "alice"  # untouched

    matching = op.model_copy(update={"text_anchor": "Hi."})
    edited, warnings = apply_edits(_report(), EditLog(ops=[matching]))
    assert warnings == [] and edited.chapters[0].segments[1].speaker == "al"


def test_character_anchors_block_id_rebinding():
    """Character ids are discovery-order slugs — the same id can denote a DIFFERENT
    character after re-attribution. Anchored ops must skip, not misapply."""
    rename = RenameCharacter(character_id="alice", new_name="X", expected_name="Someone Else")
    merge = MergeCharacters(
        loser_id="al", winner_id="alice",
        expected_loser_name="Al", expected_winner_name="Someone Else",
    )  # fmt: skip
    edited, warnings = apply_edits(_report(), EditLog(ops=[rename, merge]))
    assert len(warnings) == 2 and all("re-record" in w for w in warnings)
    assert edited.registry.get("alice").canonical_name == "Alice"
    assert edited.registry.get("al") is not None


def test_reassign_marks_human_edits_full_confidence():
    op = ReassignSegment(block_id="ch001_b0003", segment_index=0, speaker="alice")
    edited, _ = apply_edits(_report(), EditLog(ops=[op]))
    assert edited.chapters[0].segments[3].confidence == 1.0  # review queue drains


def test_op_models_refuse_garbage():
    with pytest.raises(ValueError):
        ReassignSegment(block_id="ch001_b0001", segment_index=-1, speaker="x")  # never stale
    with pytest.raises(ValueError):
        RenameCharacter(character_id="alice", new_name="")  # nameless character
    with pytest.raises(ValueError):
        MergeCharacters(loser_id="alice", winner_id="alice")  # self-merge


def test_record_edit_validates_and_anchors(tmp_path):
    from seiyuu.services import record_edit

    (tmp_path / "attribution.json").write_text(_report().model_dump_json(), encoding="utf-8")
    with pytest.raises(ServiceError, match="unknown character"):
        record_edit(tmp_path, RenameCharacter(character_id="ghost", new_name="X"))
    with pytest.raises(ServiceError, match="out of range"):
        record_edit(
            tmp_path, ReassignSegment(block_id="ch001_b0002", segment_index=9, speaker="al")
        )

    recorded = record_edit(
        tmp_path, ReassignSegment(block_id="ch001_b0002", segment_index=0, speaker="al")
    )
    assert recorded.text_anchor == "Hi."  # anchored to what the user was looking at
    recorded = record_edit(tmp_path, RenameCharacter(character_id="alice", new_name="Alicia"))
    assert recorded.expected_name == "Alice"
    assert len(load_edits(tmp_path).ops) == 2


def test_corrupt_edits_file_is_a_service_error(tmp_path):
    (tmp_path / "attribution.json").write_text(_report().model_dump_json(), encoding="utf-8")
    (tmp_path / "edits.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ServiceError, match="corrupt edits file"):
        load_report(tmp_path)


# --- attribution service ---


def _alice_script(chunk, registry, attempt):
    return ChunkAttribution(
        segments=[
            Segment(block_id=b.id, type=SegmentType.DIALOGUE, text=b.text, speaker="Alice")
            for b in chunk.owned_blocks
        ],
        characters=[CharacterMention(name="Alice", gender="female")],
    )


def test_run_attribution_writes_raw_but_returns_effective(tmp_path):
    book = make_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    append_edit(book_dir, RenameCharacter(character_id="alice", new_name="Alicia"))

    report = run_attribution(
        book,
        book_dir,
        cfg=get_settings(),
        provider=FakeProvider(_alice_script),
        progress=None,
        gpu=GpuResourceManager(),
    )
    assert report.registry.characters[0].canonical_name == "Alicia"  # overlay applied
    raw = AttributionReport.model_validate_json(
        (book_dir / "attribution.json").read_text(encoding="utf-8")
    )
    assert raw.registry.characters[0].canonical_name == "Alice"  # file stays RAW


def test_run_attribution_holds_and_frees_the_gpu(tmp_path):
    acquired: list[str] = []

    class SpyGpu(GpuResourceManager):
        def acquire(self, consumer, name):
            acquired.append(name)
            return super().acquire(consumer, name)

    book = make_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    gpu = SpyGpu()
    run_attribution(
        book, book_dir, cfg=get_settings(), provider=FakeProvider(_alice_script), gpu=gpu
    )
    assert acquired == ["llm:fake:fake-1.0"]  # the LLM is a managed GPU consumer now
    assert gpu.resident is None  # freed at the end


def test_run_attribution_frees_gpu_on_failure(tmp_path):
    def explode(chunk, registry, attempt):
        raise RuntimeError("provider blew up")

    book = make_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    gpu = GpuResourceManager()
    with pytest.raises(RuntimeError):
        run_attribution(book, book_dir, cfg=get_settings(), provider=FakeProvider(explode), gpu=gpu)
    assert gpu.resident is None  # VRAM never stays held by a dead run


def test_partial_reattribution_merges_instead_of_replacing(tmp_path):
    """`attribute --chapter 2` on a fully-attributed book must not shrink the effective
    book to one chapter for every downstream reader."""

    def bob_script(chunk, registry, attempt):
        return ChunkAttribution(
            segments=[
                Segment(block_id=b.id, type=SegmentType.DIALOGUE, text=b.text, speaker="Bob")
                for b in chunk.owned_blocks
            ],
            characters=[CharacterMention(name="Bob", gender="male")],
        )

    book = make_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    cfg = get_settings()
    run_attribution(
        book, book_dir, cfg=cfg, provider=FakeProvider(_alice_script), gpu=GpuResourceManager()
    )
    # re-attribute ONLY chapter 2 with a different model (distinct model id, or the
    # chunk cache would serve run 1's answer)
    partial = run_attribution(
        book, book_dir, cfg=cfg, provider=FakeProvider(bob_script, model="bob-2.0"),
        chapters=(2,), gpu=GpuResourceManager(),
    )  # fmt: skip
    assert [c.index for c in partial.chapters] == [1, 2]  # chapter 1 preserved
    assert {c.id for c in partial.registry.characters} == {"alice", "bob"}
    ch2_speakers = {s.speaker for s in partial.chapters[1].segments if s.speaker}
    assert ch2_speakers == {"bob"}  # chapter 2 re-attributed
    ch1_speakers = {s.speaker for s in partial.chapters[0].segments if s.speaker}
    assert ch1_speakers == {"alice"}  # chapter 1 untouched


def test_cloud_provider_never_touches_the_gpu_manager(tmp_path):
    class ExplodingGpu(GpuResourceManager):
        def acquire(self, consumer, name):  # pragma: no cover - failing is the assertion
            raise AssertionError("cloud attribution must not acquire the GPU")

        def free_all(self):
            raise AssertionError("cloud attribution must not free other consumers")

    provider = FakeProvider(_alice_script)
    provider.uses_gpu = False  # anthropic-style: pure network
    book = make_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    run_attribution(book, book_dir, cfg=get_settings(), provider=provider, gpu=ExplodingGpu())


def test_load_report_applies_overlay_and_missing_is_actionable(tmp_path):
    with pytest.raises(ServiceError, match="seiyuu attribute"):
        load_report(tmp_path)
    (tmp_path / "attribution.json").write_text(_report().model_dump_json(), encoding="utf-8")
    append_edit(tmp_path, RenameCharacter(character_id="alice", new_name="Alicia"))
    report, warnings = load_report(tmp_path)
    assert warnings == []
    assert report.registry.get("alice").canonical_name == "Alicia"


# --- characters aggregate ---


def test_characters_overview_counts_samples_and_edits(tmp_path):
    (tmp_path / "attribution.json").write_text(_report().model_dump_json(), encoding="utf-8")
    append_edit(tmp_path, MergeCharacters(loser_id="al", winner_id="alice"))
    append_edit(tmp_path, RenameCharacter(character_id="ghost", new_name="X"))  # stale

    overview = characters_overview(tmp_path, confidence_threshold=0.7, sample_lines=2)
    assert [c.id for c in overview.characters] == ["alice"]  # merged, busiest first
    alice = overview.characters[0]
    assert alice.line_count == 2 and alice.sample_lines == ["Hi.", "Yo."]
    assert overview.narration_segments == 2
    assert len(overview.edit_warnings) == 1  # the stale rename surfaced, not crashed


# --- assignment service ---


def test_draft_assignment_creates_voices_and_validates_overrides(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    assignment = draft_assignment(_report(), lib, default_preset="af_heart")
    assert assignment.narrator_voice_id == "narrator_af_heart"
    assert set(assignment.assignments) == {"alice", "al"}
    assert lib.meta_path("alice_auto").is_file()  # draft blend voices exist

    with pytest.raises(ServiceError, match="unknown character"):
        draft_assignment(
            _report(), lib, default_preset="af_heart", overrides={"ghost": "alice_auto"}
        )
    with pytest.raises(ServiceError, match="not in the library"):
        draft_assignment(_report(), lib, default_preset="af_heart", overrides={"alice": "nope"})
    with pytest.raises(ServiceError, match="narrator voice"):
        draft_assignment(_report(), lib, default_preset="af_heart", narrator_voice_id="missing")

    path = save_assignment(assignment, tmp_path / "output")
    assert path.is_file()
    reloaded = VoiceAssignment.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded == assignment


# --- voice deletion guard ---


def test_delete_voice_refuses_while_referenced_then_deletes(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(voice_id="v1", name="V", kind=VoiceKind.PRESET,
                  engine="kokoro", preset_id="af_heart")
    )  # fmt: skip
    assignment = VoiceAssignment(
        book_id="book-a", narrator_voice_id="v1", assignments={"alice": "v1"}
    )
    save_assignment(assignment, tmp_path / "output")

    refs = voice_references("v1", tmp_path / "output")
    assert {(r.book_id, r.role) for r in refs} == {
        ("book-a", "narrator"),
        ("book-a", "character:alice"),
    }
    with pytest.raises(ServiceError, match="still assigned in: book-a"):
        delete_voice("v1", library=lib, output_dir=tmp_path / "output")

    # reassign the book away from v1 -> deletion proceeds
    save_assignment(
        VoiceAssignment(book_id="book-a", narrator_voice_id="other", assignments={}),
        tmp_path / "output",
    )
    delete_voice("v1", library=lib, output_dir=tmp_path / "output")
    assert not lib.dir_for("v1").exists()

    with pytest.raises(ServiceError, match="not found"):
        delete_voice("v1", library=lib, output_dir=tmp_path / "output")


def test_delete_voice_refuses_case_variant_ids(tmp_path):
    """NTFS resolves 'V1' to voices/v1, but the reference scan is string-based — a
    case-variant id must refuse, not sail past the guard and delete a referenced voice."""
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(voice_id="v1", name="V", kind=VoiceKind.PRESET,
                  engine="kokoro", preset_id="af_heart")
    )  # fmt: skip
    save_assignment(
        VoiceAssignment(book_id="book-a", narrator_voice_id="v1", assignments={}),
        tmp_path / "output",
    )
    with pytest.raises(ServiceError):
        delete_voice("V1", library=lib, output_dir=tmp_path / "output")
    assert lib.dir_for("v1").exists()


def test_delete_voice_fails_closed_on_unreadable_assignment(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(voice_id="v1", name="V", kind=VoiceKind.PRESET,
                  engine="kokoro", preset_id="af_heart")
    )  # fmt: skip
    broken = tmp_path / "output" / "book-x"
    broken.mkdir(parents=True)
    (broken / "assignments.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ServiceError, match="cannot verify"):
        delete_voice("v1", library=lib, output_dir=tmp_path / "output")
    assert lib.dir_for("v1").exists()  # nothing deleted while the guard can't see


def test_delete_voice_refuses_path_traversal(tmp_path):
    """M6b passes voice_id from an HTTP client — rmtree must be contained to the library."""
    lib = VoiceLibrary(tmp_path / "voices")
    victim = tmp_path / "books"
    (victim / "x").mkdir(parents=True)
    (victim / "meta.json").write_text("{}", encoding="utf-8")  # satisfies the exists check
    with pytest.raises(ServiceError, match="invalid voice id"):
        delete_voice("../books", library=lib, output_dir=tmp_path / "output")
    assert victim.exists()
