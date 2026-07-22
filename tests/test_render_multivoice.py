"""Multi-voice render: per-segment voice resolution, scene-break passthrough, manifest shape."""

import pytest

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    Segment,
    SegmentType,
)
from seiyuu.gpu import GpuResourceManager
from seiyuu.ingest.models import Block, BlockType, Chapter
from seiyuu.render import RenderError, render_book_multivoice
from seiyuu.voices import VoiceAssignment, VoiceLibrary, VoiceMeta
from seiyuu.voices.models import VoiceKind


def _report() -> AttributionReport:
    # Matches factories.make_book: ch1 (heading, paragraph, scene_break, paragraph), ch2.
    return AttributionReport(
        book_id="test-book-00000000",
        provider_id="local",
        model_id="qwen2.5:7b",
        prompt_version="v3",
        registry=CharacterRegistry(characters=[Character(id="alice", canonical_name="Alice")]),
        chapters=[
            AttributedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    Segment(block_id="ch001_b0001", type=SegmentType.NARRATION, text="Chapter 1"),
                    Segment(
                        block_id="ch001_b0002",
                        type=SegmentType.DIALOGUE,
                        text="Hello world.",
                        speaker="alice",
                    ),
                    Segment(
                        block_id="ch001_b0004", type=SegmentType.NARRATION, text="After the break."
                    ),
                ],
            ),
            AttributedChapter(
                index=2,
                title="Chapter 2",
                segments=[
                    Segment(block_id="ch002_b0001", type=SegmentType.NARRATION, text="Chapter 2"),
                    Segment(
                        block_id="ch002_b0002",
                        type=SegmentType.DIALOGUE,
                        text="Second chapter.",
                        speaker="alice",
                    ),
                ],
            ),
        ],
    )


def _library(tmp_path) -> VoiceLibrary:
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(
            voice_id="narrator_v",
            name="Narrator",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="af_heart",
        )
    )
    lib.save(
        VoiceMeta(
            voice_id="alice_v",
            name="Alice",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="af_bella",
        )
    )
    return lib


def _patch_engine(monkeypatch, engine):
    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda engine_id, **kw: engine)


def test_multivoice_resolves_voices_and_passes_scene_break(tmp_path, monkeypatch):
    fake = FakeEngine()
    _patch_engine(monkeypatch, fake)
    lib = _library(tmp_path)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )

    result = render_book_multivoice(
        _report(), make_book(), lib, assignment, tmp_path / "out", gpu=GpuResourceManager()
    )

    ch1 = result.manifest.chapters[0]
    # reading order: heading(narrator), paragraph dialogue(alice), scene_break, paragraph(narrator)
    assert [(s.block_id, s.type.value, s.voice_id) for s in ch1.segments] == [
        ("ch001_b0001", "heading", "narrator_v"),
        ("ch001_b0002", "paragraph", "alice_v"),
        ("ch001_b0003", "scene_break", None),
        ("ch001_b0004", "paragraph", "narrator_v"),
    ]
    assert set(result.manifest.voices_used) == {"narrator_v", "alice_v"}
    assert result.manifest.assignment["narrator_voice_id"] == "narrator_v"
    assert result.manifest.engine is None  # multi-voice: manifest-level voice fields unset
    # the engine is handed the PRESET name, not the library voice_id (the regression a live
    # Kokoro smoke render caught: 'unknown voice narrator_v')
    sent_voices = {v for _, v in fake.calls}
    assert sent_voices == {"af_heart", "af_bella"}


def test_multivoice_manifest_round_trips(tmp_path, monkeypatch):
    _patch_engine(monkeypatch, FakeEngine())
    lib = _library(tmp_path)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    result = render_book_multivoice(
        _report(), make_book(), lib, assignment, tmp_path / "out", gpu=GpuResourceManager()
    )
    from seiyuu.render import RenderManifest

    reloaded = RenderManifest.model_validate_json(result.manifest_path.read_text(encoding="utf-8"))
    assert reloaded.voices_used["alice_v"].engine == "kokoro"


def test_cloned_without_consent_is_refused(tmp_path, monkeypatch):
    _patch_engine(monkeypatch, FakeEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    # write a cloned voice meta.json directly with consent False (bypassing the save() gate)
    d = lib.dir_for("alice_v")
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        VoiceMeta(
            voice_id="alice_v",
            name="Alice",
            kind=VoiceKind.CLONED,
            engine="chatterbox",
            reference_audio="reference.wav",
            consent_attested=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    # flip consent to False on disk
    import json

    p = d / "meta.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["consent_attested"] = False
    p.write_text(json.dumps(data), encoding="utf-8")
    lib.save(
        VoiceMeta(
            voice_id="narrator_v",
            name="N",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="af_heart",
        )
    )
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    with pytest.raises(RenderError, match="consent"):
        render_book_multivoice(
            _report(), make_book(), lib, assignment, tmp_path / "out", gpu=GpuResourceManager()
        )


def test_render_voice_args_addresses_each_kind():
    from seiyuu.voices import VoiceMeta, render_voice_args
    from seiyuu.voices.models import BlendComponent

    preset = VoiceMeta(
        voice_id="narr_x", name="N", kind=VoiceKind.PRESET, engine="kokoro", preset_id="af_heart"
    )
    voice, settings = render_voice_args(preset)
    assert voice == "af_heart" and "blend" not in settings  # preset addressed by preset_id

    blend = VoiceMeta(
        voice_id="liz_x",
        name="L",
        kind=VoiceKind.BLEND,
        engine="kokoro",
        blend=[
            BlendComponent(preset_id="af_bella", weight=1),
            BlendComponent(preset_id="af_sky", weight=1),
        ],
    )
    voice, settings = render_voice_args(blend)
    assert voice == "liz_x"  # blend addressed by voice_id; recipe rides in settings
    assert settings["blend"] == [["af_bella", 0.5], ["af_sky", 0.5]]

    cloned = VoiceMeta(
        voice_id="bob_x",
        name="B",
        kind=VoiceKind.CLONED,
        engine="chatterbox",
        reference_audio="reference.wav",
        consent_attested=True,
    )
    voice, settings = render_voice_args(cloned)
    assert voice == "bob_x" and "blend" not in settings  # cloned addressed by voice_id (conds)


def test_multivoice_subset_renders_merge_into_manifest(tmp_path, monkeypatch):
    # Regression: a chapter-subset render clobbered manifest.json wholesale — after
    # rendering ch1 then ch2, only ch2 survived for the summary/Listen/assembly.
    _patch_engine(monkeypatch, FakeEngine())
    lib = _library(tmp_path)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    out = tmp_path / "out"
    render_book_multivoice(
        _report(), make_book(), lib, assignment, out, chapters=(1,), gpu=GpuResourceManager()
    )
    # a re-saved assignment with the SAME voice map but churned metadata must still merge
    resumed = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
        stage="final",
        created_at="2020-01-01",
    )
    result = render_book_multivoice(
        _report(), make_book(), lib, resumed, out, chapters=(2,), gpu=GpuResourceManager()
    )

    assert [c.index for c in result.manifest.chapters] == [1, 2]
    assert set(result.manifest.voices_used) == {"narrator_v", "alice_v"}
    from seiyuu.render import RenderManifest

    reloaded = RenderManifest.model_validate_json(result.manifest_path.read_text(encoding="utf-8"))
    assert reloaded == result.manifest
    # carried-over chapter 1 entries intact, including the scene break
    assert [s.block_id for s in reloaded.chapters[0].segments] == [
        "ch001_b0001",
        "ch001_b0002",
        "ch001_b0003",
        "ch001_b0004",
    ]


def test_multivoice_subset_merge_drops_ghosts_and_their_voices(tmp_path, monkeypatch):
    # Re-uploading the same file with different split settings keeps the book_id but can
    # SHRINK the chapter set (3 → 2 here). The old manifest's ch3 is a ghost: it must not
    # survive a subset merge, and a voice ONLY it used must leave voices_used.
    _patch_engine(monkeypatch, FakeEngine())
    lib = _library(tmp_path)
    lib.save(
        VoiceMeta(
            voice_id="bob_v",
            name="Bob",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="am_adam",
        )  # fmt: skip
    )
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v", "bob": "bob_v"},
    )
    big_book = make_book()
    big_book.chapters.append(
        Chapter(
            title="Chapter 3",
            blocks=[Block(id="ch003_b0001", type=BlockType.PARAGRAPH, text="Bob speaking.")],
        )
    )
    big_report = _report()
    big_report.registry.characters.append(Character(id="bob", canonical_name="Bob"))
    big_report.chapters.append(
        AttributedChapter(
            index=3,
            title="Chapter 3",
            segments=[
                Segment(
                    block_id="ch003_b0001",
                    type=SegmentType.DIALOGUE,
                    text="Bob speaking.",
                    speaker="bob",
                )
            ],
        )
    )
    out = tmp_path / "out"
    before = render_book_multivoice(
        big_report, big_book, lib, assignment, out, gpu=GpuResourceManager()
    )
    assert set(before.manifest.voices_used) == {"narrator_v", "alice_v", "bob_v"}

    # the book shrank to 2 chapters (same book_id, same assignment) — subset render ch2
    result = render_book_multivoice(
        _report(), make_book(), lib, assignment, out, chapters=(2,), gpu=GpuResourceManager()
    )

    assert [c.index for c in result.manifest.chapters] == [1, 2]  # ghost ch3 dropped
    assert set(result.manifest.voices_used) == {"narrator_v", "alice_v"}  # bob_v was ghost-only
    from seiyuu.render import RenderManifest

    reloaded = RenderManifest.model_validate_json(result.manifest_path.read_text(encoding="utf-8"))
    assert reloaded == result.manifest


def test_multivoice_subset_assignment_mismatch_refused_before_synthesis(tmp_path, monkeypatch):
    # Chapters outside the subset were rendered with the OLD voice map; merging under a new
    # one would misdescribe them — refuse before any synthesis starts.
    fake = FakeEngine()
    _patch_engine(monkeypatch, fake)
    lib = _library(tmp_path)
    out = tmp_path / "out"
    original = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    render_book_multivoice(
        _report(), make_book(), lib, original, out, chapters=(1,), gpu=GpuResourceManager()
    )
    calls_before = len(fake.calls)
    recast = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "narrator_v"},
    )
    with pytest.raises(RenderError, match="assignment"):
        render_book_multivoice(
            _report(), make_book(), lib, recast, out, chapters=(2,), gpu=GpuResourceManager()
        )
    assert len(fake.calls) == calls_before  # refused up front


def test_subset_cross_mode_starts_fresh_and_preserves_the_other_mode(tmp_path, monkeypatch):
    """Pre-feature book: manifest.json is single-voice and NO mode archives exist. A
    multivoice chapter-subset render must NOT refuse (the merge base is per-mode now): it
    starts a fresh multi archive, promotes it to the active manifest.json, and the single
    render survives byte-for-byte as the single archive — the mode-switch fallback."""
    from seiyuu.render import MANIFEST_NAME, manifest_name_for_mode, render_book

    _patch_engine(monkeypatch, FakeEngine())
    lib = _library(tmp_path)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    out = tmp_path / "out"
    render_book(make_book(), FakeEngine(), "test_voice", out)
    (out / manifest_name_for_mode("single")).unlink()  # pre-feature book: pointer only
    single_raw = (out / MANIFEST_NAME).read_text(encoding="utf-8")

    result = render_book_multivoice(
        _report(), make_book(), lib, assignment, out, chapters=(2,), gpu=GpuResourceManager()
    )

    # fresh multi archive holding only the subset, promoted to the active manifest
    assert [c.index for c in result.manifest.chapters] == [2]
    multi_raw = (out / manifest_name_for_mode("multi")).read_text(encoding="utf-8")
    assert (out / MANIFEST_NAME).read_text(encoding="utf-8") == multi_raw
    # the single render was preserved untouched as the single archive
    assert (out / manifest_name_for_mode("single")).read_text(encoding="utf-8") == single_raw


def test_segments_cached_on_second_run(tmp_path, monkeypatch):
    fake = FakeEngine()
    _patch_engine(monkeypatch, fake)
    lib = _library(tmp_path)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    args = (_report(), make_book(), lib, assignment, tmp_path / "out")
    first = render_book_multivoice(*args, gpu=GpuResourceManager())
    second = render_book_multivoice(*args, gpu=GpuResourceManager())
    assert first.synthesized > 0 and second.synthesized == 0 and second.cache_hits > 0


def test_multivoice_groups_synthesis_by_engine(tmp_path, monkeypatch):
    # Alternating narrator/dialogue across two engines must synthesize grouped per
    # engine: ONE unload per engine per render (eviction at the group switch + the
    # final free_all), not one at every voice alternation — while the manifest keeps
    # strict reading order.
    class TrackingEngine(FakeEngine):
        def __init__(self, engine_id: str) -> None:
            super().__init__()
            self.engine_id = engine_id
            self.unloads = 0

        def unload(self) -> None:
            self.unloads += 1

    eng_a, eng_b = TrackingEngine("fake_a"), TrackingEngine("fake_b")
    monkeypatch.setattr(
        "seiyuu.render.pipeline.get_engine",
        lambda engine_id, **kw: {"fake_a": eng_a, "fake_b": eng_b}[engine_id],
    )
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(
            voice_id="narrator_v",
            name="N",
            kind=VoiceKind.PRESET,
            engine="fake_a",
            preset_id="voice_n",
        )  # fmt: skip
    )
    lib.save(
        VoiceMeta(
            voice_id="alice_v",
            name="A",
            kind=VoiceKind.PRESET,
            engine="fake_b",
            preset_id="voice_a",
        )  # fmt: skip
    )

    book = make_book()
    book = book.model_copy(
        update={
            "chapters": [
                Chapter(
                    title="Chapter 1",
                    blocks=[
                        Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="Narr one."),
                        Block(id="ch001_b0002", type=BlockType.PARAGRAPH, text="Line one."),
                        Block(id="ch001_b0003", type=BlockType.PARAGRAPH, text="Narr two."),
                        Block(id="ch001_b0004", type=BlockType.PARAGRAPH, text="Line two."),
                    ],
                )
            ]
        }
    )
    report = AttributionReport(
        book_id="test-book-00000000",
        provider_id="local",
        model_id="m",
        prompt_version="v3",
        registry=CharacterRegistry(characters=[Character(id="alice", canonical_name="Alice")]),
        chapters=[
            AttributedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    Segment(block_id="ch001_b0001", type=SegmentType.NARRATION, text="Narr one."),
                    Segment(
                        block_id="ch001_b0002",
                        type=SegmentType.DIALOGUE,
                        text="Line one.",
                        speaker="alice",
                    ),  # fmt: skip
                    Segment(block_id="ch001_b0003", type=SegmentType.NARRATION, text="Narr two."),
                    Segment(
                        block_id="ch001_b0004",
                        type=SegmentType.DIALOGUE,
                        text="Line two.",
                        speaker="alice",
                    ),  # fmt: skip
                ],
            )
        ],
    )
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )

    result = render_book_multivoice(
        report, book, lib, assignment, tmp_path / "out", gpu=GpuResourceManager()
    )

    # grouped synthesis order per engine...
    assert [t for t, _ in eng_a.calls] == ["Narr one.", "Narr two."]
    assert [t for t, _ in eng_b.calls] == ["Line one.", "Line two."]
    # ...one eviction of A at the group switch, one unload of B at the final free_all
    assert eng_a.unloads == 1 and eng_b.unloads == 1
    # ...and the manifest still reads in strict reading order
    rows = result.manifest.chapters[0].segments
    assert [(s.block_id, s.voice_id) for s in rows] == [
        ("ch001_b0001", "narrator_v"),
        ("ch001_b0002", "alice_v"),
        ("ch001_b0003", "narrator_v"),
        ("ch001_b0004", "alice_v"),
    ]
    assert result.synthesized == 4


def test_multivoice_uses_engine_provider_when_given(tmp_path, monkeypatch):
    # Server path: engines come from the injected provider (the process-lifetime
    # registry), so a model warmed by a warmup job or audition is reused instead of a
    # fresh instance evicting it and cold-loading the same weights.
    provided = FakeEngine()
    asked: list[str] = []

    def provider(engine_id: str):
        asked.append(engine_id)
        return provided

    monkeypatch.setattr(
        "seiyuu.render.pipeline.get_engine",
        lambda *a, **kw: pytest.fail("constructed a fresh engine despite engine_provider"),
    )
    lib = _library(tmp_path)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    result = render_book_multivoice(
        _report(), make_book(), lib, assignment, tmp_path / "out",
        gpu=GpuResourceManager(), engine_provider=provider,
    )  # fmt: skip
    assert asked == ["kokoro"]  # consulted once per engine id, then memoized
    assert result.synthesized > 0 and provided.calls
