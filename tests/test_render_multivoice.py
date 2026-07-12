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


def test_subset_mode_mismatch_refused_both_directions(tmp_path, monkeypatch):
    from seiyuu.render import render_book

    fake = FakeEngine()
    _patch_engine(monkeypatch, fake)
    lib = _library(tmp_path)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )

    # single-voice manifest exists → multivoice subset render refuses, naming both modes
    single_out = tmp_path / "single_out"
    render_book(make_book(), FakeEngine(), "test_voice", single_out)
    with pytest.raises(RenderError, match="multivoice.*single-voice"):
        render_book_multivoice(
            _report(), make_book(), lib, assignment, single_out,
            chapters=(2,), gpu=GpuResourceManager(),
        )  # fmt: skip
    assert fake.calls == []  # refused before any synthesis

    # multivoice manifest exists → single-voice subset render refuses
    multi_out = tmp_path / "multi_out"
    render_book_multivoice(
        _report(), make_book(), lib, assignment, multi_out, gpu=GpuResourceManager()
    )
    single_engine = FakeEngine()
    with pytest.raises(RenderError, match="single-voice.*multivoice"):
        render_book(make_book(), single_engine, "test_voice", multi_out, chapters=(2,))
    assert single_engine.calls == []


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
