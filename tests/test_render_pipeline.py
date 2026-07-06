import pytest
import soundfile as sf

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.ingest.models import BlockType
from seiyuu.render import MANIFEST_NAME, RenderError, RenderManifest, render_book


def test_render_full_book(tmp_path) -> None:
    engine = FakeEngine()
    result = render_book(make_book(), engine, "test_voice", tmp_path / "book")

    # 5 speakable blocks synthesized, scene break passed through without audio
    assert result.synthesized == 5
    assert result.cache_hits == 0
    assert len(engine.calls) == 5
    assert result.total_audio_seconds > 0

    manifest = result.manifest
    assert [c.index for c in manifest.chapters] == [1, 2]
    ch1 = manifest.chapters[0]
    assert [s.type for s in ch1.segments] == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.SCENE_BREAK,
        BlockType.PARAGRAPH,
    ]
    scene = ch1.segments[2]
    assert scene.wav is None and scene.duration_seconds == 0.0

    # all wavs exist, are canonical, and paths are relative to the book dir
    for chapter in manifest.chapters:
        for seg in chapter.segments:
            if seg.wav is not None:
                info = sf.info(str(tmp_path / "book" / seg.wav))
                assert info.samplerate == 24_000
                assert info.channels == 1
                assert info.subtype == "PCM_16"


def test_scene_breaks_never_synthesized(tmp_path) -> None:
    engine = FakeEngine()
    render_book(make_book(), engine, "test_voice", tmp_path / "book")
    assert len(engine.calls) == 5  # exactly the speakable blocks, nothing more


def test_rerender_hits_cache(tmp_path) -> None:
    first = FakeEngine()
    render_book(make_book(), first, "test_voice", tmp_path / "book")
    second = FakeEngine()
    result = render_book(make_book(), second, "test_voice", tmp_path / "book")
    assert len(second.calls) == 0
    assert result.cache_hits == 5
    assert result.synthesized == 0


def test_changed_seed_misses_cache(tmp_path) -> None:
    render_book(make_book(), FakeEngine(), "test_voice", tmp_path / "book", seed=1)
    second = FakeEngine()
    render_book(make_book(), second, "test_voice", tmp_path / "book", seed=2)
    assert len(second.calls) == 5


def test_chapter_subset(tmp_path) -> None:
    engine = FakeEngine()
    result = render_book(make_book(), engine, "test_voice", tmp_path / "book", chapters=(2,))
    assert [c.index for c in result.manifest.chapters] == [2]
    assert len(engine.calls) == 2


def test_unknown_chapter_rejected(tmp_path) -> None:
    with pytest.raises(RenderError, match="no such chapter"):
        render_book(make_book(), FakeEngine(), "test_voice", tmp_path / "book", chapters=(99,))


def test_failure_names_book_chapter_block(tmp_path) -> None:
    engine = FakeEngine(fail_on="After the break")
    with pytest.raises(RenderError) as err:
        render_book(make_book(), engine, "test_voice", tmp_path / "book")
    message = str(err.value)
    assert "test-book-00000000" in message
    assert "chapter=1" in message
    assert "ch001_b0004" in message


def test_manifest_round_trip(tmp_path) -> None:
    result = render_book(make_book(), FakeEngine(), "test_voice", tmp_path / "book")
    loaded = RenderManifest.model_validate_json(
        (tmp_path / "book" / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert loaded == result.manifest


class _RecordingEngine(FakeEngine):
    """FakeEngine that also records the settings dict handed to each synth call."""

    def __init__(self) -> None:
        super().__init__()
        self.settings_seen: list[tuple[str, dict]] = []

    def _synthesize_native(self, text, voice, settings):
        self.settings_seen.append((voice, dict(settings)))
        return super()._synthesize_native(text, voice, settings)


def test_single_voice_saved_kokoro_preset_renders(tmp_path) -> None:
    # A saved Kokoro PRESET whose library voice_id differs from its preset_id used to crash the
    # single-voice path (it passed voice_id verbatim as the engine voice). render_book now
    # resolves the engine voice via render_voice_args, so the engine gets the preset_id. (#4)
    from seiyuu.voices import VoiceLibrary, VoiceMeta
    from seiyuu.voices.models import VoiceKind

    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(
            voice_id="narrator_v",
            name="N",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="af_heart",
            seed=777,
        )  # fmt: skip
    )
    engine = _RecordingEngine()
    result = render_book(make_book(), engine, "narrator_v", tmp_path / "book", library=lib)

    # engine addressed by the preset id, NOT the library voice_id (the crash the fix removes)
    assert {v for v, _ in engine.settings_seen} == {"af_heart"}
    # the FROZEN SegmentKey identity stays the library voice_id, and the pinned seed is honored
    seg = next(s for s in result.manifest.chapters[0].segments if s.wav is not None)
    assert seg.voice_id == "narrator_v" and seg.seed == 777
    assert result.manifest.voice_id == "narrator_v" and result.manifest.seed == 777


def test_single_voice_saved_kokoro_blend_renders(tmp_path) -> None:
    # A saved Kokoro BLEND must fold its canonical recipe into settings (the engine builds the
    # weighted voicepack from it); passing voice_id verbatim with no recipe crashed it. (#4)
    from seiyuu.voices import VoiceLibrary, VoiceMeta
    from seiyuu.voices.models import BlendComponent, VoiceKind

    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(
            voice_id="liz_v",
            name="L",
            kind=VoiceKind.BLEND,
            engine="kokoro",
            blend=[
                BlendComponent(preset_id="af_bella", weight=1),
                BlendComponent(preset_id="af_sky", weight=1),
            ],
        )  # fmt: skip
    )
    engine = _RecordingEngine()
    result = render_book(make_book(), engine, "liz_v", tmp_path / "book", library=lib)

    voice, settings = engine.settings_seen[0]
    assert voice == "liz_v"  # blend addressed by voice_id; recipe rides in settings
    assert settings["blend"] == [["af_bella", 0.5], ["af_sky", 0.5]]
    seg = next(s for s in result.manifest.chapters[0].segments if s.wav is not None)
    assert seg.voice_id == "liz_v"


@pytest.mark.gpu
def test_render_with_real_kokoro(tmp_path) -> None:
    from seiyuu.engines import get_engine

    book = make_book()
    result = render_book(book, get_engine("kokoro"), "af_heart", tmp_path / "book", seed=41172)
    assert result.synthesized == 5
    assert result.total_audio_seconds > 3
    for seg in result.manifest.chapters[0].segments:
        if seg.wav is not None:
            assert seg.duration_seconds > 0.2
