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
