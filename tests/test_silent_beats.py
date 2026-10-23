"""Lines with nothing to pronounce ("───", "“……”", "♪♪", an emoji) render as a short silent beat
instead of reaching an engine (Kokoro returned no audio for "───" and failed a whole render), and
are never whisper-validated. Ingest also treats box-drawing / bar / ornament lines as scene breaks.
"""

import numpy as np
import pytest
import soundfile as sf

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
from seiyuu.engines import nothing_to_say
from seiyuu.engines.base import SILENT_BEAT_SECONDS, SynthesisError
from seiyuu.gpu import GpuResourceManager
from seiyuu.ingest.common import DECORATIVE_PATTERN
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.render import estimate_render_cost_single, render_book, render_book_multivoice
from seiyuu.validate import ValidationResult
from seiyuu.voices import VoiceAssignment, VoiceLibrary, VoiceMeta
from seiyuu.voices.models import VoiceKind


class CountingValidator:
    def __init__(self) -> None:
        self.expected: list[str] = []

    def validate(self, wav_path, expected_text):
        self.expected.append(expected_text)
        return ValidationResult(ok=True, score=1.0, transcript="heard", expected=expected_text)


@pytest.mark.parametrize("text", ["───", "“……”", "……!", "+++", "♩♪♬", "🥰🤯", "[ ]", "!"])
def test_nothing_to_say(text):
    assert nothing_to_say(text)


@pytest.mark.parametrize("text", ["Hi.", "“……Hm?”", "3", "한국어", "Ⅻ", "½"])
def test_something_to_say(text):
    assert not nothing_to_say(text)


def test_engine_returns_silence_without_calling_the_model():
    eng = FakeEngine(fail_on="─")  # would raise if the native path ran
    audio = eng.synthesize("───", "test_voice")
    assert eng.calls == []
    assert audio.duration_seconds == pytest.approx(SILENT_BEAT_SECONDS)
    assert not np.any(audio.samples)


def test_empty_text_still_refused():
    with pytest.raises(SynthesisError, match="empty text"):
        FakeEngine().synthesize("   ", "test_voice")


def _book(*texts: str) -> NormalizedBook:
    return NormalizedBook(
        book_meta=BookMeta(
            book_id="beat-book-0000000", title="B", source_path="b.epub", source_sha256="0" * 64
        ),
        chapters=[
            Chapter(
                title="C1",
                blocks=[
                    Block(id=f"ch001_b{i:04d}", type=BlockType.PARAGRAPH, text=t)
                    for i, t in enumerate(texts, start=1)
                ],
            )
        ],
    )


def test_single_voice_render_silences_the_line_and_skips_validation(tmp_path):
    eng = FakeEngine()
    eng.requires_validation = True  # Chatterbox-like: would otherwise whisper-check every line
    v = CountingValidator()
    result = render_book(
        _book("Before.", "───", "After."), eng, "v", tmp_path / "out", seed=1, validator=v
    )

    segs = result.manifest.chapters[0].segments
    assert [s.block_id for s in segs] == ["ch001_b0001", "ch001_b0002", "ch001_b0003"]
    beat = segs[1]
    assert beat.validation is None
    assert beat.duration_seconds == pytest.approx(SILENT_BEAT_SECONDS)
    assert [t for t, _ in eng.calls] == ["Before.", "After."]
    assert v.expected == ["Before.", "After."]  # the silent line was never checked


def _library(tmp_path) -> VoiceLibrary:
    lib = VoiceLibrary(tmp_path / "voices")
    for vid, preset in (("narrator_v", "af_heart"), ("alice_v", "af_bella")):
        lib.save(
            VoiceMeta(
                voice_id=vid, name=vid, kind=VoiceKind.PRESET, engine="kokoro", preset_id=preset
            )
        )
    return lib


def test_multivoice_render_silences_a_wordless_quote(tmp_path, monkeypatch):
    eng = FakeEngine()
    eng.requires_validation = True
    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda engine_id, **kw: eng)
    book = make_book()
    book.chapters[0].blocks[1] = Block(id="ch001_b0002", type=BlockType.PARAGRAPH, text="“……”")
    report = AttributionReport(
        book_id="test-book-00000000",
        provider_id="local",
        model_id="m",
        prompt_version="v5",
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
                        text="“……”",
                        speaker="alice",
                    ),
                    Segment(
                        block_id="ch001_b0004", type=SegmentType.NARRATION, text="After the break."
                    ),
                ],
            ),
        ],
    )
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    v = CountingValidator()
    result = render_book_multivoice(
        report,
        book,
        _library(tmp_path),
        assignment,
        tmp_path / "out",
        chapters=(1,),
        gpu=GpuResourceManager(),
        validator=v,
    )

    segs = {s.block_id: s for s in result.manifest.chapters[0].segments}
    beat = segs["ch001_b0002"]
    assert beat.voice_id == "alice_v"
    assert beat.validation is None
    assert beat.duration_seconds == pytest.approx(SILENT_BEAT_SECONDS)
    # only the heading and the narration reached the engine and the validator
    assert len(eng.calls) == 2
    assert len(v.expected) == 2


@pytest.mark.parametrize("text", ["───", "+++", "‒‒‒", "―――", "⸻", "◇◇◇", "⁂", "* * *", "☆ ☆ ☆"])
def test_decorative_lines_are_scene_breaks(text):
    assert DECORATIVE_PATTERN.match(text)


@pytest.mark.parametrize("text", ["……", "“……”", "-♩♪♬", "2+2", "Hi ───"])
def test_wordless_dialogue_is_not_a_scene_break(text):
    assert not DECORATIVE_PATTERN.match(text)


def test_stale_engine_audio_for_a_wordless_line_is_replaced(tmp_path):
    """An older render cached real engine audio for the line (maybe invented speech), with a
    verdict and read-along words: the next render, without force, rewrites the silent beat and
    drops both sidecars instead of serving the old audio."""
    eng = FakeEngine()
    eng.requires_validation = True
    out = tmp_path / "out"
    first = render_book(_book("───"), eng, "v", out, seed=1, validator=CountingValidator())
    wav = out / first.manifest.chapters[0].segments[0].wav
    t = np.arange(48_000) / 24_000
    sf.write(str(wav), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24_000)
    verdict = wav.with_name(f"{wav.stem}.validation.json")
    words = wav.with_name(f"{wav.stem}.words.json")
    verdict.write_text(
        ValidationResult(
            ok=False, score=0.0, transcript="hmm I see", expected="───"
        ).model_dump_json(),
        encoding="utf-8",
    )
    words.write_text("{}", encoding="utf-8")

    again = render_book(_book("───"), eng, "v", out, seed=1, validator=CountingValidator())

    seg = again.manifest.chapters[0].segments[0]
    assert seg.validation is None
    assert sf.info(str(wav)).duration == pytest.approx(SILENT_BEAT_SECONDS)
    assert not verdict.exists() and not words.exists()
    assert again.manifest.validation_failures == 0


class PaidEngine(FakeEngine):
    engine_id = "paid"
    uses_gpu = False

    def cost_estimate(self, text: str) -> float:
        return len(text) * 0.001


def test_wordless_lines_are_free_to_quote_and_need_no_approval(tmp_path):
    eng = PaidEngine()
    est = estimate_render_cost_single(_book("“……”", "───", "Hello."), eng, "v", tmp_path / "out")
    assert est.paid_segments == 1  # only "Hello."
    assert est.free_segments == 2
    # a render of only wordless lines bills nothing, so it needs no cost confirmation
    result = render_book(_book("“……”", "───"), eng, "v", tmp_path / "out", seed=1)
    assert len(result.manifest.chapters[0].segments) == 2
    assert eng.calls == []
