"""Lines with nothing to pronounce ("───", "“……”", "♪♪", an emoji) render as a short silent beat
instead of reaching an engine (Kokoro returned no audio for "───" and failed a whole render), and
are never whisper-validated. Ingest also treats box-drawing / bar / ornament lines as scene breaks.
"""

import numpy as np
import pytest
import soundfile as sf

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.assemble import PauseProfile
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


def test_beat_is_shorter_than_the_gaps_around_it():
    # The assembler's gaps (and the neighbours' clip edges) already surround a beat; at 0.6 s
    # the whole beat ran as long as a scene break. Keep the beat itself under every join gap.
    pauses = PauseProfile()
    assert SILENT_BEAT_SECONDS < min(pauses.dialogue, pauses.paragraph)


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
    """Whatever sits at the beat's file (here engine audio, maybe invented speech, with a
    verdict and read-along words): the next render, without force, rewrites the silent beat and
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


def _two_chapter_book() -> NormalizedBook:
    def chapter(n: int, texts: tuple[str, ...]) -> Chapter:
        return Chapter(
            title=f"C{n}",
            blocks=[
                Block(id=f"ch{n:03d}_b{i:04d}", type=BlockType.PARAGRAPH, text=t)
                for i, t in enumerate(texts, start=1)
            ],
        )

    return NormalizedBook(
        book_meta=BookMeta(
            book_id="beat-book-0000000", title="B", source_path="b.epub", source_sha256="0" * 64
        ),
        chapters=[chapter(1, ("Before.", "“……”", "After.")), chapter(2, ("Hi.", "“……”", "Bye."))],
    )


def test_new_beat_length_never_rewrites_a_beat_other_chapters_point_to(tmp_path, monkeypatch):
    """One wordless line has one key, so every chapter with it shares the cache entry. A scoped
    render after the beat length changes must not rewrite the file the other chapters'
    manifests point to, or their recorded durations (player offsets, totals) go stale."""
    book, eng, out = _two_chapter_book(), FakeEngine(), tmp_path / "out"
    monkeypatch.setattr("seiyuu.render.pipeline.SILENT_BEAT_SECONDS", 0.6)  # an older length
    render_book(book, eng, "v", out, seed=1)
    monkeypatch.undo()

    result = render_book(book, eng, "v", out, seed=1, chapters=(2,))

    beats = [ch.segments[1] for ch in result.manifest.chapters]
    for beat in beats:
        assert beat.duration_seconds == pytest.approx(sf.info(str(out / beat.wav)).duration)
    assert [b.duration_seconds for b in beats] == [0.6, pytest.approx(SILENT_BEAT_SECONDS)]
    assert beats[0].wav != beats[1].wav


def test_older_audio_under_the_key_is_left_alone_and_never_served(tmp_path):
    """Before the line was a beat, a render may have cached real engine audio (maybe invented
    speech) at {key_hash}.wav. The beat is written beside it; the old file and its verdict
    stay as the manifests that point at them describe them."""
    eng, out = FakeEngine(), tmp_path / "out"
    first = render_book(_book("───"), eng, "v", out, seed=1)
    beat_wav = out / first.manifest.chapters[0].segments[0].wav
    legacy = beat_wav.with_name(f"{beat_wav.name.split('.')[0]}.wav")
    assert legacy != beat_wav
    sf.write(str(legacy), np.full(48_000, 0.3, dtype=np.float32), 24_000)
    verdict = legacy.with_name(f"{legacy.stem}.validation.json")
    verdict.write_text("{}", encoding="utf-8")

    again = render_book(_book("───"), eng, "v", out, seed=1)

    seg = again.manifest.chapters[0].segments[0]
    assert out / seg.wav == beat_wav
    assert sf.info(str(beat_wav)).duration == pytest.approx(SILENT_BEAT_SECONDS)
    assert sf.info(str(legacy)).duration == pytest.approx(2.0) and verdict.exists()


def test_estimate_counts_a_beat_as_recorded_only_at_the_current_length(tmp_path, monkeypatch):
    eng, out = FakeEngine(), tmp_path / "out"
    render_book(_book("“……”"), eng, "v", out, seed=1)
    est = estimate_render_cost_single(_book("“……”"), eng, "v", out, seed=1)
    assert (est.cached_segments, est.free_segments) == (1, 0)
    # a new beat length is a new file the next render writes: no longer recorded
    monkeypatch.setattr("seiyuu.render.pipeline.SILENT_BEAT_SECONDS", 0.6)
    est = estimate_render_cost_single(_book("“……”"), eng, "v", out, seed=1)
    assert (est.cached_segments, est.free_segments) == (0, 1)
