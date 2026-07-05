"""F2 — render-loop word piggyback: a `requires_validation` (Chatterbox/Fish-like) render caches
`{key_hash}.words.json` sidecars inline from the SAME transcription used for the whisper verdict,
while a deterministic (Kokoro-like) render writes NONE (those engines align lazily on first
Listen). No live whisper: a fake validator exposing validate_with_words is injected.
"""

from fake_engine import FakeEngine
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.render import render_book
from seiyuu.validate import SegmentWords, ValidationResult, WordTiming


def _book() -> NormalizedBook:
    return NormalizedBook(
        book_meta=BookMeta(
            book_id="words-book-000000", title="W", source_path="w.epub", source_sha256="0" * 64
        ),
        chapters=[
            Chapter(
                title="C1",
                blocks=[Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="Hello there.")],
            )
        ],
    )


class WordsValidator:
    """A validator that both scores AND yields word timings in one pass (the real Validator's
    validate_with_words contract). Used to prove the render loop piggybacks the words."""

    def __init__(self) -> None:
        self.word_passes = 0

    def validate_with_words(self, wav_path, expected_text):
        self.word_passes += 1
        result = ValidationResult(ok=True, score=0.99, transcript="heard", expected=expected_text)
        words = [
            WordTiming(start=0.0, end=0.3, word=" Hello"),
            WordTiming(start=0.3, end=0.6, word=" there"),
        ]
        return result, words


def _cache_words_files(out_dir):
    return list((out_dir / "cache").glob("*.words.json"))


def test_validating_render_writes_words_sidecar(tmp_path):
    eng = FakeEngine()
    eng.requires_validation = True  # Chatterbox-like: transcribes every segment
    v = WordsValidator()
    out = tmp_path / "out"
    render_book(_book(), eng, "v", out, seed=41172, validator=v)

    sidecars = _cache_words_files(out)
    assert len(sidecars) == 1  # exactly the one speakable block
    sw = SegmentWords.model_validate_json(sidecars[0].read_text(encoding="utf-8"))
    assert [w.word for w in sw.words] == [" Hello", " there"]
    assert sw.audio_duration > 0
    # words came from the SAME transcription as the verdict — one pass, no extra whisper
    assert v.word_passes == 1


def test_kokoro_render_writes_no_words_sidecar(tmp_path):
    eng = FakeEngine()  # requires_validation defaults to False (deterministic, Kokoro-like)
    v = WordsValidator()
    out = tmp_path / "out"
    render_book(_book(), eng, "v", out, seed=41172, validator=v)

    assert _cache_words_files(out) == []  # stays lazy — no inline whisper pass
    assert v.word_passes == 0  # the validator was never consulted for a deterministic engine
