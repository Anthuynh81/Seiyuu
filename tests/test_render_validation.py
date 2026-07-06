"""Validation in the render loop: validate LLM-style engines, retry-with-seed, flag failures,
and persist the verdict so cache hits keep it. Kokoro (deterministic) skips validation.
"""

from fake_engine import FakeEngine
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.render import render_book
from seiyuu.validate import ValidationResult


def _one_block_book() -> NormalizedBook:
    return NormalizedBook(
        book_meta=BookMeta(
            book_id="valid-book-000000",
            title="V",
            source_path="v.epub",
            source_sha256="0" * 64,
        ),
        chapters=[
            Chapter(
                title="C1",
                blocks=[Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="Hello there.")],
            )
        ],
    )


class ScriptedValidator:
    """Returns scripted (ok, score) verdicts in call order; ignores the audio."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    def validate(self, wav_path, expected_text):
        ok, score = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return ValidationResult(ok=ok, score=score, transcript="heard", expected=expected_text)


def _validating_engine() -> FakeEngine:
    eng = FakeEngine()
    eng.requires_validation = True  # pretend it's an LLM-style (Chatterbox-like) engine
    return eng


def _seg(result):
    return result.manifest.chapters[0].segments[0]


def test_passes_first_try(tmp_path):
    eng = _validating_engine()
    v = ScriptedValidator([(True, 0.97)])
    result = render_book(_one_block_book(), eng, "v", tmp_path / "out", seed=41172, validator=v)
    seg = _seg(result)
    assert seg.synth_attempts == 1
    assert seg.validation.ok and seg.validation.score == 0.97
    assert result.validation_failures == 0
    assert len(eng.calls) == 1 and v.calls == 1


def test_retries_then_passes(tmp_path):
    eng = _validating_engine()
    v = ScriptedValidator([(False, 0.4), (True, 0.95)])
    result = render_book(_one_block_book(), eng, "v", tmp_path / "out", seed=41172, validator=v)
    seg = _seg(result)
    assert seg.synth_attempts == 2  # one retry
    assert seg.validation.ok and seg.validation.score == 0.95  # kept the better attempt
    assert result.validation_failures == 0
    assert len(eng.calls) == 2


def test_persistent_failure_is_flagged_not_dropped(tmp_path):
    eng = _validating_engine()
    v = ScriptedValidator([(False, 0.3), (False, 0.5), (False, 0.4)])
    result = render_book(
        _one_block_book(), eng, "v", tmp_path / "out", seed=41172,
        validator=v, validation_max_retries=2,
    )  # fmt: skip
    seg = _seg(result)
    assert seg.synth_attempts == 3  # initial + 2 retries
    assert seg.validation.ok is False
    assert seg.validation.score == 0.5  # best of the failed attempts
    assert seg.wav is not None  # surfaced for review, NOT dropped
    assert result.validation_failures == 1
    assert result.manifest.validation_failures == 1


def test_kokoro_skips_validation(tmp_path):
    eng = FakeEngine()  # requires_validation defaults to False
    v = ScriptedValidator([(False, 0.0)])
    result = render_book(_one_block_book(), eng, "v", tmp_path / "out", seed=41172, validator=v)
    seg = _seg(result)
    assert seg.validation is None
    assert seg.synth_attempts == 1
    assert v.calls == 0  # validator never consulted for a deterministic engine


def test_missing_verdict_revalidated_on_cache_hit(tmp_path):
    # A cache hit whose validation sidecar is MISSING (crash between the wav write and the
    # verdict write, or a pre-M4 segment) must be RE-validated, never shipped unvalidated. (#3)
    book, out = _one_block_book(), tmp_path / "out"
    first = render_book(
        book, _validating_engine(), "v", out, seed=41172,
        validator=ScriptedValidator([(True, 0.9)]),
    )  # fmt: skip
    assert first.synthesized == 1

    # simulate the lost verdict: drop the validation sidecar(s), keep the cached wav
    removed = [p for p in (out / "cache").glob("*.validation.json")]
    assert removed  # sanity: there was a verdict to lose
    for sidecar in removed:
        sidecar.unlink()

    fresh = ScriptedValidator([(False, 0.2)])
    second = render_book(book, _validating_engine(), "v", out, seed=41172, validator=fresh)
    assert second.cache_hits == 1 and second.synthesized == 0
    assert fresh.calls == 1  # re-validated the cached wav rather than trusting a missing verdict
    seg = _seg(second)
    assert seg.validation is not None and seg.validation.ok is False
    assert second.validation_failures == 1  # counted/flagged exactly like a fresh render

    # and the re-derived verdict is persisted again: a third run needs no validator call
    third_v = ScriptedValidator([(True, 1.0)])
    third = render_book(book, _validating_engine(), "v", out, seed=41172, validator=third_v)
    assert third.cache_hits == 1 and third_v.calls == 0
    assert _seg(third).validation is not None and _seg(third).validation.ok is False


def test_verdict_persists_across_cache_hit(tmp_path):
    book, out = _one_block_book(), tmp_path / "out"
    first = render_book(
        book, _validating_engine(), "v", out, seed=41172, validator=ScriptedValidator([(True, 0.9)])
    )
    assert first.synthesized == 1 and first.cache_hits == 0

    # second run: cache hit; a fresh validator must NOT be consulted, yet the verdict survives
    fresh = ScriptedValidator([(False, 0.0)])
    second = render_book(book, _validating_engine(), "v", out, seed=41172, validator=fresh)
    assert second.cache_hits == 1 and second.synthesized == 0
    assert fresh.calls == 0
    seg = _seg(second)
    assert seg.validation is not None and seg.validation.ok and seg.validation.score == 0.9
