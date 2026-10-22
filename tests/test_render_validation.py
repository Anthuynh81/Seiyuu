"""Validation in the render loop: validate LLM-style engines, retry-with-seed, flag failures,
and persist the verdict so cache hits keep it. Kokoro (deterministic) skips validation.
The whisper pass runs on a background CPU thread overlapped with GPU synthesis; the overlap
tests below prove the pipelining without any live whisper.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from fake_engine import FakeEngine
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.render import render_book
from seiyuu.validate import ValidationResult, Validator


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


def test_subset_merge_recomputes_validation_failures(tmp_path):
    # A chapter-subset render merges into the existing manifest; the merged
    # validation_failures aggregate must span the CARRIED-OVER chapters too, not just
    # this run's (which would report 0 and hide ch1's flagged segments).
    from factories import make_book

    book, out = make_book(), tmp_path / "out"
    first = render_book(
        book, _validating_engine(), "v", out, seed=1, chapters=(1,),
        validator=ScriptedValidator([(False, 0.4)]), validation_max_retries=0,
    )  # fmt: skip
    assert first.manifest.validation_failures == 3  # ch1's three speakable blocks all failed

    second = render_book(
        book, _validating_engine(), "v", out, seed=1, chapters=(2,),
        validator=ScriptedValidator([(True, 0.95)]), validation_max_retries=0,
    )  # fmt: skip
    assert second.validation_failures == 0  # this RUN was clean...
    assert second.manifest.validation_failures == 3  # ...but the merged book still flags ch1


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


def _blocks_book(*texts: str) -> NormalizedBook:
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
                blocks=[
                    Block(id=f"ch001_b{i:04d}", type=BlockType.PARAGRAPH, text=text)
                    for i, text in enumerate(texts, start=1)
                ],
            )
        ],
    )


def test_validation_overlaps_synthesis(tmp_path):
    # While segment 1's verdict is still pending, segment 2 must synthesize: the validator
    # parks until it observes a LATER synthesis start — which never happens in a serial
    # loop, where the next synthesis begins only after the verdict returns.
    second_synth_started = threading.Event()
    overlapped: list[bool] = []

    class SignalingEngine(FakeEngine):
        requires_validation = True

        def _synthesize_native(self, text, voice, settings):
            if len(self.calls) == 1:  # second segment reached
                second_synth_started.set()
            return super()._synthesize_native(text, voice, settings)

    class ParkedValidator:
        def validate(self, wav_path, expected_text):
            overlapped.append(second_synth_started.wait(timeout=10.0))
            return ValidationResult(ok=True, score=0.9, transcript="heard", expected=expected_text)

    result = render_book(
        _blocks_book("First sentence.", "Second sentence."),
        SignalingEngine(), "v", tmp_path / "out", seed=1, validator=ParkedValidator(),
    )  # fmt: skip
    assert result.synthesized == 2 and result.validation_failures == 0
    assert overlapped[0] is True  # segment 2 was synthesizing while verdict 1 was pending


def test_identical_blocks_synthesize_once(tmp_path):
    # Two blocks with the SAME text share one SegmentKey. The second must wait for the
    # first's in-flight verdict and reuse its cached wav — never synthesize a duplicate
    # (the serial loop's behavior, preserved under the overlap).
    eng = _validating_engine()
    v = ScriptedValidator([(True, 0.9)])
    result = render_book(
        _blocks_book("Twin sentence.", "Twin sentence."),
        eng, "v", tmp_path / "out", seed=7, validator=v,
    )  # fmt: skip
    assert len(eng.calls) == 1 and v.calls == 1
    assert result.synthesized == 1 and result.cache_hits == 1
    rows = result.manifest.chapters[0].segments
    assert rows[0].wav == rows[1].wav
    assert rows[1].validation is not None and rows[1].validation.ok


def test_cancel_while_verdict_pending_aborts_cleanly(tmp_path):
    # A cancel arriving while the render idles in the chapter drain (verdict still
    # pending) must abort promptly: the cancel exception propagates and no manifest
    # is written; the cached-segment resume contract is unchanged.
    class Cancel(Exception):
        pass

    entered = threading.Event()

    class SlowValidator:
        def validate(self, wav_path, expected_text):
            entered.set()
            time.sleep(0.5)  # keep the verdict pending while the cancel lands
            return ValidationResult(ok=True, score=0.9, transcript="heard", expected=expected_text)

    def check_cancel():
        if entered.is_set():
            raise Cancel()

    with pytest.raises(Cancel):
        render_book(
            _one_block_book(), _validating_engine(), "v", tmp_path / "out", seed=1,
            validator=SlowValidator(), check_cancel=check_cancel,
        )  # fmt: skip
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_real_validator_receives_in_memory_waveform(tmp_path):
    # Driven through the render loop, the REAL Validator gets the segment as an in-memory
    # 16 kHz waveform — no tmp-wav write/decode round trip (duck-typed test validators
    # above still receive paths).
    received = []

    class FakeWhisper:
        def transcribe(self, source, **kwargs):
            received.append(source)
            word = SimpleNamespace(start=0.0, end=0.5, word="Hello")
            seg = SimpleNamespace(text="Hello there.", words=[word])
            return iter([seg]), None

    v = Validator(model=FakeWhisper(), min_ratio=0.5)
    result = render_book(
        _one_block_book(), _validating_engine(), "v", tmp_path / "out", seed=1, validator=v
    )
    assert result.synthesized == 1 and result.validation_failures == 0
    assert isinstance(received[0], np.ndarray) and received[0].dtype == np.float32
    # canonical 24 kHz audio arrived resampled to whisper's 16 kHz
    duration = result.manifest.chapters[0].segments[0].duration_seconds
    assert len(received[0]) == pytest.approx(duration * 16_000, rel=0.01)
    assert not list((tmp_path / "out" / "cache").glob("*.tmp.wav"))
