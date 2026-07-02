"""Cooperative cancel checkpoints in the four heavy stage loops (M6a-3).

Each stage takes ``check_cancel`` and calls it between its expensive units. A raise must
propagate with crash-consistent state — caches keep finished units, no stage marker file
is written — because a canceled job's book must resume by simply re-running the stage.
"""

import pytest

from factories import make_book
from fake_engine import FakeEngine
from fake_provider import FakeProvider
from seiyuu.assemble import AssembleError, assemble_book
from seiyuu.assemble.pipeline import master_book
from seiyuu.attribute.cache import AttributionCache
from seiyuu.attribute.pipeline import attribute_book
from seiyuu.gpu import GpuResourceManager
from seiyuu.jobs import JobCanceled
from seiyuu.render import render_book, render_book_multivoice
from seiyuu.voices import VoiceAssignment
from test_render_multivoice import _library, _patch_engine, _report


class CancelAfter:
    """check_cancel fake: allow N checkpoint passes, then raise JobCanceled."""

    def __init__(self, allowed: int) -> None:
        self.allowed = allowed
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.calls > self.allowed:
            raise JobCanceled("test cancellation")


def test_attribute_cancels_before_any_llm_call(tmp_path):
    def script(chunk, registry, attempt):
        raise AssertionError("provider must not be called after cancellation")

    provider = FakeProvider(script)
    with AttributionCache(tmp_path / "attribution.db") as cache:
        with pytest.raises(JobCanceled):
            attribute_book(make_book(), provider, cache=cache, check_cancel=CancelAfter(0))
    assert provider.calls == []


def test_render_cancel_keeps_cache_but_writes_no_manifest(tmp_path):
    engine = FakeEngine()
    out = tmp_path / "book"
    # checks: chapter 1 (ok), block b0001 (ok, synthesized), block b0002 -> raise
    with pytest.raises(JobCanceled):
        render_book(make_book(), engine, "test_voice", out, check_cancel=CancelAfter(2))
    assert len(engine.calls) == 1
    assert not (out / "manifest.json").exists()

    # re-run without cancel: the canceled run's segment is a cache hit, then it completes
    result = render_book(make_book(), engine, "test_voice", out)
    assert result.cache_hits == 1 and result.synthesized == 4
    assert (out / "manifest.json").exists()


def test_multivoice_cancel_before_first_segment(tmp_path, monkeypatch):
    engine = FakeEngine()
    _patch_engine(monkeypatch, engine)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    with pytest.raises(JobCanceled):
        render_book_multivoice(
            _report(),
            make_book(),
            _library(tmp_path),
            assignment,
            tmp_path / "out",
            gpu=GpuResourceManager(),
            check_cancel=CancelAfter(1),  # chapter check passes; first segment raises
        )
    assert engine.calls == []
    assert not (tmp_path / "out" / "manifest.json").exists()


@pytest.fixture
def rendered_book_dir(tmp_path):
    render_book(make_book(), FakeEngine(), "test_voice", tmp_path / "book")
    return tmp_path / "book"


def test_assemble_cancel_keeps_finished_chapters(rendered_book_dir):
    with pytest.raises(JobCanceled):
        assemble_book(rendered_book_dir, check_cancel=CancelAfter(1))
    chapters = rendered_book_dir / "chapters"
    assert (chapters / "ch001.mp3").is_file()  # finished before the cancel
    assert not (chapters / "ch002.mp3").exists()


def test_master_cancel_removes_the_working_wav(rendered_book_dir):
    with pytest.raises(JobCanceled):
        master_book(rendered_book_dir, check_cancel=CancelAfter(0))
    assert not (rendered_book_dir / "master" / "book.wav").exists()
    assert not (rendered_book_dir / "test-book-00000000.m4b").exists()


def test_master_cancel_lands_between_streaming_and_encode(rendered_book_dir):
    """The full-book measure+encode tail must be cancelable, not just the chapter loop."""
    # checks: chapter 1, chapter 2, then the post-streaming checkpoint raises
    with pytest.raises(JobCanceled):
        master_book(rendered_book_dir, check_cancel=CancelAfter(2))
    assert not (rendered_book_dir / "test-book-00000000.m4b").exists()
    assert not (rendered_book_dir / "master").exists()  # working files cleaned, dir removed


class _FailedFfmpeg:
    returncode = 1
    stderr = "simulated ffmpeg failure"
    stdout = ""


def test_assemble_encode_failure_leaves_no_partial_mp3(rendered_book_dir, monkeypatch):
    """The registry infers `assembled` from chapters/*.mp3 — a failed encode must not
    leave a truncated mp3 (or a .part temp) at a name that check would count."""
    monkeypatch.setattr("seiyuu.assemble.pipeline.subprocess.run", lambda *a, **k: _FailedFfmpeg())
    with pytest.raises(AssembleError, match="simulated ffmpeg failure"):
        assemble_book(rendered_book_dir)
    chapters = rendered_book_dir / "chapters"
    assert not any(chapters.glob("*.mp3"))
    assert not any(chapters.glob("*.part"))


def test_master_encode_failure_leaves_no_m4b(rendered_book_dir, monkeypatch):
    """Same for `mastered` and {book_id}.m4b — and a re-master that fails must not have
    truncated a previous good m4b (the encode goes to a .part sibling first)."""
    monkeypatch.setattr("seiyuu.assemble.pipeline.subprocess.run", lambda *a, **k: _FailedFfmpeg())
    with pytest.raises(AssembleError, match="simulated ffmpeg failure"):
        master_book(rendered_book_dir)
    assert not (rendered_book_dir / "test-book-00000000.m4b").exists()
    assert not any(rendered_book_dir.glob("*.part"))
