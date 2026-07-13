"""Cost gate: estimate paid renders, refuse paid synthesis without allow_paid, resolve cloud
voices when authorized. ElevenLabs engine is faked (no API); narrator is a free local engine.
"""

import numpy as np
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
from seiyuu.engines import AudioFile
from seiyuu.gpu import GpuResourceManager
from seiyuu.render import RenderError, estimate_render_cost, render_book_multivoice
from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceMeta
from test_cloud_voice import FakeClient


class FakeElevenEngine(FakeEngine):
    engine_id = "elevenlabs"
    uses_gpu = False
    requires_validation = False

    def __init__(self, price_per_char=0.001):
        super().__init__()
        self.price_per_char = price_per_char
        self._client = FakeClient()

    @property
    def model_version(self):
        return "elevenlabs-test"

    @property
    def client(self):
        return self._client

    def cost_estimate(self, text):
        return len(text) * self.price_per_char


def _library(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(voice_id="narrator_v", name="Narrator", kind=VoiceKind.PRESET,
                  engine="kokoro", preset_id="af_heart")
    )  # fmt: skip
    lib.save(
        VoiceMeta(voice_id="elena_x", name="Elena", kind=VoiceKind.CLONED,
                  engine="elevenlabs", reference_audio="reference.wav", consent_attested=True)
    )  # fmt: skip
    AudioFile(samples=np.zeros(2400, dtype=np.float32)).save(lib.reference_path("elena_x"))
    return lib


def _report():
    return AttributionReport(
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


def _assignment():
    from seiyuu.voices import VoiceAssignment

    return VoiceAssignment(
        book_id="test-book-00000000", narrator_voice_id="narrator_v",
        assignments={"alice": "elena_x"},
    )  # fmt: skip


def _patch(monkeypatch, eleven):
    def fake_get(engine_id, **kw):
        return eleven if engine_id == "elevenlabs" else FakeEngine()

    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", fake_get)


def test_estimate_counts_paid_free_and_cached(tmp_path, monkeypatch):
    _patch(monkeypatch, FakeElevenEngine())
    lib, out = _library(tmp_path), tmp_path / "out"
    est = estimate_render_cost(_report(), make_book(), lib, _assignment(), out)
    assert est.paid_segments == 2  # alice dialogue -> elevenlabs
    assert est.free_segments == 3  # narration -> kokoro
    assert est.cached_segments == 0
    assert est.total_usd > 0


def test_render_refuses_paid_without_authorization(tmp_path, monkeypatch):
    _patch(monkeypatch, FakeElevenEngine())
    lib, out = _library(tmp_path), tmp_path / "out"
    with pytest.raises(RenderError, match="refusing paid synthesis"):
        render_book_multivoice(
            _report(), make_book(), lib, _assignment(), out,
            gpu=GpuResourceManager(), allow_paid=False,
        )  # fmt: skip


def test_render_allows_paid_and_resolves_cloud_voice(tmp_path, monkeypatch):
    eleven = FakeElevenEngine()
    _patch(monkeypatch, eleven)
    lib, out = _library(tmp_path), tmp_path / "out"
    result = render_book_multivoice(
        _report(), make_book(), lib, _assignment(), out,
        gpu=GpuResourceManager(), allow_paid=True,
    )  # fmt: skip
    assert "elena_x" in result.manifest.voices_used
    # the engine was handed a freshly-created CLOUD id, never the library voice_id
    voices_sent = {v for _, v in eleven.calls}
    assert all(v.startswith("cloud_") for v in voices_sent)
    assert "elena_x" not in voices_sent

    # everything is cached now -> a re-estimate shows nothing left to pay for
    est = estimate_render_cost(_report(), make_book(), lib, _assignment(), out)
    assert est.paid_segments == 0 and est.total_usd == 0.0
    assert est.cached_segments == 5


def test_force_reprices_cached_paid_and_a_nonforce_quote_is_refused(tmp_path, monkeypatch):
    # Parity: after a paid render caches everything, a FORCE estimate must re-price the cached
    # paid segments as billable (matching what a forced re-render actually calls), keep the SAME
    # paid fingerprint, and — crucially — a quote minted from the cheap non-force estimate must be
    # refused when a forced render's higher total is verified against it. That drift-up refusal is
    # why `force` rides the existing money check with NO new field in the signed token.
    from seiyuu.render import CostGateError, hash_assignment, issue_quote, verify_quote

    _patch(monkeypatch, FakeElevenEngine())
    lib, out = _library(tmp_path), tmp_path / "out"
    assignment = _assignment()
    render_book_multivoice(
        _report(), make_book(), lib, assignment, out,
        gpu=GpuResourceManager(), allow_paid=True,
    )  # fmt: skip

    cached = estimate_render_cost(_report(), make_book(), lib, assignment, out)
    forced = estimate_render_cost(_report(), make_book(), lib, assignment, out, force=True)
    assert cached.paid_segments == 0 and cached.total_usd == 0.0
    assert forced.paid_segments == 2 and forced.total_usd > 0.0 and forced.cached_segments == 0
    assert forced.fingerprint == cached.fingerprint  # fingerprint is force-independent

    ah = hash_assignment(assignment)
    quote = issue_quote(
        cached, book_id="test-book-00000000", chapters=(),
        assignment_hash=ah, max_usd=25.0, ttl_seconds=900, data_dir=tmp_path,
    )  # fmt: skip
    with pytest.raises(CostGateError, match="drifted upward"):
        verify_quote(
            quote, book_id="test-book-00000000", chapters=(),
            fingerprint=forced.fingerprint, assignment_hash=ah,
            recomputed_total_usd=forced.total_usd, max_usd=25.0, data_dir=tmp_path,
        )  # fmt: skip
