"""Gated benchmark smoke: the REAL Kokoro engine through the real benchmark loop on 2 tiny
segments. Deselected by default (``-m 'not gpu'``); run with:

    uv run pytest -m gpu -k benchmark_kokoro
"""

import importlib.util

import pytest

from seiyuu.bench import BenchSegment, EngineStatus, SegmentKind, run_benchmark
from seiyuu.bench.runner import resolve_voice_plans
from seiyuu.gpu import GpuResourceManager
from seiyuu.voices import VoiceLibrary

pytestmark = pytest.mark.gpu


def test_benchmark_kokoro_two_segments(tmp_path) -> None:
    if importlib.util.find_spec("kokoro") is None:
        pytest.skip("kokoro SDK not installed")
    segments = [
        BenchSegment(index=0, kind=SegmentKind.DIALOGUE, text='"Hello," she said.'),
        BenchSegment(index=1, kind=SegmentKind.NARRATION, text="The rain fell softly."),
    ]
    plans = resolve_voice_plans(["kokoro"], library=VoiceLibrary(tmp_path / "voices"))
    gpu = GpuResourceManager()
    report = run_benchmark(plans, segments, gpu=gpu, warmup=1, audio_dir=tmp_path / "audio")
    (r,) = report.results
    assert r.status is EngineStatus.OK, r.reason
    assert r.cold_load_s is not None and r.cold_load_s > 0
    assert len(r.segments) == 2 and all(s.audio_s > 0.3 for s in r.segments)
    assert r.rtf is not None and r.rtf > 0
    assert r.determinism is not None and r.determinism.identical  # Kokoro is deterministic
    assert report.aborted is None
    assert gpu.resident == "kokoro"  # unloaded, but the card stays claimed until free_all()
    gpu.free_all()
    assert gpu.resident is None
