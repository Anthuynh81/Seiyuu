"""Engine benchmark (step 1): stats/RTF/report math, the orchestration loop with fake engines,
GPU-manager discipline, VRAM probe selection (no torch.cuda when it can't apply), and the
skip-with-reason paths. CPU-only; no real engine, GPU, nvidia-smi, or whisper is touched."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from seiyuu.bench import (
    AbortKind,
    BenchError,
    BenchSegment,
    EngineStatus,
    SegmentKind,
    VoicePlan,
    book_segments,
    bundled_segments,
    console_summary,
    parse_voice_overrides,
    render_markdown,
    resolve_voice_plans,
    run_benchmark,
    write_report,
)
from seiyuu.bench.models import (
    BenchReport,
    EngineBenchResult,
    SegmentResult,
    distribution,
    finalize_engine_stats,
    percentile,
    rtf,
    summarize_validation,
)
from seiyuu.bench.runner import (
    REFERENCE_VOICE_ID,
    benchmark_engine,
    check_engines,
    compare_samples,
    engine_preflight,
    stage_reference,
)
from seiyuu.bench.segments import BenchInputError, classify
from seiyuu.bench.vram import (
    SmiSamplerProbe,
    TorchCudaProbe,
    VramStats,
    default_probes,
)
from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine
from seiyuu.gpu import GpuBusyError, GpuResourceManager
from seiyuu.validate import ValidationResult
from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceMeta

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- fakes -----------------------------------------------------------------------------------


class BenchFakeEngine(TTSEngine):
    """Deterministic sine at 24 kHz: 0.01 s of audio per character. Records lifecycle events."""

    def __init__(
        self,
        engine_id: str = "kokoro",
        *,
        uses_gpu: bool = False,
        events: list[str] | None = None,
        fail_on: str | None = None,
        cost: float = 0.0,
        jitter: bool = False,
        version_error: Exception | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.uses_gpu = uses_gpu
        self.events = events if events is not None else []
        self.fail_on = fail_on
        self.cost = cost
        self.jitter = jitter
        self.version_error = version_error
        self._n = 0

    @property
    def model_version(self) -> str:
        if self.version_error is not None:
            raise self.version_error
        return f"{self.engine_id}-fake-1"

    @property
    def native_sample_rate(self) -> int:
        return 24_000

    def list_voices(self) -> list[EngineVoice]:
        return []

    def cost_estimate(self, text: str) -> float:
        return self.cost

    def warm(self) -> None:
        self.events.append(f"warm:{self.engine_id}")

    def unload(self) -> None:
        self.events.append(f"unload:{self.engine_id}")

    def _synthesize_native(self, text: str, voice: str, settings: dict[str, Any]):
        if self.fail_on and self.fail_on in text:
            raise SynthesisError(f"boom on {self.fail_on!r}")
        self.events.append(f"synth:{self.engine_id}:{settings.get('seed')}")
        self._n += 1
        n = int(0.01 * len(text) * 24_000)
        t = np.arange(n) / 24_000
        samples = 0.3 * np.sin(2 * np.pi * 220 * t)
        if self.jitter:
            samples = samples + 1e-3 * self._n
        return samples.astype(np.float32), 24_000


class StepClock:
    """Each call advances 0.5 s, so every measured interval (t1 - t0) is exactly 0.5 s."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.5
        return self.t


def _ident(text: str, _engine_id: str) -> str:
    return text


SEGS = [
    BenchSegment(index=0, kind=SegmentKind.DIALOGUE, text="Hello there."),  # 12 chars
    BenchSegment(index=1, kind=SegmentKind.NARRATION, text="A" * 100),  # 100 chars
]


def _plan(engine_id: str = "kokoro", **kw: Any) -> VoicePlan:
    return VoicePlan(
        engine_id=engine_id, voice_id="v", engine_voice="v", seed=kw.pop("seed", 7), **kw
    )


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Any torch.cuda call except is_available() fails the test; is_available() is counted."""
    calls: list[str] = []

    def forbidden(name: str):
        def _f(*_a: Any, **_k: Any) -> Any:
            raise AssertionError(f"torch.cuda.{name} must not be called")

        return _f

    for name in (
        "reset_peak_memory_stats",
        "memory_allocated",
        "max_memory_allocated",
        "max_memory_reserved",
        "synchronize",
        "empty_cache",
        "get_device_name",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden(name))

    def is_available() -> bool:
        calls.append("is_available")
        return False

    monkeypatch.setattr(torch.cuda, "is_available", is_available)
    return calls


# --- pure stats ------------------------------------------------------------------------------


def test_percentile_matches_numpy_linear() -> None:
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    for q in (0, 10, 50, 90, 100):
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_rtf_and_distribution() -> None:
    assert rtf(2.0, 4.0) == 0.5
    assert rtf(1.0, 0.0) is None  # no audio -> no RTF, never a ZeroDivisionError
    d = distribution([1.0, 2.0, 3.0, 4.0])
    assert d is not None and (d.n, d.mean, d.p50, d.min, d.max) == (4, 2.5, 2.5, 1.0, 4.0)
    assert distribution([]) is None


def test_finalize_engine_stats_aggregates() -> None:
    r = EngineBenchResult(engine="kokoro")
    r.segments = [
        SegmentResult(
            index=0, kind=SegmentKind.DIALOGUE, chars=10, words=2, wall_s=1.0, audio_s=2.0, rtf=0.5
        ),
        SegmentResult(
            index=1, kind=SegmentKind.NARRATION, chars=30, words=6, wall_s=3.0, audio_s=2.0, rtf=1.5
        ),
    ]
    finalize_engine_stats(r)
    assert r.total_wall_s == 4.0 and r.total_audio_s == 4.0
    assert r.rtf == 1.0  # aggregate is total/total, NOT the mean of per-segment RTFs
    assert r.rtf_dist is not None and r.rtf_dist.mean == 1.0
    assert r.chars_per_s == 10.0
    empty = finalize_engine_stats(EngineBenchResult(engine="x"))
    assert empty.rtf is None and empty.rtf_dist is None


def test_summarize_validation() -> None:
    segs = [
        SegmentResult(
            index=i,
            kind=SegmentKind.DIALOGUE,
            chars=1,
            words=1,
            wall_s=1,
            audio_s=1,
            rtf=1,
            validation_score=s,
            validation_ok=s >= 0.85,
        )  # fmt: skip
        for i, s in enumerate([0.9, 0.5, 1.0, 0.86])
    ]
    v = summarize_validation(segs, 0.85)
    assert v is not None and (v.passed, v.total, v.pass_rate, v.min_score) == (3, 4, 0.75, 0.5)
    assert summarize_validation(segs[:0], 0.85) is None


def test_compare_samples() -> None:
    a = np.array([0.1, 0.2], dtype=np.float32)
    assert compare_samples(a, a.copy(), 0, 1).identical
    diff = compare_samples(a, a + 0.5, 0, 1)
    assert not diff.identical and diff.max_abs_diff == pytest.approx(0.5)
    ragged = compare_samples(a, a[:1], 0, 1)
    assert not ragged.same_length and ragged.max_abs_diff is None


# --- segments --------------------------------------------------------------------------------


def test_bundled_segments_mix() -> None:
    segs = bundled_segments()
    assert 8 <= len(segs) <= 12
    kinds = {s.kind for s in segs}
    assert kinds == {SegmentKind.DIALOGUE, SegmentKind.NARRATION}
    narr = [len(s.text) for s in segs if s.kind is SegmentKind.NARRATION]
    dia = [len(s.text) for s in segs if s.kind is SegmentKind.DIALOGUE]
    assert min(narr) > 3 * max(dia)  # long narration vs short dialogue
    assert [s.index for s in segs] == list(range(len(segs)))


def _write_book(book_dir: Path) -> None:
    book = {
        "book_meta": {
            "book_id": "bk",
            "title": "T",
            "source_path": "x.epub",
            "source_sha256": "0" * 64,
        },
        "chapters": [
            {"title": "One", "blocks": [{"id": "ch001_b0001", "type": "paragraph", "text": "a"}]},
            {
                "title": "Two",
                "blocks": [
                    {"id": "ch002_b0001", "type": "heading", "text": "Chapter Two"},
                    {"id": "ch002_b0002", "type": "scene_break", "text": ""},
                    {"id": "ch002_b0003", "type": "paragraph", "text": '"Run!" she said.'},
                    {"id": "ch002_b0004", "type": "paragraph", "text": "The night was cold."},
                ],
            },
        ],
    }
    book_dir.mkdir(parents=True)
    (book_dir / "normalized.json").write_text(json.dumps(book), encoding="utf-8")


def test_book_segments_takes_speakable_blocks(tmp_path: Path) -> None:
    _write_book(tmp_path / "bk")
    segs = book_segments(tmp_path / "bk", 2, limit=2)
    assert [s.source_id for s in segs] == ["ch002_b0001", "ch002_b0003"]  # scene break skipped
    assert segs[1].kind is SegmentKind.DIALOGUE
    with pytest.raises(BenchInputError, match="no chapter 9"):
        book_segments(tmp_path / "bk", 9)
    with pytest.raises(BenchInputError, match="normalized.json"):
        book_segments(tmp_path / "missing", 1)
    assert classify("“Hi”") is SegmentKind.DIALOGUE
    assert classify("plain") is SegmentKind.NARRATION


# --- orchestration ---------------------------------------------------------------------------


def test_cpu_engine_loop_measures_without_touching_cuda(no_cuda: list[str]) -> None:
    events: list[str] = []
    engine = BenchFakeEngine(events=events)
    result, audio = benchmark_engine(
        _plan(), SEGS, gpu=GpuResourceManager(), engine_factory=lambda *_: engine,
        normalizer=_ident, clock=StepClock(), warmup=1,
    )  # fmt: skip
    assert no_cuda == []  # uses_gpu False: not even is_available() is consulted
    assert result.status is EngineStatus.OK
    assert result.cold_load_s == 0.5 and result.first_synth_s == 0.5
    assert [s.wall_s for s in result.segments] == [0.5, 0.5]
    assert [s.audio_s for s in result.segments] == pytest.approx([0.12, 1.0])
    assert result.segments[1].rtf == pytest.approx(0.5)
    assert result.total_audio_s == pytest.approx(1.12)
    assert result.vram is None  # no GPU engine -> no probes at all
    assert result.determinism is not None and result.determinism.identical
    assert result.determinism.segment_index == 0  # the shortest segment
    assert len(audio) == 2
    # warm, 1 warm-up, 2 timed, 2 determinism synths — all with the plan's seed — then unload
    assert events == ["warm:kokoro"] + ["synth:kokoro:7"] * 5 + ["unload:kokoro"]


def test_gpu_engine_without_cuda_gets_no_torch_probe(no_cuda: list[str]) -> None:
    engine = BenchFakeEngine(uses_gpu=True)
    assert default_probes(engine, has_smi=lambda: False) == []
    assert no_cuda == ["is_available"]  # asked once, and nothing else on torch.cuda
    # a subprocess engine never even asks (a torch.cuda call would create a parent context)
    no_cuda.clear()
    assert default_probes(BenchFakeEngine("indextts2", uses_gpu=True), has_smi=lambda: False) == []
    assert no_cuda == []
    # non-GPU engine: no probes, no questions
    assert default_probes(BenchFakeEngine(), has_smi=lambda: True) == []


def test_default_probes_selection() -> None:
    gpu_engine = BenchFakeEngine(uses_gpu=True)
    probes = default_probes(gpu_engine, cuda_available=lambda: True, has_smi=lambda: True)
    assert [type(p) for p in probes] == [TorchCudaProbe, SmiSamplerProbe]
    sub = default_probes(
        BenchFakeEngine("indextts2", uses_gpu=True),
        cuda_available=lambda: pytest.fail("must not ask CUDA for a subprocess engine"),
        has_smi=lambda: True,
    )
    assert [type(p) for p in sub] == [SmiSamplerProbe]


def test_gpu_engines_acquire_and_unload_between(no_cuda: list[str]) -> None:
    events: list[str] = []
    gpu = GpuResourceManager()
    engines = {
        "kokoro": BenchFakeEngine("kokoro", uses_gpu=True, events=events),
        "chatterbox": BenchFakeEngine("chatterbox", uses_gpu=True, events=events),
    }
    seen_events: list[list[str]] = []

    def factory(engine_id: str, _vd: Path | None) -> TTSEngine:
        seen_events.append(list(events))
        return engines[engine_id]

    report = run_benchmark(
        [_plan("kokoro"), _plan("chatterbox")], SEGS, gpu=gpu, engine_factory=factory,
        probe_factory=lambda _e: [], normalizer=_ident, clock=StepClock(), warmup=0,
        determinism=False, preflight=None,
    )  # fmt: skip
    assert [r.status for r in report.results] == [EngineStatus.OK, EngineStatus.OK]
    assert report.aborted is None
    # kokoro was unloaded at the end of ITS hold, before chatterbox was even constructed ...
    assert seen_events[1][-1] == "unload:kokoro"
    assert events.index("unload:kokoro") < events.index("warm:chatterbox")
    # ... but the card stays claimed: the last engine is still the manager's resident (already
    # unloaded) until the caller's single free_all(), whose unload is an idempotent repeat
    assert gpu.resident == "chatterbox" and events[-1] == "unload:chatterbox"
    gpu.free_all()
    assert gpu.resident is None


def test_card_stays_claimed_between_engines(tmp_path: Path, no_cuda: list[str]) -> None:
    """Another process (a second manager on the same gpu.lock) can never grab the card in the
    gap between two engines; it is released only by the caller's final free_all()."""
    lock = tmp_path / "gpu.lock"
    gpu = GpuResourceManager(lock_path=lock)
    rival = GpuResourceManager(lock_path=lock)  # stands in for the API server
    rival_attempts: list[str] = []

    def factory(engine_id: str, _vd: Path | None) -> TTSEngine:
        if engine_id == "chatterbox":  # the gap between the two engines
            with pytest.raises(GpuBusyError):
                with rival.acquire(BenchFakeEngine(uses_gpu=True), "server"):
                    pass
            rival_attempts.append("refused")
        return BenchFakeEngine(engine_id, uses_gpu=True)

    report = run_benchmark(
        [_plan("kokoro"), _plan("chatterbox")], SEGS, gpu=gpu, engine_factory=factory,
        probe_factory=lambda _e: [], normalizer=_ident, clock=StepClock(), warmup=0,
        determinism=False, preflight=None,
    )  # fmt: skip
    assert rival_attempts == ["refused"]
    assert [r.status for r in report.results] == [EngineStatus.OK, EngineStatus.OK]
    gpu.free_all()
    with rival.acquire(BenchFakeEngine(uses_gpu=True), "server"):  # free after the run
        pass
    rival.free_all()


def test_probes_run_inside_gpu_hold_and_report() -> None:
    order: list[str] = []

    class Probe:
        def begin(self) -> None:
            order.append("begin")

        def mark_loaded(self) -> None:
            order.append("loaded")

        def end(self, stats: VramStats) -> None:
            order.append("end")
            stats.methods.append("fake")
            stats.smi_peak_delta_mb = 1234.0

    engine = BenchFakeEngine(uses_gpu=True, events=order)
    result, _ = benchmark_engine(
        _plan(), SEGS, gpu=GpuResourceManager(), engine_factory=lambda *_: engine,
        probe_factory=lambda _e: [Probe()], normalizer=_ident, clock=StepClock(),
        warmup=0, determinism=False,
    )  # fmt: skip
    assert order[:3] == ["begin", "warm:kokoro", "loaded"]
    assert order[-2:] == ["end", "unload:kokoro"]  # measured before the free
    assert result.vram is not None and result.vram.smi_peak_delta_mb == 1234.0


def test_nondeterministic_engine_is_reported() -> None:
    engine = BenchFakeEngine(jitter=True)
    result, _ = benchmark_engine(
        _plan(), SEGS, gpu=GpuResourceManager(), engine_factory=lambda *_: engine,
        normalizer=_ident, clock=StepClock(), warmup=0,
    )  # fmt: skip
    assert result.determinism is not None and not result.determinism.identical
    assert result.determinism.max_abs_diff == pytest.approx(1e-3, rel=1e-3)


def test_failure_mid_run_keeps_partials_and_frees_gpu() -> None:
    events: list[str] = []
    gpu = GpuResourceManager()
    engine = BenchFakeEngine(uses_gpu=True, events=events, fail_on="AAAA")
    result, audio = benchmark_engine(
        _plan(), SEGS, gpu=gpu, engine_factory=lambda *_: engine, probe_factory=lambda _e: [],
        normalizer=_ident, clock=StepClock(), warmup=0,
    )  # fmt: skip
    assert result.status is EngineStatus.FAILED
    assert "segment 1" in (result.reason or "") and "boom" in (result.reason or "")
    assert len(result.segments) == 1 and len(audio) == 1  # partial numbers kept
    assert result.rtf is not None
    assert events[-1] == "unload:kokoro" and not result.unload_failed


def test_gpu_busy_propagates_from_engine_and_run_keeps_partials(tmp_path: Path) -> None:
    class BusyGpu:
        def acquire(self, *_a: Any) -> Any:
            raise GpuBusyError("another process holds the GPU")

    events: list[str] = []
    with pytest.raises(GpuBusyError):
        benchmark_engine(
            _plan(), SEGS, gpu=BusyGpu(),
            engine_factory=lambda *_: BenchFakeEngine(uses_gpu=True, events=events),
            probe_factory=lambda _e: [], normalizer=_ident,
        )  # fmt: skip
    assert events == []  # nothing loaded, so nothing to unload

    # run level: the server takes the card before the run (busy at the first GPU engine), but
    # a CPU engine measured before it is kept and the run returns a report instead of raising
    lock = tmp_path / "gpu.lock"
    server = GpuResourceManager(lock_path=lock)
    gpu = GpuResourceManager(lock_path=lock)
    engines = {
        "kokoro": BenchFakeEngine("kokoro"),  # uses_gpu False: never touches the lock
        "chatterbox": BenchFakeEngine("chatterbox", uses_gpu=True),
        "indextts2": BenchFakeEngine("indextts2", uses_gpu=True),
    }
    with server.acquire(BenchFakeEngine(uses_gpu=True), "server"):
        report = run_benchmark(
            [_plan("kokoro"), _plan("chatterbox"), _plan("indextts2")], SEGS, gpu=gpu,
            engine_factory=lambda eid, _vd: engines[eid], probe_factory=lambda _e: [],
            normalizer=_ident, clock=StepClock(), warmup=0, determinism=False, preflight=None,
        )  # fmt: skip
    server.free_all()
    assert report.aborted is AbortKind.GPU_BUSY and "another seiyuu process" in (
        report.abort_detail or ""
    )
    k, c, i = report.results
    assert k.status is EngineStatus.OK and len(k.segments) == 2
    assert c.status is EngineStatus.SKIPPED and (c.reason or "").startswith("GPU busy")
    assert i.status is EngineStatus.SKIPPED and "not run" in (i.reason or "")
    md = render_markdown(report)
    assert "RUN STOPPED EARLY (gpu_busy)" in md and "| kokoro | v | ok |" in md


class _UnloadBoom(BenchFakeEngine):
    def unload(self) -> None:
        super().unload()
        raise RuntimeError("CUDA error: an illegal memory access")


def test_unload_failure_is_recorded_and_stops_the_run(no_cuda: list[str]) -> None:
    events: list[str] = []
    gpu = GpuResourceManager()
    built: list[str] = []

    def factory(engine_id: str, _vd: Path | None) -> TTSEngine:
        built.append(engine_id)
        if engine_id == "kokoro":
            return _UnloadBoom("kokoro", uses_gpu=True, events=events, fail_on="AAAA")
        return BenchFakeEngine(engine_id, uses_gpu=True, events=events)

    report = run_benchmark(
        [_plan("kokoro"), _plan("chatterbox")], SEGS, gpu=gpu, engine_factory=factory,
        probe_factory=lambda _e: [], normalizer=_ident, clock=StepClock(), warmup=0,
        determinism=False, preflight=None,
    )  # fmt: skip
    k, c = report.results
    assert k.status is EngineStatus.FAILED and k.unload_failed
    reason = k.reason or ""
    assert "segment 1" in reason and "boom" in reason  # the real failure is NOT replaced ...
    assert "unload failed: RuntimeError: CUDA error" in reason  # ... the unload error is added
    assert len(k.segments) == 1  # partials kept
    # one heavy model at a time: nothing else is loaded on top of a possibly-resident model
    assert built == ["kokoro"] and "warm:chatterbox" not in events
    assert c.status is EngineStatus.SKIPPED and "not run" in (c.reason or "")
    assert report.aborted is AbortKind.UNLOAD_FAILED


def test_ctrl_c_mid_engine_keeps_measured_results(no_cuda: list[str]) -> None:
    events: list[str] = []

    class CtrlC(BenchFakeEngine):
        def _synthesize_native(self, text: str, voice: str, settings: dict[str, Any]):
            if "AAAA" in text:
                raise KeyboardInterrupt
            return super()._synthesize_native(text, voice, settings)

    engines = {
        "kokoro": BenchFakeEngine("kokoro", uses_gpu=True, events=events),
        "chatterbox": CtrlC("chatterbox", uses_gpu=True, events=events),
        "indextts2": BenchFakeEngine("indextts2", uses_gpu=True, events=events),
    }
    report = run_benchmark(
        [_plan("kokoro"), _plan("chatterbox"), _plan("indextts2")], SEGS,
        gpu=GpuResourceManager(), engine_factory=lambda eid, _vd: engines[eid],
        probe_factory=lambda _e: [], normalizer=_ident, clock=StepClock(), warmup=0,
        determinism=False, preflight=None,
    )  # fmt: skip
    k, c, i = report.results
    assert k.status is EngineStatus.OK and len(k.segments) == 2
    assert c.status is EngineStatus.FAILED and "interrupted (Ctrl+C) during segment 1" in (
        c.reason or ""
    )
    assert len(c.segments) == 1  # the segment timed before Ctrl+C is kept
    assert "unload:chatterbox" in events  # unloaded before the interrupt was surfaced
    assert i.status is EngineStatus.SKIPPED and "warm:indextts2" not in events
    assert report.aborted is AbortKind.INTERRUPTED


def test_worker_restarts_are_counted_and_flagged() -> None:
    class Restarty(BenchFakeEngine):
        """Mimics IndexTTS2Engine's counters: the 100-char segment OOMs once, then retries."""

        worker_restarts = 0
        worker_ooms = 0

        def _synthesize_native(self, text: str, voice: str, settings: dict[str, Any]):
            if text.startswith("AAAA"):
                self.worker_restarts += 1
                self.worker_ooms += 1
            return super()._synthesize_native(text, voice, settings)

    report = run_benchmark(
        [_plan("indextts2")], SEGS, gpu=GpuResourceManager(),
        engine_factory=lambda *_: Restarty("indextts2"), normalizer=_ident, clock=StepClock(),
        warmup=1, determinism=True, preflight=None,
    )  # fmt: skip
    (r,) = report.results
    assert [s.worker_restarts for s in r.segments] == [0, 1]  # attributed to the segment
    assert (r.worker_restarts, r.worker_ooms) == (1, 1)  # warm-up/determinism used seg 0
    assert r.status is EngineStatus.OK
    assert "WORKER RESTARTS: 1 (1 OOM)" in render_markdown(report)
    assert "1 (includes reload)" in render_markdown(report)
    assert "WORKER RESTARTS: 1 (1 OOM)" in console_summary(report)


def test_indextts2_engine_counts_worker_restarts(tmp_path: Path) -> None:
    """The real adapter's counters (fake transport; no subprocess): an OOM reply kills the
    worker, the retry on a fresh one succeeds, and both counters say so."""
    import soundfile as sf

    from seiyuu.engines.indextts2_engine import INDEXTTS2_SAMPLE_RATE, IndexTTS2Engine

    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "reference.wav").write_bytes(b"RIFF")
    made: list[int] = []

    class Transport:
        def __init__(self) -> None:
            made.append(1)
            self.n = len(made)

        def request(self, message: dict, *, timeout: float) -> dict:
            if message["cmd"] == "synthesize":
                if self.n == 1:
                    return {"ok": False, "error": "CUDA OOM", "oom": True}
                sf.write(message["out_path"], np.zeros(2205, np.float32), INDEXTTS2_SAMPLE_RATE)
            return {"ok": True}

        def close(self) -> None:
            pass

    eng = IndexTTS2Engine(
        voices_dir=tmp_path, checkpoints_dir=tmp_path, worker_python=tmp_path / "py",
        use_fp16=True, load_timeout=1, request_timeout=1, max_restarts=1,
        transport_factory=Transport,
    )  # fmt: skip
    assert (eng.worker_restarts, eng.worker_ooms) == (0, 0)
    eng.synthesize("Hello.", "v1", {"seed": 1})
    assert (eng.worker_restarts, eng.worker_ooms) == (1, 1) and len(made) == 2


@pytest.mark.parametrize(
    ("kwargs", "plan_kw", "match"),
    [
        ({}, {"skip_reason": "no voice configured"}, "no voice configured"),
        ({"preflight": lambda _e: "worker python missing"}, {}, "worker python missing"),
        ({"engine_factory": lambda *_: (_ for _ in ()).throw(ValueError("bad"))}, {}, "bad"),
        (
            {"engine_factory": lambda *_: BenchFakeEngine(version_error=SynthesisError("no ckpt"))},
            {},
            "no ckpt",
        ),
        ({"engine_factory": lambda *_: BenchFakeEngine(cost=0.01)}, {}, "paid"),
    ],
)
def test_skip_with_reason(kwargs: dict, plan_kw: dict, match: str, no_cuda: list[str]) -> None:
    gpu = GpuResourceManager()
    params: dict[str, Any] = {
        "engine_factory": lambda *_: pytest.fail("must not construct"),
        "normalizer": _ident,
        **kwargs,
    }
    result, audio = benchmark_engine(_plan(**plan_kw), SEGS, gpu=gpu, **params)
    assert result.status is EngineStatus.SKIPPED
    assert match in (result.reason or "")
    assert audio == [] and result.segments == []
    assert gpu.resident is None and no_cuda == []


def test_non_local_engine_is_refused() -> None:
    result, _ = benchmark_engine(
        _plan("elevenlabs"), SEGS, gpu=GpuResourceManager(),
        engine_factory=lambda *_: pytest.fail("a paid engine must never be constructed"),
    )  # fmt: skip
    assert result.status is EngineStatus.SKIPPED and "paid" in (result.reason or "")
    with pytest.raises(BenchError, match="not benchmarkable"):
        check_engines(["kokoro", "elevenlabs"])
    assert check_engines(["kokoro", "kokoro", "indextts2"]) == ["kokoro", "indextts2"]


def test_validation_and_audio_saving(tmp_path: Path) -> None:
    class FakeValidator:
        min_ratio = 0.85

        def validate(self, audio: Any, expected: str) -> ValidationResult:
            assert isinstance(audio, np.ndarray)  # in-memory 16 kHz, like the render loop
            ok = expected.startswith("Hello")
            return ValidationResult(
                ok=ok, score=1.0 if ok else 0.2, transcript="x", expected=expected
            )

    report = run_benchmark(
        [_plan()], SEGS, gpu=GpuResourceManager(), engine_factory=lambda *_: BenchFakeEngine(),
        normalizer=_ident, clock=StepClock(), warmup=0, determinism=False,
        validator=FakeValidator(), audio_dir=tmp_path / "audio", preflight=None,
    )  # fmt: skip
    r = report.results[0]
    assert r.validation is not None and (r.validation.passed, r.validation.total) == (1, 2)
    assert [s.validation_ok for s in r.segments] == [True, False]
    saved = sorted(p.name for p in (tmp_path / "audio" / "kokoro").iterdir())
    assert saved == ["00_dialogue.wav", "01_narration.wav"]


def test_report_writing_and_rendering(tmp_path: Path) -> None:
    report = run_benchmark(
        [_plan(), _plan("indextts2", skip_reason="indextts2: worker python not configured")],
        SEGS, gpu=GpuResourceManager(), engine_factory=lambda *_: BenchFakeEngine(),
        normalizer=_ident, clock=StepClock(), warmup=1, preflight=None,
        environment={"gpu": "Fake 12GB"},
    )  # fmt: skip
    json_path, md_path = write_report(report, tmp_path / "out")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["results"][0]["status"] == "ok"
    assert data["results"][1]["status"] == "skipped"
    assert data["environment"]["gpu"] == "Fake 12GB"
    md = md_path.read_text(encoding="utf-8")
    assert "| kokoro | v | ok |" in md
    assert "worker python not configured" in md
    assert "## kokoro — per segment" in md
    assert md == render_markdown(report)
    summary = console_summary(report)
    assert "SKIPPED" in summary and "RTF" in summary


# --- voice resolution ------------------------------------------------------------------------


def _clone(lib: VoiceLibrary, voice_id: str, *, engine: str = "chatterbox", consent: bool = True):
    d = lib.dir_for(voice_id)
    d.mkdir(parents=True)
    (d / "reference.wav").write_bytes(b"RIFF fake")
    meta = VoiceMeta(
        voice_id=voice_id, name=voice_id, kind=VoiceKind.CLONED, engine=engine,
        reference_audio="reference.wav", consent_attested=True, seed=99,
        settings={"chatterbox": {"exaggeration": 0.7}},
    )  # fmt: skip
    lib.save(meta)
    if not consent:  # simulate a tampered meta (bypassing save()'s gate)
        raw = json.loads(lib.meta_path(voice_id).read_text(encoding="utf-8"))
        raw["consent_attested"] = False
        lib.meta_path(voice_id).write_text(json.dumps(raw), encoding="utf-8")


def test_resolve_voice_plans(tmp_path: Path) -> None:
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib, "zz_unconsented", consent=False)
    _clone(lib, "narrator_ab12")
    plans = resolve_voice_plans(
        ["kokoro", "chatterbox", "indextts2"], library=lib, kokoro_default="bf_emma", seed=5
    )
    k, c, i = plans
    assert (k.engine_voice, k.seed, k.source, k.skip_reason) == ("bf_emma", 5, "preset", None)
    assert (c.voice_id, c.seed, c.source) == ("narrator_ab12", 99, "auto")  # pinned seed
    assert c.settings == {"exaggeration": 0.7} and c.voices_dir == lib.voices_dir
    assert i.voice_id == "narrator_ab12" and i.settings == {}  # only indextts2's own tunables

    # explicit override that fails consent -> skip with the library's reason
    (bad,) = resolve_voice_plans(
        ["chatterbox"], library=lib, overrides={"chatterbox": "zz_unconsented"}
    )
    assert bad.skip_reason and "consent" in bad.skip_reason

    # empty library, no reference -> actionable skip
    (none,) = resolve_voice_plans(["indextts2"], library=VoiceLibrary(tmp_path / "empty"))
    assert none.skip_reason and "--reference" in none.skip_reason


def test_auto_clone_passes_over_bad_voice_dirs(tmp_path: Path) -> None:
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib, "aa_mismatch")  # sorts first; its meta names another voice -> VoiceLibraryError
    raw = json.loads(lib.meta_path("aa_mismatch").read_text(encoding="utf-8"))
    raw["voice_id"] = "somebody_else"
    lib.meta_path("aa_mismatch").write_text(json.dumps(raw), encoding="utf-8")
    bad = lib.dir_for("ab_malformed")  # not valid VoiceMeta -> pydantic ValidationError
    bad.mkdir(parents=True)
    (bad / "meta.json").write_text('{"voice_id": "ab_malformed"}', encoding="utf-8")
    _clone(lib, "ac_no_reference")  # grandfathered consent, but the audio is gone
    lib.reference_path("ac_no_reference").unlink()
    _clone(lib, "zz_good")

    (plan,) = resolve_voice_plans(["chatterbox"], library=lib)
    assert (plan.voice_id, plan.source, plan.skip_reason) == ("zz_good", "auto", None)

    # an explicit pick of the malformed voice is a skip reason, not a traceback
    (plan,) = resolve_voice_plans(
        ["chatterbox"], library=lib, overrides={"chatterbox": "ab_malformed"}
    )
    assert plan.skip_reason and "validation error" in plan.skip_reason.lower()


def test_resolve_kokoro_library_preset(tmp_path: Path) -> None:
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(voice_id="hero_1", name="Hero", kind=VoiceKind.PRESET, engine="kokoro",
                  preset_id="am_adam", seed=123)
    )  # fmt: skip
    (plan,) = resolve_voice_plans(["kokoro"], library=lib, overrides={"kokoro": "hero_1"})
    assert (plan.voice_id, plan.engine_voice, plan.seed, plan.source) == (
        "hero_1", "am_adam", 123, "library",
    )  # fmt: skip
    (plan,) = resolve_voice_plans(["chatterbox"], library=lib, overrides={"chatterbox": "hero_1"})
    assert plan.skip_reason and "cloned" in plan.skip_reason


def test_reference_is_staged_without_consent_record(tmp_path: Path) -> None:
    ref = tmp_path / "clip.wav"
    ref.write_bytes(b"RIFF clip")
    scratch = stage_reference(ref, tmp_path / "scratch" / "voices")
    voice_dir = scratch / REFERENCE_VOICE_ID
    assert (voice_dir / "reference.wav").read_bytes() == b"RIFF clip"
    assert not (voice_dir / "meta.json").exists()  # never becomes a library voice
    lib = VoiceLibrary(tmp_path / "voices")
    (plan,) = resolve_voice_plans(["chatterbox"], library=lib, reference_voices_dir=scratch)
    assert (plan.voice_id, plan.source) == (REFERENCE_VOICE_ID, "reference")
    assert plan.voices_dir == scratch
    assert not lib.voices_dir.exists()  # the user's library is untouched
    with pytest.raises(BenchError, match="not found"):
        stage_reference(tmp_path / "nope.wav", tmp_path / "s2")


def test_parse_voice_overrides() -> None:
    assert parse_voice_overrides(["kokoro=bf_emma", "indextts2 = narr_1"]) == {
        "kokoro": "bf_emma",
        "indextts2": "narr_1",
    }
    for bad in (["kokoro"], ["=x"], ["elevenlabs=abc"]):
        with pytest.raises(BenchError):
            parse_voice_overrides(bad)


def test_indextts2_preflight_reasons(tmp_path: Path) -> None:
    cfg = SimpleNamespace(indextts2_worker_python=None, indextts2_checkpoints_dir=None)
    assert "worker python" in (engine_preflight("indextts2", cfg) or "")
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    cfg.indextts2_worker_python = py
    cfg.indextts2_checkpoints_dir = tmp_path / "ckpt"
    assert "checkpoints not found" in (engine_preflight("indextts2", cfg) or "")
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "config.yaml").write_text("x", encoding="utf-8")
    assert engine_preflight("indextts2", cfg) is None


def test_sdk_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert "not installed" in (engine_preflight("chatterbox") or "")


# --- probes ----------------------------------------------------------------------------------


def test_smi_sampler_peak_delta() -> None:
    readings = iter([1000.0, 1500.0, 5200.0, 4800.0])
    probe = SmiSamplerProbe(lambda: next(readings, 4000.0), interval_s=3600)  # thread idles
    probe.begin()
    probe.mark_loaded()  # 1500
    probe._sample()  # 5200
    stats = VramStats()
    probe.end(stats)  # final sample 4800
    assert stats.smi_baseline_mb == 1000.0 and stats.smi_peak_mb == 5200.0
    assert stats.smi_peak_delta_mb == 4200.0 and stats.methods == ["nvidia-smi"]

    dead = SmiSamplerProbe(lambda: None)
    dead.begin()
    empty = VramStats()
    dead.end(empty)
    assert empty.methods == [] and empty.smi_peak_delta_mb is None


def test_torch_probe_with_fake_cuda() -> None:
    mb = 1024 * 1024
    fake = SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        memory_allocated=lambda: 250 * mb,  # resident once loaded (transients freed)
        max_memory_allocated=iter([300 * mb, 900 * mb]).__next__,  # load peak, then run peak
        max_memory_reserved=lambda: 1024 * mb,
    )
    probe = TorchCudaProbe(cuda=fake)
    probe.begin()
    probe.mark_loaded()
    stats = VramStats()
    probe.end(stats)
    assert stats.torch_after_load_allocated_mb == 250.0  # CURRENT, not the load peak
    assert stats.torch_load_peak_allocated_mb == 300.0
    assert stats.torch_peak_allocated_mb == 900.0 and stats.torch_peak_reserved_mb == 1024.0


# --- the script ------------------------------------------------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "benchmark_engines_script", REPO_ROOT / "scripts" / "benchmark_engines.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_end_to_end_with_fake_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_cuda: list[str]
) -> None:
    script = _load_script()
    real_run = script.run_benchmark

    def run_with_fakes(plans, segments, **kw):
        kw.pop("probe_factory", None)
        return real_run(
            plans, segments, engine_factory=lambda *_: BenchFakeEngine(), preflight=None,
            clock=StepClock(), **kw,
        )  # fmt: skip

    monkeypatch.setattr(script, "run_benchmark", run_with_fakes)
    monkeypatch.setattr(script, "collect_environment", lambda _i: {"gpu": "fake"})
    monkeypatch.setenv("VOICES_DIR", str(tmp_path / "voices"))
    monkeypatch.setenv("INDEXTTS2_TEST_REFERENCE", str(tmp_path / "absent.wav"))
    from seiyuu.settings import get_settings

    get_settings.cache_clear()
    out = tmp_path / "bench"
    code = script.main(["--engines", "kokoro", "indextts2", "--out", str(out)])
    assert code == 0
    data = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    statuses = {r["engine"]: r["status"] for r in data["results"]}
    assert statuses == {"kokoro": "ok", "indextts2": "skipped"}  # no clone, no reference
    assert data["segment_count"] == len(bundled_segments())
    assert (out / "bench.md").is_file()
    assert len(list((out / "audio" / "kokoro").glob("*.wav"))) == len(bundled_segments())

    assert script.main(["--engines", "kokoro", "--voice", "bogus"]) == 2  # clean usage error

    # a configured test clip is NOT used silently: nothing measured -> exit 3, with a hint
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF clip")
    monkeypatch.setenv("INDEXTTS2_TEST_REFERENCE", str(clip))
    get_settings.cache_clear()
    out1 = tmp_path / "bench1"
    assert script.main(["--engines", "indextts2", "--out", str(out1)]) == 3
    (r,) = json.loads((out1 / "bench.json").read_text(encoding="utf-8"))["results"]
    assert r["status"] == "skipped" and "--use-test-reference --consent" in r["reason"]

    # ad-hoc clips need the (non-persisted) consent attestation
    for flags in (["--use-test-reference"], ["--reference", str(clip)]):
        assert script.main(["--engines", "indextts2", "--out", str(out1), *flags]) == 2

    # opted in + attested -> used ephemerally and recorded as an ad-hoc clip
    out2 = tmp_path / "bench2"
    argv = ["--engines", "indextts2", "--out", str(out2), "--no-audio"]
    assert script.main([*argv, "--use-test-reference", "--consent"]) == 0
    (r,) = json.loads((out2 / "bench.json").read_text(encoding="utf-8"))["results"]
    assert (r["status"], r["voice_source"]) == ("ok", "test-reference")
    assert r["reference_clip"] == str(clip.resolve())
    assert "not persisted" in (out2 / "bench.md").read_text(encoding="utf-8")
    assert not (tmp_path / "voices").exists()  # nothing persisted to the library
    assert not (out2 / "audio").exists()

    out3 = tmp_path / "bench3"
    argv = ["--engines", "chatterbox", "--out", str(out3), "--no-audio"]
    assert script.main([*argv, "--reference", str(clip), "--consent"]) == 0
    (r,) = json.loads((out3 / "bench.json").read_text(encoding="utf-8"))["results"]
    assert (r["voice_source"], r["reference_clip"]) == ("reference", str(clip.resolve()))


def test_script_exit_codes() -> None:
    script = _load_script()
    ok = EngineBenchResult(engine="kokoro")
    skipped = EngineBenchResult(engine="indextts2", status=EngineStatus.SKIPPED, reason="x")
    failed = EngineBenchResult(engine="chatterbox", status=EngineStatus.FAILED, reason="x")

    def report(*results: EngineBenchResult, aborted: AbortKind | None = None) -> BenchReport:
        return BenchReport(
            segments_source="bundled", segment_count=1, warmup=0, results=list(results),
            aborted=aborted,
        )  # fmt: skip

    assert script.exit_code(report(ok, skipped)) == 0
    assert script.exit_code(report(skipped, skipped)) == 3  # nothing measured is not success
    assert script.exit_code(report(ok, failed)) == 1
    assert script.exit_code(report(ok, skipped, aborted=AbortKind.GPU_BUSY)) == 2
    assert script.exit_code(report(ok, aborted=AbortKind.INTERRUPTED)) == 130
