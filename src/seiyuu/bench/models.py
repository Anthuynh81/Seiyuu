"""Benchmark payloads (pydantic) + the pure statistics behind them.

Everything here is I/O-free and deterministic so the RTF/stat math is unit-tested on CPU. The
report is a documented JSON file (``bench.json``) so before/after runs on different cards can be
diffed by a script, plus a markdown rendering for humans (``report.py``).

RTF convention (matches the rest of the project's notes): ``wall seconds / audio seconds`` —
lower is better, < 1.0 is faster than real time.
"""

import math
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Only LOCAL engines are ever benchmarked: a benchmark is an automatic loop, and paid APIs must
# never run in an automatic code path (CLAUDE.md). Anything else is refused before construction.
LOCAL_ENGINES: tuple[str, ...] = ("kokoro", "chatterbox", "indextts2")


class SegmentKind(StrEnum):
    DIALOGUE = "dialogue"
    NARRATION = "narration"


class BenchSegment(BaseModel):
    """One input text. ``text`` is RAW; it is normalized per engine profile at run time, exactly
    as render does, so the benchmark synthesizes what a real render would."""

    index: int
    kind: SegmentKind
    text: str
    source_id: str | None = None  # block id when taken from a real book


class VoicePlan(BaseModel):
    """The resolved voice for one engine (or the reason it can't run)."""

    engine_id: str
    voice_id: str  # what the report shows (library voice id, preset id, or 'bench_reference')
    engine_voice: str  # the arg handed to engine.synthesize()
    settings: dict[str, Any] = Field(default_factory=dict)
    seed: int
    voices_dir: Path | None = None  # cloning engines resolve reference.wav under this
    source: str = "preset"  # preset | library | auto | reference | test-reference
    # Ad-hoc clips only (--reference / --use-test-reference): the original clip path. Consent was
    # attested for THIS run via --consent and deliberately not persisted anywhere.
    reference_clip: str | None = None
    skip_reason: str | None = None


class SegmentResult(BaseModel):
    index: int
    kind: SegmentKind
    chars: int  # normalized text length
    words: int
    wall_s: float
    audio_s: float
    rtf: float | None  # None when the engine returned (near-)empty audio
    # Worker kills + retries inside this synth (IndexTTS-2 OOM/death/timeout): its wall time then
    # includes a full worker reboot + model reload, so its RTF is NOT a steady-state number.
    worker_restarts: int = 0
    validation_score: float | None = None
    validation_ok: bool | None = None
    transcript: str | None = None


class Distribution(BaseModel):
    """Summary of one per-segment metric."""

    n: int
    mean: float
    p50: float
    p90: float
    min: float
    max: float


class VramStats(BaseModel):
    """Peak memory. ``torch_*`` = this process's allocator (in-process engines only);
    ``smi_*`` = whole-card nvidia-smi samples (the only view of a subprocess worker)."""

    methods: list[str] = Field(default_factory=list)
    torch_peak_allocated_mb: float | None = None
    torch_peak_reserved_mb: float | None = None
    torch_after_load_allocated_mb: float | None = None  # CURRENT allocation once loaded
    torch_load_peak_allocated_mb: float | None = None  # peak DURING load (casts, temp copies)
    smi_baseline_mb: float | None = None
    smi_peak_mb: float | None = None
    smi_peak_delta_mb: float | None = None  # peak - baseline: this engine's incremental use
    smi_samples: int = 0


class DeterminismResult(BaseModel):
    segment_index: int
    seed: int
    identical: bool
    same_length: bool
    max_abs_diff: float | None = None  # None when lengths differ


class ValidationSummary(BaseModel):
    passed: int
    total: int
    pass_rate: float
    min_score: float | None
    mean_score: float | None
    min_ratio: float  # the validator's pass threshold


class EngineStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"  # never started (not configured / no voice / paid)
    FAILED = "failed"  # started and broke; partial numbers are kept


class EngineBenchResult(BaseModel):
    engine: str
    voice_id: str | None = None
    voice_source: str | None = None
    reference_clip: str | None = None  # ad-hoc clip (consent attested for this run only)
    status: EngineStatus = EngineStatus.OK
    reason: str | None = None
    model_version: str | None = None
    uses_gpu: bool | None = None
    seed: int | None = None
    cold_load_s: float | None = None  # engine.warm(): weights to device (or worker boot+load)
    first_synth_s: float | None = None  # untimed warm-up synth(s): conds/kernels/2nd pipeline
    segments: list[SegmentResult] = Field(default_factory=list)
    total_wall_s: float | None = None
    total_audio_s: float | None = None
    rtf: float | None = None  # aggregate: total wall / total audio
    rtf_dist: Distribution | None = None
    wall_dist: Distribution | None = None
    chars_per_s: float | None = None
    vram: VramStats | None = None
    determinism: DeterminismResult | None = None
    validation: ValidationSummary | None = None
    # Worker kills + retries over the whole engine run (timed + warm-up + determinism synths);
    # ooms = how many of those the worker flagged as out-of-memory. Non-zero = the card ran out.
    worker_restarts: int = 0
    worker_ooms: int = 0
    # unload() raised (or the handoff unload did): the model may still be on the card, so the run
    # stops rather than load a second heavy model on top of it.
    unload_failed: bool = False


class AbortKind(StrEnum):
    GPU_BUSY = "gpu_busy"  # another seiyuu process holds the card
    INTERRUPTED = "interrupted"  # Ctrl+C
    UNLOAD_FAILED = "unload_failed"  # an engine could not be freed


class BenchReport(BaseModel):
    schema_version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    environment: dict[str, Any] = Field(default_factory=dict)
    segments_source: str  # "bundled" or "book:<id>:ch<N>"
    segment_count: int
    warmup: int
    results: list[EngineBenchResult] = Field(default_factory=list)
    # Set when the run stopped early; every engine measured before that is still in ``results``
    # and engines never reached are SKIPPED with the abort reason.
    aborted: AbortKind | None = None
    abort_detail: str | None = None


# --- pure statistics -------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method), q in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty list")
    if not 0 <= q <= 100:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def distribution(values: list[float]) -> Distribution | None:
    if not values:
        return None
    return Distribution(
        n=len(values),
        mean=sum(values) / len(values),
        p50=percentile(values, 50),
        p90=percentile(values, 90),
        min=min(values),
        max=max(values),
    )


# Below this much audio an RTF is noise (or a divide-by-~zero), so it is reported as None.
MIN_AUDIO_S_FOR_RTF = 1e-3


def rtf(wall_s: float, audio_s: float) -> float | None:
    """Real-time factor = wall / audio; None when there is no meaningful audio."""
    if audio_s < MIN_AUDIO_S_FOR_RTF:
        return None
    return wall_s / audio_s


def finalize_engine_stats(result: EngineBenchResult) -> EngineBenchResult:
    """Fill the aggregate fields from ``result.segments`` (idempotent; mutates and returns)."""
    segs = result.segments
    if not segs:
        result.total_wall_s = result.total_audio_s = result.rtf = None
        result.rtf_dist = result.wall_dist = None
        result.chars_per_s = None
        return result
    total_wall = sum(s.wall_s for s in segs)
    total_audio = sum(s.audio_s for s in segs)
    result.total_wall_s = total_wall
    result.total_audio_s = total_audio
    result.rtf = rtf(total_wall, total_audio)
    result.rtf_dist = distribution([s.rtf for s in segs if s.rtf is not None])
    result.wall_dist = distribution([s.wall_s for s in segs])
    total_chars = sum(s.chars for s in segs)
    result.chars_per_s = total_chars / total_wall if total_wall > 0 else None
    return result


def summarize_validation(segs: list[SegmentResult], min_ratio: float) -> ValidationSummary | None:
    scored = [s for s in segs if s.validation_score is not None]
    if not scored:
        return None
    scores = [s.validation_score for s in scored if s.validation_score is not None]
    passed = sum(1 for s in scored if s.validation_ok)
    return ValidationSummary(
        passed=passed,
        total=len(scored),
        pass_rate=passed / len(scored),
        min_score=min(scores),
        mean_score=sum(scores) / len(scores),
        min_ratio=min_ratio,
    )
