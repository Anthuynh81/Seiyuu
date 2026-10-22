"""Engine benchmark (step 1): re-measure local TTS speed / VRAM / determinism on the current GPU.

Driven by ``scripts/benchmark_engines.py``; the pieces here are importable and CPU-testable.
"""

from seiyuu.bench.models import (
    LOCAL_ENGINES,
    AbortKind,
    BenchReport,
    BenchSegment,
    EngineBenchResult,
    EngineStatus,
    SegmentKind,
    VoicePlan,
)
from seiyuu.bench.report import console_summary, render_markdown, write_report
from seiyuu.bench.runner import (
    BenchError,
    BenchInterrupted,
    benchmark_engine,
    parse_voice_overrides,
    resolve_voice_plans,
    run_benchmark,
)
from seiyuu.bench.segments import BenchInputError, book_segments, bundled_segments

__all__ = [
    "LOCAL_ENGINES",
    "AbortKind",
    "BenchError",
    "BenchInterrupted",
    "BenchInputError",
    "BenchReport",
    "BenchSegment",
    "EngineBenchResult",
    "EngineStatus",
    "SegmentKind",
    "VoicePlan",
    "benchmark_engine",
    "book_segments",
    "bundled_segments",
    "console_summary",
    "parse_voice_overrides",
    "render_markdown",
    "resolve_voice_plans",
    "run_benchmark",
    "write_report",
]
