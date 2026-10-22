"""Benchmark report rendering: ``bench.json`` (machine-diffable) + ``bench.md`` (human table)
+ a short console summary. Pure string building apart from ``write_report``."""

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from seiyuu.bench.models import BenchReport, EngineBenchResult, EngineStatus
from seiyuu.repository import atomic_write_text

JSON_NAME = "bench.json"
MARKDOWN_NAME = "bench.md"


def _f(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def _vram_cells(r: EngineBenchResult) -> tuple[str, str]:
    v = r.vram
    if v is None:
        return "-", "-"
    return _f(v.torch_peak_allocated_mb, 0), _f(v.smi_peak_delta_mb, 0)


def _det_cell(r: EngineBenchResult) -> str:
    d = r.determinism
    if d is None:
        return "-"
    if d.identical:
        return "yes"
    return "no (len differs)" if not d.same_length else f"no (max diff {d.max_abs_diff:.2g})"


def _val_cell(r: EngineBenchResult) -> str:
    v = r.validation
    if v is None:
        return "-"
    return f"{v.passed}/{v.total} ({v.pass_rate:.0%})"


def _restart_note(r: EngineBenchResult) -> str | None:
    if not r.worker_restarts:
        return None
    return (
        f"WORKER RESTARTS: {r.worker_restarts} ({r.worker_ooms} OOM); the affected timings "
        "include a full worker reboot + model reload, so RTF/p90 overstate steady state"
    )


def _voice_line(r: EngineBenchResult) -> str:
    line = f"voice `{r.voice_id}` ({r.voice_source})"
    if r.reference_clip:
        line += (
            f" from ad-hoc clip `{r.reference_clip}` (consent attested via --consent for this "
            "run only; not persisted)"
        )
    return line


def render_markdown(report: BenchReport) -> str:
    env = report.environment
    lines = [
        "# Seiyuu engine benchmark",
        "",
        f"- created: {report.created_at}",
        f"- segments: {report.segment_count} ({report.segments_source}); warm-up synths: "
        f"{report.warmup}",
    ]
    for key in ("gpu", "torch", "python", "platform", "git_commit"):
        if env.get(key):
            lines.append(f"- {key}: {env[key]}")
    if report.aborted is not None:
        lines += [
            "",
            f"**RUN STOPPED EARLY ({report.aborted.value}):** {report.abort_detail}. Engines "
            "measured before that are below; engines not reached are marked skipped.",
        ]
    lines += [
        "",
        "RTF = wall / audio (lower is better; < 1 is faster than real time). Cold load is "
        "`warm()` (weights to device; for indextts2 the worker boot + load). Warm-up synth is "
        "excluded from the timed segments. VRAM: torch = this process's allocator peak "
        "(in-process engines only); smi = whole-card nvidia-smi peak minus the baseline at "
        "engine start. Worker restarts = IndexTTS-2 worker kills after an OOM/crash (the retry "
        "reloads the model inside the timed call).",
        "",
        "| Engine | Voice | Status | Cold load (s) | Warm-up (s) | Timed wall (s) | Audio (s) "
        "| RTF agg | RTF p50 | RTF p90 | Peak VRAM torch (MB) | Peak VRAM smi Δ (MB) "
        "| Same-seed identical | Whisper pass | Worker restarts (OOM) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for r in report.results:
        torch_mb, smi_mb = _vram_cells(r)
        restarts = f"{r.worker_restarts} ({r.worker_ooms})" if r.worker_restarts else "0"
        lines.append(
            f"| {r.engine} | {r.voice_id or '-'} | {r.status.value} | {_f(r.cold_load_s, 1)} "
            f"| {_f(r.first_synth_s, 1)} | {_f(r.total_wall_s, 1)} | {_f(r.total_audio_s, 1)} "
            f"| {_f(r.rtf)} | {_f(r.rtf_dist.p50 if r.rtf_dist else None)} "
            f"| {_f(r.rtf_dist.p90 if r.rtf_dist else None)} | {torch_mb} | {smi_mb} "
            f"| {_det_cell(r)} | {_val_cell(r)} | {restarts} |"
        )
    notes = [(r, text) for r in report.results for text in (r.reason, _restart_note(r)) if text]
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- **{r.engine}** ({r.status.value}): {text}" for r, text in notes]
    for r in report.results:
        if not r.segments:
            continue
        lines += [
            "",
            f"## {r.engine} — per segment",
            "",
            f"model `{r.model_version}`, {_voice_line(r)}, seed {r.seed}",
            "",
            "| # | Kind | Chars | Wall (s) | Audio (s) | RTF | Whisper score | Worker restarts |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in r.segments:
            score = "-" if s.validation_score is None else f"{s.validation_score:.3f}"
            if s.validation_ok is False:
                score += " FAIL"
            restarted = f"{s.worker_restarts} (includes reload)" if s.worker_restarts else "0"
            lines.append(
                f"| {s.index} | {s.kind.value} | {s.chars} | {s.wall_s:.2f} | {s.audio_s:.1f} "
                f"| {_f(s.rtf)} | {score} | {restarted} |"
            )
    return "\n".join(lines) + "\n"


def console_summary(report: BenchReport) -> str:
    out = []
    for r in report.results:
        if r.status is EngineStatus.SKIPPED:
            out.append(f"{r.engine:<11} SKIPPED  {r.reason}")
            continue
        torch_mb, smi_mb = _vram_cells(r)
        line = (
            f"{r.engine:<11} {r.status.value.upper():<8} load {_f(r.cold_load_s, 1, 's')}  "
            f"RTF {_f(r.rtf)} (p90 {_f(r.rtf_dist.p90 if r.rtf_dist else None)})  "
            f"VRAM torch {torch_mb}MB / smi Δ {smi_mb}MB  det {_det_cell(r)}  "
            f"whisper {_val_cell(r)}"
        )
        if (note := _restart_note(r)) is not None:
            line += f"\n            {note}"
        if r.status is EngineStatus.FAILED:
            line += f"\n            {r.reason}"
        out.append(line)
    if report.aborted is not None:
        out.append(f"RUN STOPPED EARLY ({report.aborted.value}): {report.abort_detail}")
    return "\n".join(out)


def write_report(report: BenchReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = atomic_write_text(out_dir / JSON_NAME, report.model_dump_json(indent=2))
    md_path = atomic_write_text(out_dir / MARKDOWN_NAME, render_markdown(report))
    return json_path, md_path


def collect_environment(gpu_index: int = 0) -> dict[str, Any]:
    """Report header facts. Deliberately makes NO torch.cuda call (that would create a CUDA
    context in this process); GPU facts come from nvidia-smi instead."""
    from seiyuu.bench.vram import nvidia_smi_info

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = f"{torch.__version__} (cuda build {torch.version.cuda})"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        env["torch"] = None
    info = nvidia_smi_info(gpu_index)
    if info:
        env["gpu"] = (
            f"{info['name']} ({info['memory.total']}, driver {info['driver_version']}, "
            f"index {gpu_index})"
        )
    try:
        from seiyuu.settings import REPO_ROOT

        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        env["git_commit"] = None
    return env
