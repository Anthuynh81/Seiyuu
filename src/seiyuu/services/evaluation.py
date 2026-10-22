"""Attribution accuracy harness: run several models ("arms") over the same chapters into
ISOLATED per-arm directories and score each against gold quote labels.

Isolation is the hard rule: an arm never reads or writes the book's own ``attribution.json``,
``edits.json`` or ``attribution.db``, and ``-o`` may not be (or sit inside) the book folder.
Each arm gets ``<out>/arm-<slug>/`` holding its own chunk cache (``attribution.db``), its raw
report (``attribution.json``) and its run stats, so a re-run of the same arm is served from
ITS cache (cached chunks are reported, and excluded from the seconds-per-chunk figure, which
only counts chunks that actually called the model).

Hybrid arms live in ``arm-<slug>-hybrid/``: the pipeline caches an escalated chunk under the
PRIMARY model's key, so sharing a cache with a local-only arm would later score the
escalation provider's answers as the local model's. Escalated chunk keys are also recorded
(``escalated_chunks.json``) so a cached re-run still reports them as escalations.

Instrumentation wraps the provider and the cache rather than touching the pipeline: every
``attribute_chunk`` call is timed and classified (ok / schema failure), the cache records
hits, and the narration fast path records synthesized chunks. GPU discipline matches
``run_attribution``: each arm holds the GPU manager and frees it in ``finally``, so two local
models are never resident together. Paid escalation runs only on an explicit ``hybrid=True``.
"""

import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from pydantic import BaseModel

import seiyuu.services.attribution as attribution_svc
from seiyuu.attribute import AttributionCache, attribute_book, write_attribution
from seiyuu.attribute.cache import ChunkCacheKey
from seiyuu.attribute.chunking import Chunk
from seiyuu.attribute.evaluate import (
    ArmScore,
    Bucket,
    GoldConflict,
    GoldQuote,
    describe_conflict,
    gold_conflicts,
    gold_speakers,
    score_report,
)
from seiyuu.attribute.models import AttributionReport, CharacterRegistry, ChunkAttribution
from seiyuu.attribute.providers.base import AttributionError, AttributionLLM, MalformedOutputError
from seiyuu.attribute.providers.local import OllamaModelMissingError, OllamaPreflightError
from seiyuu.gpu import get_gpu_manager
from seiyuu.ingest.models import NormalizedBook
from seiyuu.repository import atomic_write_text
from seiyuu.services.common import ServiceError

REPORT_JSON = "eval_report.json"
REPORT_MD = "eval_report.md"
ESCALATED_MARKER = "escalated_chunks.json"
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def arm_slug(model: str) -> str:
    """A short filesystem-safe directory name for a model id (``qwen2.5:7b`` -> ``qwen2.5_7b``)."""
    return _SLUG.sub("_", model).strip("._") or "model"


# --------------------------------------------------------------------- instrumentation


class CallRecord(BaseModel):
    chapter_index: int
    chunk_index: int
    role: str  # "primary" | "escalation"
    attempt: int
    seconds: float
    outcome: str  # "ok" | "schema_failure" | "error"


class _Recorder:
    def __init__(self, block_chapter: dict[str, int]) -> None:
        self.block_chapter = block_chapter
        self.calls: list[CallRecord] = []
        self.cache_hits = 0
        self.cached_escalations = 0
        self.fast_path = 0

    def chapter_of(self, chunk: Chunk) -> int:
        first = chunk.owned_blocks[0].id if chunk.owned_blocks else ""
        return self.block_chapter.get(first, 0)


class _InstrumentedProvider(AttributionLLM):
    """Transparent proxy: forwards everything to ``inner``, timing each chunk call."""

    def __init__(self, inner: AttributionLLM, recorder: _Recorder, role: str) -> None:
        # Deliberately no super().__init__: identity (and thus the cache key) is the inner's.
        self._inner = inner
        self._recorder = recorder
        self._role = role
        self.provider_id = inner.provider_id
        self.model_id = inner.model_id
        self.prompt_version = inner.prompt_version
        self.emit_thoughts = inner.emit_thoughts
        self.uses_gpu = getattr(inner, "uses_gpu", True)

    @property  # type: ignore[override]
    def dialogue_convention(self):  # the pipeline sets it per run; the inner splits on it
        return self._inner.dialogue_convention

    @dialogue_convention.setter
    def dialogue_convention(self, value) -> None:
        self._inner.dialogue_convention = value

    def narration_only_attribution(self, chunk: Chunk) -> ChunkAttribution | None:
        result = self._inner.narration_only_attribution(chunk)
        if result is not None:
            self._recorder.fast_path += 1
        return result

    def attribute_chunk(
        self, chunk: Chunk, registry: CharacterRegistry, attempt: int = 0
    ) -> ChunkAttribution:
        outcome = "error"
        start = time.perf_counter()
        try:
            result = self._inner.attribute_chunk(chunk, registry, attempt)
            outcome = "ok"
            return result
        except MalformedOutputError:
            outcome = "schema_failure"
            raise
        finally:
            self._recorder.calls.append(
                CallRecord(
                    chapter_index=self._recorder.chapter_of(chunk),
                    chunk_index=chunk.index,
                    role=self._role,
                    attempt=attempt,
                    seconds=time.perf_counter() - start,
                    outcome=outcome,
                )
            )

    def unload(self) -> None:
        self._inner.unload()

    def _complete_json(self, prompt, schema, attempt=0):  # pragma: no cover - never reached
        return self._inner._complete_json(prompt, schema, attempt)


class _CountingCache(AttributionCache):
    """Counts hits, and remembers which cached chunks the ESCALATION provider produced (the
    pipeline stores them under the primary key) so a cached re-run still reports them."""

    def __init__(self, db_path: Path, recorder: _Recorder, marker: Path) -> None:
        super().__init__(db_path)
        self._recorder = recorder
        self._marker = marker
        self._escalated: set[str] = (
            set(json.loads(marker.read_text(encoding="utf-8"))) if marker.exists() else set()
        )

    @staticmethod
    def _id(key: ChunkCacheKey) -> str:
        return key.model_dump_json()

    def get(self, key: ChunkCacheKey) -> ChunkAttribution | None:
        hit = super().get(key)
        if hit is not None:
            self._recorder.cache_hits += 1
            if self._id(key) in self._escalated:
                self._recorder.cached_escalations += 1
        return hit

    def put(self, key: ChunkCacheKey, attribution: ChunkAttribution) -> None:
        super().put(key, attribution)
        calls = self._recorder.calls
        # The pipeline's put() directly follows the call that produced this attribution.
        if calls and calls[-1].role == "escalation" and calls[-1].outcome == "ok":
            self._escalated.add(self._id(key))
            atomic_write_text(self._marker, json.dumps(sorted(self._escalated), indent=1))


# ------------------------------------------------------------------------------ stats


class ChunkTiming(BaseModel):
    chapter_index: int
    chunk_index: int
    seconds: float  # summed model-call wall time for this chunk (all attempts, all roles)
    attempts: int
    escalated: bool


class RunStats(BaseModel):
    wall_seconds: float  # whole arm, including fast path / cache / alias post-pass
    llm_chunks: int  # chunks that called the model
    cached_chunks: int
    fast_path_chunks: int
    llm_calls: int
    schema_failures: int  # MalformedOutputError attempts (retried locally)
    local_retries: int  # primary-provider attempts beyond the first
    escalations: int  # chunks handed to the escalation provider (hybrid only)
    cached_escalations: int = 0  # cache hits the escalation provider originally produced
    flagged_blocks: int  # blocks that fell back to narration after all retries
    seconds_per_chunk: float | None  # mean over llm_chunks
    p50_seconds_per_chunk: float | None
    max_seconds_per_chunk: float | None
    chunks: list[ChunkTiming]


def build_stats(
    calls: list[CallRecord],
    *,
    wall_seconds: float,
    cache_hits: int,
    fast_path: int,
    flagged_blocks: int,
    cached_escalations: int = 0,
) -> RunStats:
    """Aggregate raw call records into per-chunk timings and run totals (pure)."""
    per_chunk: dict[tuple[int, int], list[CallRecord]] = defaultdict(list)
    for c in calls:
        per_chunk[(c.chapter_index, c.chunk_index)].append(c)
    chunks = [
        ChunkTiming(
            chapter_index=ci,
            chunk_index=k,
            seconds=round(sum(c.seconds for c in recs), 4),
            attempts=len(recs),
            escalated=any(c.role == "escalation" for c in recs),
        )
        for (ci, k), recs in sorted(per_chunk.items())
    ]
    secs = sorted(t.seconds for t in chunks)
    return RunStats(
        wall_seconds=round(wall_seconds, 3),
        llm_chunks=len(chunks),
        cached_chunks=cache_hits,
        fast_path_chunks=fast_path,
        llm_calls=len(calls),
        schema_failures=sum(c.outcome == "schema_failure" for c in calls),
        local_retries=sum(c.role == "primary" and c.attempt > 0 for c in calls),
        escalations=sum(t.escalated for t in chunks),
        cached_escalations=cached_escalations,
        flagged_blocks=flagged_blocks,
        seconds_per_chunk=round(sum(secs) / len(secs), 3) if secs else None,
        p50_seconds_per_chunk=round(secs[len(secs) // 2], 3) if secs else None,
        max_seconds_per_chunk=round(secs[-1], 3) if secs else None,
        chunks=chunks,
    )


# -------------------------------------------------------------------------------- run


class ArmResult(BaseModel):
    arm: str  # directory name under out/ (``arm-<slug>`` or ``arm-<slug>-hybrid``)
    hybrid: bool = False  # an escalation provider was active (its answers may be scored)
    provider_id: str
    model_id: str
    prompt_version: str
    arm_dir: str
    stats: RunStats
    score: ArmScore


class ArmFailure(BaseModel):
    """An arm that died with a fatal AttributionError; the other arms are still reported."""

    model: str
    arm: str
    error: str


class EvalReport(BaseModel):
    book_id: str
    chapters: list[int]
    gold_quotes: int
    arms: list[ArmResult]
    failed: list[ArmFailure] = []
    gold_conflicts: list[GoldConflict] = []  # likely-unmerged duplicate gold speakers


def check_out_dir(out_dir: Path, book_dir: Path | None) -> None:
    """``-o`` must not be the book folder or anywhere inside it (isolation)."""
    if book_dir is None:
        return
    out, book = Path(out_dir).resolve(), Path(book_dir).resolve()
    if out == book or book in out.parents:
        raise ServiceError(
            f"output {out_dir} is inside the book folder {book_dir}; "
            "eval output must live outside it - pick another -o"
        )


def arm_dir_name(model: str, *, hybrid: bool) -> str:
    return f"arm-{arm_slug(model)}" + ("-hybrid" if hybrid else "")


def preflight_models(cfg, provider_id: str, models: list[str]) -> None:
    """Fail fast before ANY arm runs: Ollama up and EVERY local model pulled. Missing
    models are reported together, each with its ``ollama pull`` fix. Resolved through
    ``services.attribution`` at call time so the test suite's offline stub covers it."""
    if provider_id != "local":
        return
    missing: list[str] = []
    for model in dict.fromkeys(models):
        try:
            attribution_svc.ollama_preflight(cfg.ollama_base_url, [model])
        except OllamaModelMissingError:
            missing.append(model)
    if missing:
        fixes = "; ".join(f"ollama pull {m}" for m in missing)
        raise OllamaPreflightError(
            f"model(s) not pulled in Ollama: {', '.join(missing)} - run: {fixes}",
            base_url=cfg.ollama_base_url,
            model=missing[0],
        )


def gold_chapters(gold: list[GoldQuote]) -> tuple[int, ...]:
    return tuple(sorted({r.chapter_index for r in gold}))


def run_arm(
    book: NormalizedBook,
    gold: list[GoldQuote],
    *,
    cfg,
    model: str,
    out_dir: Path,
    chapters: tuple[int, ...],
    provider_id: str = "local",
    prompt_version: str | None = None,
    hybrid: bool = False,
    fresh: bool = False,
    book_dir: Path | None = None,
    provider: AttributionLLM | None = None,  # injectable for tests
    escalation_provider: AttributionLLM | None = None,
    progress: Callable[[str], None] | None = None,
    gpu=None,
) -> ArmResult:
    """Attribute ``chapters`` with one model into ``<out_dir>/arm-<slug>/`` and score it."""
    say = progress or (lambda _msg: None)
    check_out_dir(out_dir, book_dir)

    emit_thoughts = cfg.emit_thoughts
    prompt_version = prompt_version or ("v6" if emit_thoughts else cfg.attribution_prompt_version)
    inner = provider or attribution_svc.build_provider(
        cfg, provider_id, model, prompt_version, emit_thoughts=emit_thoughts
    )
    if escalation_provider is None and hybrid and inner.provider_id != "anthropic":
        escalation_provider = attribution_svc.build_provider(
            cfg, "anthropic", cfg.anthropic_model, prompt_version, emit_thoughts=emit_thoughts
        )
    escalating = escalation_provider is not None
    arm = arm_dir_name(model, hybrid=escalating)
    arm_dir = Path(out_dir) / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    db_path = arm_dir / "attribution.db"
    marker = arm_dir / ESCALATED_MARKER
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
    elif not escalating and marker.exists():
        # Unreachable with the arm-<slug>[-hybrid] split; refuse rather than score a cache
        # holding another provider's answers as this local model's own.
        raise ServiceError(
            f"{arm_dir} holds escalated (hybrid) chunks; re-run with --fresh or another -o"
        )

    block_chapter = {b.id: ci for ci, ch in enumerate(book.chapters, start=1) for b in ch.blocks}
    recorder = _Recorder(block_chapter)
    wrapped = _InstrumentedProvider(inner, recorder, "primary")
    wrapped_escalation = (
        _InstrumentedProvider(escalation_provider, recorder, "escalation")
        if escalation_provider is not None
        else None
    )

    gpu = gpu or get_gpu_manager()
    uses_gpu = getattr(inner, "uses_gpu", True)
    ctx = (
        gpu.acquire(inner, f"llm:eval:{inner.provider_id}:{inner.model_id}")
        if uses_gpu
        else nullcontext()
    )
    say(f"arm {arm}: {inner.provider_id}/{inner.model_id} (prompt {inner.prompt_version})")
    start = time.perf_counter()
    try:
        with ctx, _CountingCache(db_path, recorder, marker) as cache:
            report = attribute_book(
                book,
                wrapped,
                cache=cache,
                budget_tokens=cfg.attribution_chunk_tokens,
                overlap_blocks=cfg.attribution_chunk_overlap_blocks,
                max_local_retries=cfg.attribution_max_local_retries,
                escalation_provider=wrapped_escalation,
                chapters=chapters,
                progress=say,
                narration_fast_path=cfg.attribution_narration_fast_path,
            )
    finally:
        if uses_gpu:
            gpu.free_all()  # unload this arm's model before the next arm loads its own
    wall = time.perf_counter() - start

    write_attribution(report, arm_dir)
    stats = build_stats(
        recorder.calls,
        wall_seconds=wall,
        cache_hits=recorder.cache_hits,
        fast_path=recorder.fast_path,
        flagged_blocks=len(report.flagged),
        cached_escalations=recorder.cached_escalations,
    )
    atomic_write_text(arm_dir / "run_stats.json", stats.model_dump_json(indent=2))
    score = score_report(gold, report)
    atomic_write_text(arm_dir / "score.json", score.model_dump_json(indent=2))
    return ArmResult(
        arm=arm,
        hybrid=escalating,
        provider_id=report.provider_id,
        model_id=report.model_id,
        prompt_version=report.prompt_version,
        arm_dir=str(arm_dir),
        stats=stats,
        score=score,
    )


def run_eval(
    book: NormalizedBook,
    gold: list[GoldQuote],
    *,
    cfg,
    models: list[str],
    out_dir: Path,
    provider_id: str = "local",
    chapters: tuple[int, ...] = (),
    prompt_version: str | None = None,
    hybrid: bool = False,
    fresh: bool = False,
    book_dir: Path | None = None,
    provider_factory: Callable[[str], AttributionLLM] | None = None,  # tests
    progress: Callable[[str], None] | None = None,
) -> EvalReport:
    """Run every arm, score it, and write ``eval_report.json`` + ``eval_report.md``."""
    if not models:
        raise ServiceError("no models given; pass --models m1,m2")
    if not gold:
        raise ServiceError("gold file has no rows")
    book_id = book.book_meta.book_id
    stray = sorted({r.book_id for r in gold} - {book_id})
    if stray:
        raise ServiceError(f"gold rows belong to other book(s) {stray}, not {book_id}")
    chapters = chapters or gold_chapters(gold)
    gold_in_scope = [r for r in gold if r.chapter_index in set(chapters)]
    if not gold_in_scope:
        raise ServiceError(f"no gold rows in chapters {list(chapters)}")
    if len(set(models)) != len(models) or len({arm_slug(m) for m in models}) != len(models):
        raise ServiceError(f"duplicate model arms in {models}")
    check_out_dir(out_dir, book_dir)
    say = progress or (lambda _msg: None)
    conflicts = gold_conflicts(gold_speakers(gold_in_scope))
    for conflict in conflicts:
        say(f"WARNING: {describe_conflict(conflict)}")

    if provider_factory is None:
        preflight_models(cfg, provider_id, models)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms: list[ArmResult] = []
    failed: list[ArmFailure] = []
    for m in models:
        try:
            arms.append(
                run_arm(
                    book,
                    gold_in_scope,
                    cfg=cfg,
                    model=m,
                    out_dir=out_dir,
                    chapters=chapters,
                    provider_id=provider_id,
                    prompt_version=prompt_version,
                    hybrid=hybrid,
                    fresh=fresh,
                    book_dir=book_dir,
                    provider=provider_factory(m) if provider_factory else None,
                    progress=progress,
                )
            )
        except AttributionError as exc:
            # One bad candidate (context overflow, backend gone mid-run, ...) must not
            # discard the arms that finished: record it and keep comparing.
            say(f"arm {m} FAILED: {exc}")
            failed.append(
                ArmFailure(
                    model=m,
                    arm=arm_dir_name(m, hybrid=hybrid and provider_id != "anthropic"),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    report = EvalReport(
        book_id=book_id,
        chapters=list(chapters),
        gold_quotes=len(gold_in_scope),
        arms=arms,
        failed=failed,
        gold_conflicts=conflicts,
    )
    atomic_write_text(
        out_dir / REPORT_JSON,
        report.model_dump_json(indent=2, exclude={"arms": {"__all__": {"score": {"outcomes"}}}}),
    )
    atomic_write_text(out_dir / REPORT_MD, render_markdown(report))
    return report


# ---------------------------------------------------------------------------- report


def _pct(bucket: Bucket) -> str:
    return (
        "—"
        if bucket.accuracy is None
        else f"{bucket.accuracy * 100:.1f}% ({bucket.correct}/{bucket.n})"
    )


def _secs(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def render_markdown(report: EvalReport) -> str:
    """The human comparison: one summary row per arm, then per-arm breakdowns."""
    lines = [
        f"# Attribution eval — {report.book_id}",
        "",
        f"Chapters {report.chapters}; {report.gold_quotes} gold quotes.",
        "",
        "| Arm | Accuracy | Unattributed | Alias drift | s/chunk (mean / p50 / max) "
        "| LLM chunks | Cached | Schema fails | Retries | Escalations | Flagged blocks | Wall s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in report.arms:
        s = a.stats
        lines.append(
            f"| {_arm_label(a)} | {_pct(a.score.overall)} | {a.score.unattributed} "
            f"| {a.score.alias_drift} | {_secs(s.seconds_per_chunk)} / "
            f"{_secs(s.p50_seconds_per_chunk)} / {_secs(s.max_seconds_per_chunk)} "
            f"| {s.llm_chunks} | {s.cached_chunks} | {s.schema_failures} | {s.local_retries} "
            f"| {_escalations(s)} | {s.flagged_blocks} | {s.wall_seconds:.1f} |"
        )
    for f in report.failed:
        error = " ".join(f.error.replace("|", "/").split())
        lines.append(f"| {f.model} | FAILED: {error} |" + " |" * 10)
    if report.gold_conflicts:
        lines += ["", "## WARNING: gold speakers that look like one person", ""]
        lines += [f"- {describe_conflict(c)}" for c in report.gold_conflicts]
    types = sorted({t for a in report.arms for t in a.score.by_quote_type})
    if types:
        lines += ["", "## Accuracy by quote type", ""]
        lines.append("| Arm | " + " | ".join(types) + " |")
        lines.append("|---|" + "---|" * len(types))
        for a in report.arms:
            cells = [_pct(a.score.by_quote_type.get(t, Bucket())) for t in types]
            lines.append(f"| {a.model_id} | " + " | ".join(cells) + " |")
    chapters = sorted({c for a in report.arms for c in a.score.by_chapter})
    if chapters:
        lines += ["", "## Accuracy by chapter", ""]
        lines.append("| Arm | " + " | ".join(f"ch {c}" for c in chapters) + " |")
        lines.append("|---|" + "---|" * len(chapters))
        for a in report.arms:
            cells = [_pct(a.score.by_chapter.get(c, Bucket())) for c in chapters]
            lines.append(f"| {a.model_id} | " + " | ".join(cells) + " |")
    for a in report.arms:
        lines += ["", f"## {a.model_id}", ""]
        if a.score.confusions:
            lines += ["Top mistakes (gold → predicted):", ""]
            lines += [f"- {c.gold} → {c.predicted}: {c.count}" for c in a.score.confusions]
        else:
            lines.append("No mistakes.")
        if a.score.drift:
            lines += ["", "Alias drift (several predicted characters for one gold speaker):", ""]
            lines += [
                f"- {d.gold_speaker}: {', '.join(d.predicted_characters)}" for d in a.score.drift
            ]
        if a.score.ambiguous:
            lines += [
                "",
                "Ambiguous characters (match several gold speakers equally, scored wrong; if "
                "those gold speakers are one person, merge them in Review and re-export):",
                "",
            ]
            lines += [
                f"- {m.name} [{m.character_id}] → {', '.join(m.ambiguous_with)}"
                for m in a.score.ambiguous
            ]
        if a.score.unaligned:
            lines += ["", f"{a.score.unaligned} gold quote(s) could not be aligned (stale gold?)."]
    return "\n".join(lines) + "\n"


def _arm_label(a: ArmResult) -> str:
    return f"{a.provider_id}/{a.model_id}" + (" (hybrid)" if a.hybrid else "")


def _escalations(s: RunStats) -> str:
    if s.cached_escalations:
        return f"{s.escalations} (+{s.cached_escalations} cached)"
    return str(s.escalations)


def console_summary(report: EvalReport) -> list[str]:
    out = [f"{report.book_id}: {report.gold_quotes} gold quotes, chapters {report.chapters}"]
    for a in sorted(report.arms, key=lambda a: -(a.score.overall.accuracy or 0.0)):
        s = a.stats
        out.append(
            f"  {a.model_id + (' (hybrid)' if a.hybrid else ''):<24} "
            f"acc {_pct(a.score.overall):<22} drift {a.score.alias_drift:<3} "
            f"unattributed {a.score.unattributed:<4} s/chunk {_secs(s.seconds_per_chunk):<7} "
            f"schema-fails {s.schema_failures} retries {s.local_retries} "
            f"escalations {_escalations(s)} flagged {s.flagged_blocks}"
        )
        if a.score.ambiguous:
            names = ", ".join(m.name for m in a.score.ambiguous)
            out.append(f"    {len(a.score.ambiguous)} ambiguous character(s): {names}")
    for f in report.failed:
        out.append(f"  {f.model:<24} FAILED: {f.error}")
    for c in report.gold_conflicts:
        out.append(f"  WARNING: {describe_conflict(c)}")
    return out


def load_effective_report(book_dir: Path) -> tuple[AttributionReport, list[str]]:
    """The book's EFFECTIVE report (raw attribution.json + Review edits) — read-only."""
    return attribution_svc.load_report(book_dir)
