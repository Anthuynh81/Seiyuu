"""Job handlers: thin adapters from :class:`JobContext` to the stage services.

Each handler re-parses ``Job.params`` with the SAME pydantic model the route validated
(a mismatch is a bug and fails the job loudly), runs inside the heavy-work gate (one
GPU-heavy activity per process — auditions refuse instead of colliding), and threads
``ctx.progress`` / ``ctx.check_cancel`` into the stage loops. Resource lifecycles live
in the services (``run_attribution`` owns the GPU hold + Ollama unload-in-finally);
handlers add only what is server-specific — for RENDER, that is the money gate's
consume-at-job-start verify (sign-off Q5).
"""

from collections.abc import Mapping
from pathlib import Path

from seiyuu.api.concurrency import BorrowBroker, HeavyWorkGate
from seiyuu.api.registry import EngineRegistry
from seiyuu.api.schemas import (
    AssembleParams,
    AttributeParams,
    LoudnessWrite,
    MasterParams,
    PauseWrite,
    RenderParams,
    WarmupParams,
)
from seiyuu.gpu import get_gpu_manager
from seiyuu.ingest.models import NormalizedBook
from seiyuu.jobs import JobContext, JobHandler
from seiyuu.repository import JobKind
from seiyuu.repository.books import NORMALIZED_NAME
from seiyuu.services import ServiceError, run_attribution
from seiyuu.settings import Settings

COVER_NAMES = ("cover.jpg", "cover.png")


def _load_normalized(cfg: Settings, book_id: str) -> NormalizedBook:
    path = cfg.books_dir / book_id / NORMALIZED_NAME
    if not path.is_file():
        raise ServiceError(f"no normalized book at {path}; run ingest first")
    return NormalizedBook.model_validate_json(path.read_text(encoding="utf-8"))


def _pause_profile(write: PauseWrite | None):
    """PauseWrite -> PauseProfile: None = the profile's default, explicit 0.0 honored
    (`is None` checks, not `or` — the falsy-zero fix the scoping doc calls out)."""
    from seiyuu.assemble import PauseProfile

    defaults = PauseProfile()
    if write is None:
        return defaults
    return PauseProfile(
        paragraph=defaults.paragraph if write.paragraph is None else write.paragraph,
        after_heading=(
            defaults.after_heading if write.after_heading is None else write.after_heading
        ),
        scene_break=defaults.scene_break if write.scene_break is None else write.scene_break,
        dialogue=defaults.dialogue if write.dialogue is None else write.dialogue,
        chapter_lead_in=(
            defaults.chapter_lead_in if write.chapter_lead_in is None else write.chapter_lead_in
        ),
        chapter_lead_out=(
            defaults.chapter_lead_out if write.chapter_lead_out is None else write.chapter_lead_out
        ),
    )


def _loudness_target(cfg: Settings, write: LoudnessWrite | None):
    """LoudnessWrite -> LoudnessTarget | None (None = normalization disabled)."""
    from seiyuu.assemble import LoudnessTarget

    write = write or LoudnessWrite()
    enabled = cfg.loudness_enabled if write.enabled is None else write.enabled
    if not enabled:
        return None
    target = cfg.loudness_target_lufs if write.target_lufs is None else write.target_lufs
    return LoudnessTarget(i=target, tp=cfg.loudness_true_peak, lra=cfg.loudness_range)


def _find_cover(book_output_dir: Path) -> Path | None:
    for name in COVER_NAMES:
        candidate = book_output_dir / name
        if candidate.is_file():
            return candidate
    return None


def build_handlers(
    cfg: Settings,
    registry: EngineRegistry,
    gate: HeavyWorkGate,
    broker: BorrowBroker | None = None,
) -> Mapping[JobKind, JobHandler]:
    def warmup(ctx: JobContext) -> None:
        params = WarmupParams.model_validate(ctx.job.params or {})
        engine = registry.get(params.engine_id)
        gpu = get_gpu_manager()
        with gate.hold("job"):
            ctx.check_cancel()
            ctx.progress(f"loading {params.engine_id} weights (downloads on first use)…")
            try:
                with gpu.acquire(engine, f"engine:{params.engine_id}"):
                    engine.warm()
            except BaseException:
                # A failed/canceled load must not read as resident (registry residency
                # is identity vs the manager). free_all() only AFTER acquire released
                # the manager lock — it holds it for the whole with-body.
                gpu.free_all()
                raise
            # Lazy release by design: the model STAYS resident so the next audition's
            # re-acquire is an identity no-op — that is the whole point of warmup.
            ctx.progress(f"{params.engine_id} resident")

    def attribute(ctx: JobContext) -> None:
        params = AttributeParams.model_validate(ctx.job.params or {})
        book = _load_normalized(cfg, ctx.job.book_id)
        with gate.hold("job"):
            # run_attribution owns provider build, the whole-run GPU hold (skipped for
            # cloud providers), the chunk cache, and Ollama unload in finally.
            run_attribution(
                book,
                cfg.books_dir / ctx.job.book_id,
                cfg=cfg,
                provider_id=params.provider,
                model=params.model,
                prompt_version=params.prompt_version,
                use_hybrid=params.use_hybrid,
                emit_thoughts=params.emit_thoughts,
                chapters=tuple(params.chapters),
                progress=ctx.progress,
                check_cancel=ctx.check_cancel,
            )

    def render(ctx: JobContext) -> None:
        """Verify-then-render. The FRESH estimate + verify_quote(consume=True) run
        immediately before synthesis — consumption is the LAST step of verification, so
        a refusal (drift, expiry-while-queued, reuse) fails the job with the verbatim
        gate reason in Job.error and never burns the token; a crash AFTER consumption
        requires a re-estimate, and the refusal message says so."""
        from seiyuu.api.money import compute_estimate, resolve_single
        from seiyuu.render.gate import CostQuote, verify_quote
        from seiyuu.render.pipeline import render_book, render_book_multivoice
        from seiyuu.services import load_assignment, load_report
        from seiyuu.validate import Validator
        from seiyuu.voices import VoiceLibrary

        params = RenderParams.model_validate(ctx.job.params or {})
        book_id = ctx.job.book_id
        book = _load_normalized(cfg, book_id)
        book_output_dir = cfg.output_dir / book_id
        library = VoiceLibrary(cfg.voices_dir)
        chapters = tuple(sorted(set(params.chapters)))
        single = resolve_single(cfg, params.single) if params.mode == "single" else None
        with gate.hold("job"):
            ctx.check_cancel()
            ctx.progress("verifying cost approval…")
            est_ctx = compute_estimate(
                cfg, registry, book, book_id, mode=params.mode, chapters=chapters, single=single
            )
            # The estimate walks every block (seconds on a big book): a cancel filed in
            # that window must land BEFORE consumption, or a job that synthesizes
            # nothing burns the user's single-use approval.
            ctx.check_cancel()
            approved_usd: float | None = None
            if est_ctx.est.total_usd > 0:
                if not params.cost_token:
                    # Unreachable via the API (the enqueue dry-run requires a token);
                    # cache eviction between enqueue and start can still land here.
                    raise ServiceError(
                        f"render now bills ${est_ctx.est.total_usd:.2f} but the job "
                        "carries no cost token; re-run estimate-cost and quote"
                    )
                quote = CostQuote.decode(params.cost_token)
                verify_quote(
                    quote,
                    book_id=book_id,
                    chapters=chapters,
                    fingerprint=est_ctx.est.fingerprint,
                    assignment_hash=est_ctx.assignment_hash,
                    recomputed_total_usd=est_ctx.est.total_usd,
                    max_usd=cfg.render_max_usd,
                    data_dir=cfg.data_dir,
                    consume=True,  # burned only by a job that actually starts rendering
                )
                approved_usd = quote.total_usd  # the hard cumulative spend cap
            validator = Validator(
                model_size=cfg.validation_model_size,
                device=cfg.whisper_device,
                compute_type=cfg.validation_compute_type,
                min_ratio=cfg.validation_min_ratio,
            )
            if params.mode == "multivoice":
                report, _warnings = load_report(cfg.books_dir / book_id)
                assignment = load_assignment(cfg.output_dir, book_id)
                render_book_multivoice(
                    report,
                    book,
                    library,
                    assignment,
                    book_output_dir,
                    chapters=chapters,
                    progress=ctx.progress,
                    validator=validator,
                    validation_max_retries=cfg.validation_max_retries,
                    allow_paid=approved_usd is not None,
                    max_paid_usd=approved_usd,
                    cloud_max_slots=cfg.elevenlabs_max_voice_slots,
                    check_cancel=ctx.check_cancel,
                    broker=broker,  # lend the resident engine to auditions between segments
                )
            else:
                render_book(
                    book,
                    registry.get(single.engine_id),  # shared instance: warm re-acquire
                    single.voice_id,
                    book_output_dir,
                    settings=single.settings,
                    seed=single.seed,
                    chapters=chapters,
                    progress=ctx.progress,
                    library=library,  # the clone consent gate is never skipped
                    validator=validator,
                    validation_max_retries=cfg.validation_max_retries,
                    allow_paid=approved_usd is not None,
                    max_paid_usd=approved_usd,
                    check_cancel=ctx.check_cancel,
                    broker=broker,  # lend the resident engine to auditions between segments
                )

    # assemble/master deliberately do NOT hold the heavy-work gate: they are pure
    # ffmpeg/CPU stages, and holding it would make a running assemble refuse auditions
    # with a phantom "audition in flight" (the refusal predicate rightly allows them).
    def assemble(ctx: JobContext) -> None:
        from seiyuu.assemble import assemble_book

        params = AssembleParams.model_validate(ctx.job.params or {})
        assemble_book(
            cfg.output_dir / ctx.job.book_id,
            pauses=_pause_profile(params.pauses),
            loudness=_loudness_target(cfg, params.loudness),
            progress=ctx.progress,
            check_cancel=ctx.check_cancel,
        )

    def master(ctx: JobContext) -> None:
        from seiyuu.assemble import master_book

        params = MasterParams.model_validate(ctx.job.params or {})
        book_output_dir = cfg.output_dir / ctx.job.book_id
        master_book(
            book_output_dir,
            pauses=_pause_profile(params.pauses),
            loudness=_loudness_target(cfg, params.loudness),
            cover=_find_cover(book_output_dir) if params.use_cover else None,
            bitrate=params.bitrate,
            target_seconds=(
                params.target_minutes * 60.0 if params.target_minutes is not None else None
            ),
            tempo_bounds=(cfg.tempo_min, cfg.tempo_max),
            progress=ctx.progress,
            check_cancel=ctx.check_cancel,
        )

    return {
        JobKind.WARMUP: warmup,
        JobKind.ATTRIBUTE: attribute,
        JobKind.RENDER: render,
        JobKind.ASSEMBLE: assemble,
        JobKind.MASTER: master,
    }
