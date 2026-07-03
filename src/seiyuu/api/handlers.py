"""Job handlers: thin adapters from :class:`JobContext` to the stage services.

Each handler re-parses ``Job.params`` with the SAME pydantic model the route validated
(a mismatch is a bug and fails the job loudly), runs inside the heavy-work gate (one
GPU-heavy activity per process — auditions refuse instead of colliding), and threads
``ctx.progress`` / ``ctx.check_cancel`` into the stage loops. Resource lifecycles live
in the services (``run_attribution`` owns the GPU hold + Ollama unload-in-finally);
handlers add only what is server-specific. The RENDER handler arrives with the money
gate in M6b-5 — until then nothing can enqueue a render job.
"""

from collections.abc import Mapping
from pathlib import Path

from seiyuu.api.concurrency import HeavyWorkGate
from seiyuu.api.registry import EngineRegistry
from seiyuu.api.schemas import (
    AssembleParams,
    AttributeParams,
    LoudnessWrite,
    MasterParams,
    PauseWrite,
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
    cfg: Settings, registry: EngineRegistry, gate: HeavyWorkGate
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
                chapters=tuple(params.chapters),
                progress=ctx.progress,
                check_cancel=ctx.check_cancel,
            )

    def assemble(ctx: JobContext) -> None:
        from seiyuu.assemble import assemble_book

        params = AssembleParams.model_validate(ctx.job.params or {})
        with gate.hold("job"):
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
        with gate.hold("job"):
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
        JobKind.ASSEMBLE: assemble,
        JobKind.MASTER: master,
    }
