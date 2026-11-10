"""Qwen-first casting, step 1: the per-book cast plan (design/qwen-first-casting-2026-11-08.md).

POST /books/{id}/cast-plan/describe   draft every character's voice description (a job)
GET  /books/{id}/cast-plan            the plan (404 until drafted)
PUT  /books/{id}/cast-plan            Review-table edits (description, sample line, same voice
                                      as, keep current voice, gender, age) and the book's
                                      delivery direction (three instruction texts, on/off)
POST /books/{id}/cast-plan/design     design every row's voice and assign them (a GPU job)
POST /books/{id}/cast-plan/rows/{character_id}/take   use another of a row's takes (no GPU)

The plan's rows are a working file for the Review table and the design-cast job; they never
change an assignment, a voice, or a cache key by themselves. Its ``delivery`` block is different:
the multivoice render reads it, and for a Breeze voice the instruction is part of each line's
cache key, so an edit re-records the lines it directs. That is why the PUT is refused while a
render is queued or running.
"""

from fastapi import APIRouter, Request, Response

from seiyuu.api.deps import RunnerDep, SettingsDep, StoreDep
from seiyuu.api.enqueue import enqueue_job
from seiyuu.api.errors import ApiError
from seiyuu.api.routes.common import effective_report, guard_render_active, status_or_404
from seiyuu.api.schemas import (
    CastDesignedOut,
    CastPlanOut,
    CastPlanUpdate,
    CastTakeRequest,
    DescribeCastParams,
    DescribeCastRequest,
    JobOut,
)
from seiyuu.attribute.providers.local import OllamaPreflightError, ollama_preflight
from seiyuu.repository import JobKind
from seiyuu.services import ServiceError
from seiyuu.services.cast_plan import apply_delivery, apply_updates, load_plan, save_plan
from seiyuu.services.llm_advisory import resolve_advisory

router = APIRouter(tags=["cast-plan"])


def _plan_or_404(cfg, book_id: str):
    try:
        plan = load_plan(cfg.output_dir, book_id)
    except ValueError as exc:  # pydantic ValidationError: a corrupt file
        raise ApiError(500, "corrupt_artifact", f"cast_plan.json for {book_id!r}: {exc}") from exc
    if plan is None:
        raise ApiError(404, "not_found", f"no cast plan for {book_id!r} yet; draft one first")
    return plan


@router.post("/books/{book_id}/cast-plan/describe", response_model=JobOut, status_code=202)
def describe_cast(
    book_id: str,
    body: DescribeCastRequest,
    request: Request,
    response: Response,
    cfg: SettingsDep,
    store: StoreDep,
    runner: RunnerDep,
) -> JobOut:
    """Queue the describer. Local (default) is free; anthropic is paid and needs confirm_paid.
    An unreachable Ollama or an unpulled model is a 503 with the fix now, not a failed job."""
    status = status_or_404(cfg, book_id)
    effective_report(cfg, book_id, status)  # 404 not attributed / 500 corrupt
    resolved = resolve_advisory(cfg, cfg.describe_provider, cfg.describe_model, None)
    if resolved.is_paid:
        if not body.confirm_paid:
            raise ApiError(
                402,
                "payment_confirmation_required",
                "describing the cast with describe_provider=anthropic calls the paid Anthropic "
                "API; re-send with confirm_paid=true to approve the spend",
            )
        if not cfg.anthropic_api_key:
            raise ApiError(503, "not_ready", "ANTHROPIC_API_KEY not set; required to describe")
    else:
        try:
            ollama_preflight(cfg.ollama_base_url, [resolved.model])
        except OllamaPreflightError as exc:
            raise ApiError(
                503, exc.code, str(exc), detail={"base_url": exc.base_url, "model": exc.model}
            ) from exc
    job = enqueue_job(
        store=store,
        runner=runner,
        mutex=request.app.state.enqueue_mutex,
        book_id=book_id,
        kind=JobKind.DESCRIBE_CAST,
        params=DescribeCastParams(confirm_paid=body.confirm_paid).model_dump(),
    )
    response.headers["Location"] = f"/api/jobs/{job.job_id}"
    return JobOut.from_job(job)


@router.get("/books/{book_id}/cast-plan", response_model=CastPlanOut)
def get_cast_plan(book_id: str, cfg: SettingsDep) -> CastPlanOut:
    status_or_404(cfg, book_id)
    return CastPlanOut(**_plan_or_404(cfg, book_id).model_dump())


def _refuse_while_casting(store, book_id: str) -> None:
    live = store.list_jobs(book_id=book_id)
    for job in live:
        if job.kind in (JobKind.DESCRIBE_CAST, JobKind.DESIGN_CAST) and not job.is_terminal:
            raise ApiError(
                409,
                "conflict",
                f"a {job.kind.value} job for this book is still {job.state.value}; wait for it",
                detail=JobOut.from_job(job).model_dump(mode="json"),
            )


@router.put("/books/{book_id}/cast-plan", response_model=CastPlanOut)
def update_cast_plan(
    book_id: str, body: CastPlanUpdate, request: Request, cfg: SettingsDep, store: StoreDep
) -> CastPlanOut:
    status_or_404(cfg, book_id)
    guard_render_active(store, book_id)
    with request.app.state.enqueue_mutex:  # never interleave with a cast job starting
        # Both cast jobs load the plan once and save it later: an edit accepted while one
        # runs would be written over (a row edit) or, for the delivery block, raced.
        _refuse_while_casting(store, book_id)
        updated = _plan_or_404(cfg, book_id)
        if body.updates:
            try:
                updated = apply_updates(updated, body.updates)
            except ServiceError as exc:
                raise ApiError(422, "invalid", str(exc)) from exc
        if body.delivery is not None:
            updated = apply_delivery(updated, body.delivery)
        save_plan(cfg.output_dir, updated)
    return CastPlanOut(**updated.model_dump())


@router.post("/books/{book_id}/cast-plan/design", response_model=JobOut, status_code=202)
def design_cast(
    book_id: str,
    request: Request,
    response: Response,
    cfg: SettingsDep,
    store: StoreDep,
    runner: RunnerDep,
) -> JobOut:
    """Queue the design-cast job: a voice per row that needs one (local Qwen3-TTS, free), then
    the book's assignment. Refused while the book renders or its cast is being described."""
    status_or_404(cfg, book_id)
    _plan_or_404(cfg, book_id)
    guard_render_active(store, book_id)
    with request.app.state.enqueue_mutex:
        _refuse_while_casting(store, book_id)
    job = enqueue_job(
        store=store,
        runner=runner,
        mutex=request.app.state.enqueue_mutex,
        book_id=book_id,
        kind=JobKind.DESIGN_CAST,
        params={},
    )
    response.headers["Location"] = f"/api/jobs/{job.job_id}"
    return JobOut.from_job(job)


@router.post("/books/{book_id}/cast-plan/rows/{character_id}/take", response_model=CastDesignedOut)
def keep_cast_take(
    book_id: str,
    character_id: str,
    body: CastTakeRequest,
    request: Request,
    cfg: SettingsDep,
    store: StoreDep,
) -> CastDesignedOut:
    """Swap a designed row to another of its takes and re-point everyone using that voice."""
    from seiyuu.services.cast_design import keep_other_take

    status_or_404(cfg, book_id)
    guard_render_active(store, book_id)
    with request.app.state.enqueue_mutex:
        _refuse_while_casting(store, book_id)
        with request.app.state.voices_mutex:
            try:
                plan, assignment = keep_other_take(cfg, book_id, character_id, body.take)
            except ServiceError as exc:
                raise ApiError(422, "invalid", str(exc)) from exc
    return CastDesignedOut(plan=CastPlanOut(**plan.model_dump()), assignment=assignment)
