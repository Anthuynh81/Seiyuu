"""Qwen-first casting, step 1: the per-book cast plan (design/qwen-first-casting-2026-11-08.md).

POST /books/{id}/cast-plan/describe   draft every character's voice description (a job)
GET  /books/{id}/cast-plan            the plan (404 until drafted)
PUT  /books/{id}/cast-plan            Review-table edits (description, sample line, same voice
                                      as, keep current voice, gender, age)

The plan is a working file for the Review table and the design-cast job; it never changes an
assignment, a voice, or a cache key by itself.
"""

from fastapi import APIRouter, Request, Response

from seiyuu.api.deps import RunnerDep, SettingsDep, StoreDep
from seiyuu.api.enqueue import enqueue_job
from seiyuu.api.errors import ApiError
from seiyuu.api.routes.common import effective_report, guard_render_active, status_or_404
from seiyuu.api.schemas import (
    CastPlanOut,
    CastPlanUpdate,
    DescribeCastParams,
    DescribeCastRequest,
    JobOut,
)
from seiyuu.attribute.providers.local import OllamaPreflightError, ollama_preflight
from seiyuu.repository import JobKind
from seiyuu.services import ServiceError
from seiyuu.services.cast_plan import apply_updates, load_plan, save_plan
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


@router.put("/books/{book_id}/cast-plan", response_model=CastPlanOut)
def update_cast_plan(
    book_id: str, body: CastPlanUpdate, request: Request, cfg: SettingsDep, store: StoreDep
) -> CastPlanOut:
    status_or_404(cfg, book_id)
    guard_render_active(store, book_id)
    with request.app.state.enqueue_mutex:  # never interleave with a draft finishing
        live = store.list_jobs(book_id=book_id)
        if any(j.kind is JobKind.DESCRIBE_CAST and not j.is_terminal for j in live):
            raise ApiError(
                409, "conflict", "the cast is still being described; edit it when that finishes"
            )
        plan = _plan_or_404(cfg, book_id)
        try:
            updated = apply_updates(plan, body.updates)
        except ServiceError as exc:
            raise ApiError(422, "invalid", str(exc)) from exc
        save_plan(cfg.output_dir, updated)
    return CastPlanOut(**updated.model_dump())
