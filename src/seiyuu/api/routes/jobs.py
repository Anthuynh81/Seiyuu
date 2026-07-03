"""GET /jobs, GET /jobs/{id} (THE progress poll), POST /jobs/{id}/cancel.

``GET /api/jobs/{job_id}`` is the single poll target (1-2s while non-terminal) — book
list/detail deliberately omit progress so they are useless as progress polls. Unknown
ids raise ``JobNotFoundError`` which the app-level handler maps to the enveloped 404.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from seiyuu.api.deps import StoreDep
from seiyuu.api.errors import ApiError
from seiyuu.api.schemas import JobOut, JobsOut
from seiyuu.repository import JobKind, JobState

router = APIRouter(tags=["jobs"])


@router.get("/jobs", response_model=JobsOut)
def list_jobs(
    store: StoreDep,
    book_id: str | None = None,
    state: Annotated[list[str], Query()] = [],  # noqa: B006 — FastAPI treats this as a default factory
    kind: Annotated[list[str], Query()] = [],  # noqa: B006
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> JobsOut:
    try:
        states = [JobState(s) for s in state]
        kinds = {JobKind(k) for k in kind}
    except ValueError as exc:
        raise ApiError(422, "invalid", str(exc)) from exc
    # Kind is filtered API-side (no store support); the limit applies AFTER it, so fetch
    # unbounded when kinds are given or the store's pre-filter limit would underfill.
    jobs = store.list_jobs(book_id=book_id, states=states or None, limit=None if kinds else limit)
    if kinds:
        jobs = [j for j in jobs if j.kind in kinds][:limit]
    return JobsOut(jobs=[JobOut.from_job(j) for j in jobs])


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, store: StoreDep) -> JobOut:
    return JobOut.from_job(store.get(job_id))


@router.post("/jobs/{job_id}/cancel", response_model=JobOut, status_code=202)
def cancel_job(job_id: str, store: StoreDep) -> JobOut:
    """Cooperative cancel: queued -> canceled immediately; running -> flag only, settled
    at the handler's next checkpoint (the UI renders running+cancel_requested as
    "canceling"). Idempotent on terminal jobs. Always 202 + the post-update snapshot."""
    return JobOut.from_job(store.request_cancel(job_id))
