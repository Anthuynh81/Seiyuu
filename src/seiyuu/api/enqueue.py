"""The enqueue ladder (scoping doc section 2), shared by every job-creating route.

Runs under the process-wide enqueue mutex so the dedupe check-then-act cannot race a
concurrent enqueue (the audition busy-check shares the same mutex in M6b-6). The store
itself does not dedupe; single-process deployment is what makes this sound.
"""

import threading

from seiyuu.api.errors import ApiError
from seiyuu.api.schemas import JobOut
from seiyuu.jobs import JobRunner
from seiyuu.repository import Job, JobKind, JobState, JobStore


def enqueue_job(
    *,
    store: JobStore,
    runner: JobRunner,
    mutex: threading.Lock,
    book_id: str,
    kind: JobKind,
    params: dict | None = None,
) -> Job:
    """Dedupe per (book_id, kind) against live rows, then enqueue. Raises ApiError
    409 ``duplicate_job`` (detail = the full existing Job, so the UI links straight to
    it) or 503 ``not_ready`` when the runner isn't accepting work."""
    with mutex:
        live = store.list_jobs(book_id=book_id, states=[JobState.QUEUED, JobState.RUNNING])
        existing = next((j for j in live if j.kind is kind), None)
        if existing is not None:
            raise ApiError(
                409,
                "duplicate_job",
                f"a {kind.value} job for {book_id!r} is already {existing.state.value}",
                detail=JobOut.from_job(existing).model_dump(mode="json"),
            )
        try:
            return runner.enqueue(book_id, kind, params=params)
        except RuntimeError as exc:  # runner not started / already stopped
            raise ApiError(503, "not_ready", str(exc)) from exc
