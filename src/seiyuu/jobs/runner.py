"""Single-flight job runner: one daemon worker thread, one job at a time.

One-at-a-time IS the GPU discipline at the job level — a single consumer GPU cannot host
two heavy stages, so the queue is global, not per-book. The runner is deliberately
generic: handlers are injected per ``JobKind`` (the API wires real stage services; tests
use fakes) and each handler receives a :class:`JobContext` whose ``check_cancel`` raises
:class:`JobCanceled` when the job's durable cancel flag is set — the stage loops call it
between chunks/segments/chapters. Handlers own their resource lifecycles (e.g. the
attribution handler must ``provider.unload()`` in a ``finally``); the runner guarantees
those ``finally`` blocks run on cancel and failure alike, then settles the job row
exactly once: succeeded / failed(error) / canceled.

``start()`` reconciles the store FIRST (rows a dead process left behind terminate) and
only then accepts work, so enqueue-before-start raises rather than creating rows the
reconcile would eat. Jobs still queued when the process dies are canceled by the next
startup's reconcile; the worker claims each dequeued id through ``try_mark_running`` so
a job canceled while waiting is a silent skip, never an exception.
"""

import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from seiyuu.repository import Job, JobKind, JobNotFoundError, JobState, JobStore


class JobCanceled(Exception):
    """Raised by a cooperative checkpoint when the job's cancel flag is set."""


@dataclass(frozen=True)
class JobContext:
    """What a handler gets: the claimed job, a progress sink, and the cancel checkpoint."""

    job: Job
    progress: Callable[[str], None]
    check_cancel: Callable[[], None]


JobHandler = Callable[[JobContext], None]


class _StoreCancelToken:
    """Checkpoint bound to one job: polls the durable flag, raises JobCanceled when set."""

    __slots__ = ("_store", "_job_id")

    def __init__(self, store: JobStore, job_id: str) -> None:
        self._store = store
        self._job_id = job_id

    def check(self) -> None:
        if self._store.cancel_requested(self._job_id):
            raise JobCanceled(f"job {self._job_id!r}: cancellation requested")


class JobRunner:
    """Global single-flight executor over a :class:`JobStore`."""

    def __init__(
        self,
        store: JobStore,
        handlers: Mapping[JobKind, JobHandler],
        *,
        poll_seconds: float = 0.2,
    ) -> None:
        self._store = store
        self._handlers = dict(handlers)
        self._poll_seconds = poll_seconds
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_job_id: str | None = None

    @property
    def active_job_id(self) -> str | None:
        """The job currently executing, if any (an instantaneous snapshot)."""
        return self._active_job_id

    def start(self) -> int:
        """Reconcile the store, then start the worker. Returns the rows reconciled."""
        if self._stop.is_set():
            raise RuntimeError("job runner was stopped; construct a new one")
        if self._thread is not None:
            raise RuntimeError("job runner already started")
        reconciled = self._store.reconcile_startup()
        self._thread = threading.Thread(target=self._run, name="seiyuu-job-runner", daemon=True)
        self._thread.start()
        return reconciled

    def enqueue(self, book_id: str, kind: JobKind | str, *, params: dict | None = None) -> Job:
        """Create a queued job row and hand it to the worker."""
        if self._thread is None or self._stop.is_set():
            raise RuntimeError("job runner is not running; start() it before enqueueing")
        job = self._store.create(book_id, kind, params=params)
        self._queue.put(job.job_id)
        return job

    def request_cancel(self, job_id: str) -> Job:
        """Cooperative cancel (see JobStore.request_cancel); the worker honors it at the
        stage's next checkpoint, or skips the job entirely if it never started."""
        return self._store.request_cancel(job_id)

    def stop(self, *, cancel_pending: bool = True, timeout: float = 10.0) -> bool:
        """Stop accepting work and wait for the worker to exit; True if it did.

        ``cancel_pending`` sweeps the STORE (not an in-memory snapshot, which would race
        the worker's claim of a just-dequeued job): every queued row cancels immediately
        and every running row gets the cooperative flag, so whichever side of the claim
        the worker is on, the job either skips or cancels at its next checkpoint. An
        ill-behaved handler that never checkpoints can outlive the timeout — then this
        returns False and the daemon thread dies with the process; the next startup's
        reconcile settles its row. With ``cancel_pending=False`` outstanding rows are
        left as-is and reconcile the same way."""
        self._stop.set()
        if cancel_pending:
            for job in self._store.list_jobs(states=[JobState.QUEUED, JobState.RUNNING]):
                self._store.request_cancel(job.job_id)
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=self._poll_seconds)
            except queue.Empty:
                continue
            try:
                self._execute(job_id)
            except Exception:  # noqa: BLE001 — the worker must outlive a failing ledger
                # Settling the row itself failed (e.g. disk trouble). Keep the worker
                # alive for later jobs; the stuck row settles at the next reconcile.
                continue

    def _execute(self, job_id: str) -> None:
        if self._stop.is_set():
            return  # dequeued but not claimed: the row stays queued for reconcile/sweep
        try:
            job = self._store.try_mark_running(job_id)
        except JobNotFoundError:
            return  # row vanished underneath us; nothing to run
        if job is None:
            return  # canceled while queued — routine skip, not an error
        handler = self._handlers.get(job.kind)
        self._active_job_id = job_id
        try:
            if handler is None:
                raise RuntimeError(f"no handler registered for job kind {job.kind.value!r}")
            handler(
                JobContext(
                    job=job,
                    progress=lambda text: self._store.update_progress(job_id, text),
                    check_cancel=_StoreCancelToken(self._store, job_id).check,
                )
            )
        except JobCanceled:
            self._store.finish(job_id, JobState.CANCELED)
        except Exception as exc:
            self._store.finish(job_id, JobState.FAILED, error=f"{type(exc).__name__}: {exc}")
        else:
            self._store.finish(job_id, JobState.SUCCEEDED)
        finally:
            self._active_job_id = None
