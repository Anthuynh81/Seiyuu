"""Durable job store: SQLite (WAL) at ``data/jobs.db``, the server's job ledger.

The pipeline was built CLI-synchronous — progress lives in the terminal and a crash is
visible in the console. A server needs those facts to survive process death, so every job
(one stage run for one book) gets a row here: ``queued -> running -> succeeded | failed |
canceled``. Transitions are enforced with guarded UPDATEs (``WHERE state IN (...)``) so a
racing writer can never resurrect a terminal row. Cancellation is cooperative:
``request_cancel`` flips a flag the runner's token polls (M6a-3); a still-queued job
cancels immediately. Because of that, cancel-while-queued is a ROUTINE event, not an
error — the runner claims work via ``try_mark_running`` (returns ``None`` on a lost
claim) instead of the raising ``mark_running``. Genuine misuse raises typed errors
(``JobNotFoundError``, ``IllegalTransitionError``) so callers never sniff messages.
``reconcile_startup`` runs once at server startup and terminates whatever a dead process
left behind, so the UI never shows a ghost job.

Connections are per-call and short-lived, so a single ``JobStore`` is safe to share across
the runner thread and API request threads; WAL keeps readers unblocked during writes.
Unlike ``books/{id}/attribution.db`` (a per-book cache) this DB is global server state,
which is why it lives under ``data_dir``, not next to book artifacts.
"""

import secrets
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from seiyuu.repository.books import RepositoryError

JOBS_DB_NAME = "jobs.db"


class JobNotFoundError(RepositoryError):
    """No row for this job id — a stale handle or caller bug, never a benign race."""


class JobKind(StrEnum):
    INGEST = "ingest"
    ATTRIBUTE = "attribute"
    RENDER = "render"
    ASSEMBLE = "assemble"
    MASTER = "master"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED})

_ALLOWED_FROM: dict[JobState, tuple[JobState, ...]] = {
    JobState.RUNNING: (JobState.QUEUED,),
    JobState.SUCCEEDED: (JobState.RUNNING,),
    JobState.FAILED: (JobState.RUNNING,),
    JobState.CANCELED: (JobState.QUEUED, JobState.RUNNING),
}


class IllegalTransitionError(RepositoryError):
    """A guarded transition matched no row: the job was not in an allowed source state.

    ``current`` is a snapshot read after the failed UPDATE — under a live race it can
    differ from the state that actually blocked the transition; treat it as diagnostic.
    """

    def __init__(self, job_id: str, current: "JobState", target: "JobState") -> None:
        super().__init__(f"job {job_id!r}: illegal transition {current.value} -> {target.value}")
        self.job_id = job_id
        self.current = current
        self.target = target


class Job(BaseModel):
    job_id: str
    book_id: str
    kind: JobKind
    state: JobState
    progress_text: str = ""
    error: str | None = None
    cancel_requested: bool = False
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    book_id          TEXT NOT NULL,
    kind             TEXT NOT NULL,
    state            TEXT NOT NULL,
    progress_text    TEXT NOT NULL DEFAULT '',
    error            TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT
);
CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs (state);
CREATE INDEX IF NOT EXISTS jobs_by_book ON jobs (book_id, created_at);
"""


def _now() -> str:
    # Microsecond ISO-8601 UTC: sorts lexicographically == chronologically.
    return datetime.now(UTC).isoformat()


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        book_id=row["book_id"],
        kind=row["kind"],
        state=row["state"],
        progress_text=row["progress_text"],
        error=row["error"],
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def default_jobs_db_path() -> Path:
    """The server-wide jobs DB under ``data_dir`` (lazy settings import, like books.py)."""
    from seiyuu.settings import get_settings

    return get_settings().data_dir / JOBS_DB_NAME


class JobStore:
    """Repository over the global jobs DB. Instances hold no connection — share freely."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()  # skipped on exception; close() then discards the transaction
        finally:
            conn.close()

    # -- creation / reads ------------------------------------------------------------

    def create(self, book_id: str, kind: JobKind | str) -> Job:
        job_id = secrets.token_hex(8)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, book_id, kind, state, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, book_id, JobKind(kind).value, JobState.QUEUED.value, _now()),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> Job:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(f"job {job_id!r} not found")
        return _row_to_job(row)

    def list_jobs(
        self,
        *,
        book_id: str | None = None,
        states: Sequence[JobState | str] | None = None,
        limit: int | None = None,
    ) -> list[Job]:
        """Jobs newest-first, optionally filtered by book and/or a set of states."""
        clauses: list[str] = []
        params: list = []
        if book_id is not None:
            clauses.append("book_id=?")
            params.append(book_id)
        if states:
            values = [JobState(s).value for s in states]
            clauses.append(f"state IN ({','.join('?' * len(values))})")
            params.extend(values)
        sql = "SELECT * FROM jobs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_job(r) for r in rows]

    # -- state transitions -----------------------------------------------------------

    def mark_running(self, job_id: str) -> Job:
        """Raising claim — use only where losing the claim IS a bug (prefer
        :meth:`try_mark_running` in the runner)."""
        return self._transition(job_id, JobState.RUNNING, set_started=True)

    def try_mark_running(self, job_id: str) -> Job | None:
        """Atomically claim a queued job; ``None`` if the claim lost — the job is no
        longer queued, typically because it was canceled while waiting in the runner's
        queue. Cancel-while-queued is a routine user action, so the M6a-3 runner claims
        through this and treats ``None`` as "skip". An unknown id still raises
        ``JobNotFoundError``: a stale handle is a bug, a lost claim is not."""
        if self._guarded_update(job_id, JobState.RUNNING, set_started=True):
            return self.get(job_id)
        self.get(job_id)  # bogus id -> JobNotFoundError; otherwise the claim just lost
        return None

    def finish(self, job_id: str, state: JobState, *, error: str | None = None) -> Job:
        if state not in TERMINAL_STATES:
            raise ValueError(f"finish() requires a terminal state, got {state!r}")
        if state is JobState.FAILED and not error:
            raise ValueError("a failed job must carry an error message")
        return self._transition(job_id, state, error=error, set_finished=True)

    def _guarded_update(
        self,
        job_id: str,
        target: JobState,
        *,
        error: str | None = None,
        set_started: bool = False,
        set_finished: bool = False,
    ) -> bool:
        """One atomic guarded UPDATE; True iff the row was in an allowed source state."""
        allowed = _ALLOWED_FROM[target]
        sets = ["state=?"]
        params: list = [target.value]
        if error is not None:
            sets.append("error=?")
            params.append(error)
        if set_started:
            sets.append("started_at=?")
            params.append(_now())
        if set_finished:
            sets.append("finished_at=?")
            params.append(_now())
        params.extend([job_id, *(s.value for s in allowed)])
        with self._connect() as conn:
            changed = conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} "
                f"WHERE job_id=? AND state IN ({','.join('?' * len(allowed))})",
                params,
            ).rowcount
        return changed > 0

    def _transition(self, job_id: str, target: JobState, **kwargs) -> Job:
        if not self._guarded_update(job_id, target, **kwargs):
            current = self.get(job_id).state  # raises JobNotFoundError for a bogus id
            raise IllegalTransitionError(job_id, current, target)
        return self.get(job_id)

    # -- cancellation ----------------------------------------------------------------

    def request_cancel(self, job_id: str) -> Job:
        """Cooperative cancel: a queued job cancels immediately; a running job only gets
        the flag — the runner's token (M6a-3) sees it and finishes ``canceled`` at its
        next checkpoint. Idempotent no-op on terminal jobs."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET cancel_requested=1 WHERE job_id=? AND state IN (?, ?)",
                (job_id, JobState.QUEUED.value, JobState.RUNNING.value),
            )
            conn.execute(
                "UPDATE jobs SET state=?, finished_at=? WHERE job_id=? AND state=?",
                (JobState.CANCELED.value, _now(), job_id, JobState.QUEUED.value),
            )
        return self.get(job_id)

    def cancel_requested(self, job_id: str) -> bool:
        """The cheap poll the runner's cooperative cancel token calls between segments."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(f"job {job_id!r} not found")
        return bool(row["cancel_requested"])

    # -- progress / startup ----------------------------------------------------------

    def update_progress(self, job_id: str, text: str) -> None:
        """Best-effort: guarded to ``running`` so a tick racing a cancel/finish is
        dropped rather than mutating a terminal row."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress_text=? WHERE job_id=? AND state=?",
                (text, job_id, JobState.RUNNING.value),
            )

    def reconcile_startup(self) -> int:
        """Terminate every job a dead process left behind; call once at server startup,
        before the runner accepts work. A ``running`` row with a pending cancel request
        honors the user's intent and cancels; other ``running`` rows were interrupted
        mid-stage and fail (partial artifacts are safe: durable writes are atomic and
        expensive calls are cached, so a re-run resumes cheaply); ``queued`` rows lived
        in the dead process's in-memory queue and will never start, so they cancel.
        Returns the number of rows reconciled."""
        now = _now()
        with self._connect() as conn:
            cancel_honored = conn.execute(
                "UPDATE jobs SET state=?, error=?, finished_at=? "
                "WHERE state=? AND cancel_requested=1",
                (
                    JobState.CANCELED.value,
                    "canceled: server stopped before the cancellation completed",
                    now,
                    JobState.RUNNING.value,
                ),
            ).rowcount
            failed = conn.execute(
                "UPDATE jobs SET state=?, error=?, finished_at=? WHERE state=?",
                (
                    JobState.FAILED.value,
                    "interrupted: server stopped mid-job",
                    now,
                    JobState.RUNNING.value,
                ),
            ).rowcount
            canceled = conn.execute(
                "UPDATE jobs SET state=?, error=?, finished_at=? WHERE state=?",
                (
                    JobState.CANCELED.value,
                    "server stopped before the job started",
                    now,
                    JobState.QUEUED.value,
                ),
            ).rowcount
        return cancel_honored + failed + canceled
