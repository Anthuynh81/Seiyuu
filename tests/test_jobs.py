"""Durable job store: lifecycle, guarded transitions, cancel, reconcile (M6a commit 2)."""

import sqlite3

import pytest

from seiyuu.repository import (
    IllegalTransitionError,
    Job,
    JobKind,
    JobNotFoundError,
    JobState,
    JobStore,
    RepositoryError,
)


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "data" / "jobs.db")  # parent dir must be auto-created


# --- creation / reads ---


def test_create_starts_queued(store):
    job = store.create("book-1", JobKind.ATTRIBUTE)
    assert job.state is JobState.QUEUED
    assert job.book_id == "book-1" and job.kind is JobKind.ATTRIBUTE
    assert job.progress_text == "" and job.error is None
    assert not job.cancel_requested and not job.is_terminal
    assert job.created_at is not None
    assert job.started_at is None and job.finished_at is None


def test_create_accepts_kind_string_and_rejects_junk(store):
    assert store.create("b", "render").kind is JobKind.RENDER
    with pytest.raises(ValueError):
        store.create("b", "explode")


def test_get_roundtrip_and_unknown(store):
    job = store.create("book-1", JobKind.RENDER)
    assert store.get(job.job_id) == job
    with pytest.raises(RepositoryError, match="not found"):
        store.get("no-such-job")


def test_db_is_in_wal_mode(store):
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


# --- lifecycle transitions ---


def test_success_lifecycle(store):
    job = store.create("book-1", JobKind.RENDER)
    running = store.mark_running(job.job_id)
    assert running.state is JobState.RUNNING and running.started_at is not None

    store.update_progress(job.job_id, "chapter 2/10")
    assert store.get(job.job_id).progress_text == "chapter 2/10"

    done = store.finish(job.job_id, JobState.SUCCEEDED)
    assert done.state is JobState.SUCCEEDED and done.is_terminal
    assert done.finished_at is not None and done.error is None
    assert done.progress_text == "chapter 2/10"  # last progress survives


def test_mark_running_requires_queued(store):
    job = store.create("b", JobKind.ASSEMBLE)
    store.mark_running(job.job_id)
    with pytest.raises(RepositoryError, match="illegal transition running -> running"):
        store.mark_running(job.job_id)


def test_finish_succeeded_straight_from_queued_is_illegal(store):
    job = store.create("b", JobKind.MASTER)
    with pytest.raises(RepositoryError, match="illegal transition queued -> succeeded"):
        store.finish(job.job_id, JobState.SUCCEEDED)


def test_finish_rejects_non_terminal_state(store):
    job = store.create("b", JobKind.INGEST)
    with pytest.raises(ValueError, match="terminal"):
        store.finish(job.job_id, JobState.RUNNING)


def test_failed_requires_and_stores_error(store):
    job = store.create("b", JobKind.RENDER)
    store.mark_running(job.job_id)
    with pytest.raises(ValueError, match="error message"):
        store.finish(job.job_id, JobState.FAILED)
    failed = store.finish(job.job_id, JobState.FAILED, error="chapter 3: ffmpeg exploded")
    assert failed.state is JobState.FAILED
    assert failed.error == "chapter 3: ffmpeg exploded"


def test_transition_on_unknown_job_raises_not_found(store):
    with pytest.raises(JobNotFoundError):
        store.mark_running("no-such-job")


def test_errors_are_typed_for_the_api_layer(store):
    """M6b maps JobNotFoundError -> 404 and IllegalTransitionError -> 409 by TYPE,
    never by message-sniffing; both stay RepositoryError for broad catches."""
    assert issubclass(JobNotFoundError, RepositoryError)
    assert issubclass(IllegalTransitionError, RepositoryError)
    job = store.create("b", JobKind.RENDER)
    with pytest.raises(IllegalTransitionError) as exc:
        store.finish(job.job_id, JobState.SUCCEEDED)
    assert exc.value.current is JobState.QUEUED and exc.value.target is JobState.SUCCEEDED


# --- the runner's claim primitive ---


def test_try_mark_running_claims_a_queued_job(store):
    job = store.create("b", JobKind.RENDER)
    claimed = store.try_mark_running(job.job_id)
    assert claimed is not None and claimed.state is JobState.RUNNING
    assert claimed.started_at is not None


def test_try_mark_running_loses_quietly_when_canceled_underneath(store):
    """Cancel-while-queued is routine: the runner dequeues an id the user already
    canceled and must get None (skip), not an exception that kills the runner thread."""
    job = store.create("b", JobKind.RENDER)
    store.request_cancel(job.job_id)  # queued -> canceled immediately
    assert store.try_mark_running(job.job_id) is None
    assert store.get(job.job_id).state is JobState.CANCELED  # untouched by the lost claim


def test_try_mark_running_unknown_id_still_raises(store):
    with pytest.raises(JobNotFoundError):
        store.try_mark_running("no-such-job")  # stale handle is a bug, not a lost claim


# --- cancellation ---


def test_cancel_queued_job_is_immediate(store):
    job = store.create("b", JobKind.ATTRIBUTE)
    canceled = store.request_cancel(job.job_id)
    assert canceled.state is JobState.CANCELED
    assert canceled.cancel_requested and canceled.finished_at is not None


def test_cancel_running_job_only_sets_the_flag(store):
    job = store.create("b", JobKind.RENDER)
    store.mark_running(job.job_id)
    flagged = store.request_cancel(job.job_id)
    assert flagged.state is JobState.RUNNING  # still running: cancel is cooperative
    assert flagged.cancel_requested
    assert store.cancel_requested(job.job_id)
    # the runner honors the flag at its next checkpoint:
    assert store.finish(job.job_id, JobState.CANCELED).state is JobState.CANCELED


def test_cancel_terminal_job_is_a_noop(store):
    job = store.create("b", JobKind.RENDER)
    store.mark_running(job.job_id)
    store.finish(job.job_id, JobState.SUCCEEDED)
    after = store.request_cancel(job.job_id)
    assert after.state is JobState.SUCCEEDED and not after.cancel_requested


def test_update_progress_after_finish_is_dropped(store):
    job = store.create("b", JobKind.RENDER)
    store.mark_running(job.job_id)
    store.finish(job.job_id, JobState.SUCCEEDED)
    store.update_progress(job.job_id, "zombie tick")  # racing worker callback: no-op
    assert store.get(job.job_id).progress_text == ""


# --- listing ---


def test_list_jobs_filters_orders_limits(store):
    a = store.create("book-a", JobKind.ATTRIBUTE)
    b = store.create("book-b", JobKind.RENDER)
    c = store.create("book-a", JobKind.RENDER)
    store.mark_running(b.job_id)

    assert [j.job_id for j in store.list_jobs()] == [c.job_id, b.job_id, a.job_id]  # newest first
    assert [j.job_id for j in store.list_jobs(book_id="book-a")] == [c.job_id, a.job_id]
    assert [j.job_id for j in store.list_jobs(states=[JobState.RUNNING])] == [b.job_id]
    assert [j.job_id for j in store.list_jobs(states=["queued", "running"], limit=2)] == [
        c.job_id,
        b.job_id,
    ]


# --- startup reconcile ---


def test_reconcile_startup_terminates_orphans_only(store):
    running = store.create("b1", JobKind.RENDER)
    store.mark_running(running.job_id)
    queued = store.create("b2", JobKind.ATTRIBUTE)
    done = store.create("b3", JobKind.MASTER)
    store.mark_running(done.job_id)
    store.finish(done.job_id, JobState.SUCCEEDED)

    assert store.reconcile_startup() == 2

    orphan = store.get(running.job_id)
    assert orphan.state is JobState.FAILED
    assert "interrupted" in orphan.error and orphan.finished_at is not None
    never_ran = store.get(queued.job_id)
    assert never_ran.state is JobState.CANCELED
    assert store.get(done.job_id).state is JobState.SUCCEEDED  # terminal rows untouched
    assert store.reconcile_startup() == 0  # idempotent


def test_reconcile_honors_a_pending_cancel_request(store):
    """A running job the user canceled just before a crash must come back CANCELED,
    not FAILED — a failed-jobs 'retry' listing must never offer a user-killed job."""
    job = store.create("b", JobKind.RENDER)
    store.mark_running(job.job_id)
    store.request_cancel(job.job_id)  # flag only; server dies before the next checkpoint
    assert store.reconcile_startup() == 1
    after = store.get(job.job_id)
    assert after.state is JobState.CANCELED
    assert "cancel" in after.error and after.finished_at is not None


# --- cross-instance visibility (runner thread vs API request threads) ---


def test_two_store_instances_share_state(tmp_path):
    path = tmp_path / "jobs.db"
    writer, reader = JobStore(path), JobStore(path)
    job = writer.create("b", JobKind.RENDER)
    assert reader.get(job.job_id).state is JobState.QUEUED
    writer.mark_running(job.job_id)
    writer.update_progress(job.job_id, "seg 5")
    seen = reader.get(job.job_id)
    assert seen.state is JobState.RUNNING and seen.progress_text == "seg 5"


def test_job_model_serializes_for_the_api(store):
    """Job must round-trip through JSON — it's the payload M6b returns verbatim."""
    job = store.create("b", JobKind.RENDER)
    assert Job.model_validate_json(job.model_dump_json()) == job
