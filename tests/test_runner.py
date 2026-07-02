"""Job runner: single-flight execution, cooperative cancel, exactly-once settle (M6a-3)."""

import threading
import time
from contextlib import contextmanager

import pytest

from seiyuu.jobs import JobRunner
from seiyuu.repository import JobKind, JobState, JobStore

POLL = 0.01


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


@contextmanager
def running(store, handlers):
    runner = JobRunner(store, handlers, poll_seconds=POLL)
    runner.start()
    try:
        yield runner
    finally:
        runner.stop(timeout=5.0)


def wait_terminal(store, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job.is_terminal:
            return job
        time.sleep(0.005)
    pytest.fail(f"job {job_id!r} did not settle within {timeout}s")


def spin_until_canceled(ctx, timeout=5.0) -> None:
    """A well-behaved handler body: checkpoint until the cancel flag raises."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ctx.check_cancel()
        time.sleep(0.005)
    raise AssertionError("cancel flag never arrived at the checkpoint")


def test_success_lifecycle_and_progress(store):
    seen = []

    def handler(ctx):
        seen.append((ctx.job.book_id, ctx.job.kind))
        ctx.progress("chapter 1/2")

    with running(store, {JobKind.RENDER: handler}) as runner:
        job = runner.enqueue("book-1", JobKind.RENDER)
        done = wait_terminal(store, job.job_id)
    assert done.state is JobState.SUCCEEDED
    assert done.progress_text == "chapter 1/2"
    assert done.started_at is not None and done.finished_at is not None
    assert seen == [("book-1", JobKind.RENDER)]


def test_failure_settles_row_and_worker_survives(store):
    def handler(ctx):
        if ctx.job.book_id == "bad":
            raise ValueError("chapter 3: kaboom")

    with running(store, {JobKind.RENDER: handler}) as runner:
        bad = runner.enqueue("bad", JobKind.RENDER)
        good = runner.enqueue("good", JobKind.RENDER)
        assert wait_terminal(store, bad.job_id).state is JobState.FAILED
        assert wait_terminal(store, good.job_id).state is JobState.SUCCEEDED  # thread survived
    assert store.get(bad.job_id).error == "ValueError: chapter 3: kaboom"


def test_single_flight_never_overlaps(store):
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def handler(ctx):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1

    with running(store, {JobKind.ATTRIBUTE: handler}) as runner:
        jobs = [runner.enqueue(f"b{i}", JobKind.ATTRIBUTE) for i in range(3)]
        for job in jobs:
            assert wait_terminal(store, job.job_id).state is JobState.SUCCEEDED
    assert state["max_active"] == 1


def test_cancel_while_queued_is_a_silent_skip(store):
    release = threading.Event()
    blocker_started = threading.Event()
    executed = []

    def handler(ctx):
        executed.append(ctx.job.book_id)
        blocker_started.set()
        assert release.wait(5.0)

    with running(store, {JobKind.RENDER: handler}) as runner:
        blocker = runner.enqueue("blocker", JobKind.RENDER)
        assert blocker_started.wait(5.0)
        victim = runner.enqueue("victim", JobKind.RENDER)
        canceled = runner.request_cancel(victim.job_id)
        assert canceled.state is JobState.CANCELED  # queued: canceled immediately
        release.set()
        assert wait_terminal(store, blocker.job_id).state is JobState.SUCCEEDED
        # give the worker a beat to dequeue the victim id and (correctly) skip it
        time.sleep(POLL * 10)
    assert executed == ["blocker"]
    assert store.get(victim.job_id).state is JobState.CANCELED


def test_cooperative_cancel_mid_run(store):
    started = threading.Event()

    def handler(ctx):
        started.set()
        spin_until_canceled(ctx)

    with running(store, {JobKind.RENDER: handler}) as runner:
        job = runner.enqueue("b", JobKind.RENDER)
        assert started.wait(5.0)
        runner.request_cancel(job.job_id)
        assert wait_terminal(store, job.job_id).state is JobState.CANCELED


def test_handler_finally_runs_on_cancel(store):
    """The unload guarantee: a handler's finally (e.g. provider.unload()) must run even
    when the job is canceled mid-flight — VRAM may never stay held by a dead job."""
    started = threading.Event()
    unloaded = []

    def handler(ctx):
        try:
            started.set()
            spin_until_canceled(ctx)
        finally:
            unloaded.append(True)

    with running(store, {JobKind.ATTRIBUTE: handler}) as runner:
        job = runner.enqueue("b", JobKind.ATTRIBUTE)
        assert started.wait(5.0)
        runner.request_cancel(job.job_id)
        assert wait_terminal(store, job.job_id).state is JobState.CANCELED
    assert unloaded == [True]


def test_missing_handler_fails_the_job_loudly(store):
    with running(store, {JobKind.RENDER: lambda ctx: None}) as runner:
        job = runner.enqueue("b", JobKind.MASTER)
        done = wait_terminal(store, job.job_id)
    assert done.state is JobState.FAILED
    assert "no handler registered" in done.error and "master" in done.error


def test_enqueue_requires_a_running_runner(store):
    runner = JobRunner(store, {}, poll_seconds=POLL)
    with pytest.raises(RuntimeError, match="start"):
        runner.enqueue("b", JobKind.RENDER)
    runner.start()
    runner.stop(timeout=5.0)
    with pytest.raises(RuntimeError, match="start"):
        runner.enqueue("b", JobKind.RENDER)


def test_start_reconciles_dead_process_rows_first(store):
    orphan_running = store.create("b1", JobKind.RENDER)
    store.mark_running(orphan_running.job_id)
    orphan_queued = store.create("b2", JobKind.ATTRIBUTE)

    runner = JobRunner(store, {}, poll_seconds=POLL)
    try:
        assert runner.start() == 2
    finally:
        runner.stop(timeout=5.0)
    assert store.get(orphan_running.job_id).state is JobState.FAILED
    assert store.get(orphan_queued.job_id).state is JobState.CANCELED


def test_stop_cancels_the_active_job(store):
    started = threading.Event()

    def handler(ctx):
        started.set()
        spin_until_canceled(ctx)

    runner = JobRunner(store, {JobKind.RENDER: handler}, poll_seconds=POLL)
    runner.start()
    job = runner.enqueue("b", JobKind.RENDER)
    assert started.wait(5.0)
    assert runner.stop(cancel_pending=True, timeout=5.0)
    assert store.get(job.job_id).state is JobState.CANCELED
    assert runner.active_job_id is None


def test_stop_sweeps_queued_jobs_too(store):
    """stop() cancels via a STORE sweep, not an in-memory snapshot — a queued job (and a
    job mid-claim) must not escape shutdown and run to completion afterwards."""
    started = threading.Event()

    def handler(ctx):
        started.set()
        spin_until_canceled(ctx)

    runner = JobRunner(store, {JobKind.RENDER: handler}, poll_seconds=POLL)
    runner.start()
    active = runner.enqueue("active", JobKind.RENDER)
    assert started.wait(5.0)
    queued = runner.enqueue("queued", JobKind.RENDER)
    assert runner.stop(cancel_pending=True, timeout=5.0)
    assert store.get(active.job_id).state is JobState.CANCELED
    assert store.get(queued.job_id).state is JobState.CANCELED


def test_start_after_stop_raises(store):
    runner = JobRunner(store, {}, poll_seconds=POLL)
    runner.start()
    runner.stop(timeout=5.0)
    with pytest.raises(RuntimeError, match="construct a new one"):
        runner.start()
