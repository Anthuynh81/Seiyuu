"""M6b-2: job-store params column + warmup kind, /api/jobs endpoints, stage handlers."""

import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from fake_engine import FakeEngine
from seiyuu.api.handlers import _loudness_target, _pause_profile, build_handlers
from seiyuu.api.main import create_app
from seiyuu.api.schemas import JobOut, LoudnessWrite, PauseWrite
from seiyuu.repository import Job, JobKind, JobState, JobStore
from test_api_m6b1 import make_settings

# The pre-M6b schema, verbatim minus the params column — the migration source.
_OLD_SCHEMA = """
CREATE TABLE jobs (
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
"""


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        yield c


def _error(resp) -> dict:
    return resp.json()["error"]


def _wait_terminal(store: JobStore, job_id: str, timeout: float = 5.0) -> Job:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job.is_terminal:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} not terminal within {timeout}s: {store.get(job_id)}")


def _wait_state(store: JobStore, job_id: str, state: JobState, timeout: float = 5.0) -> Job:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job.state is state:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} never reached {state}: {store.get(job_id)}")


# -- store: params column + migration ----------------------------------------------------


def test_params_roundtrip(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    plain = store.create("bk", "attribute")
    assert plain.params is None
    with_params = store.create("bk2", "assemble", params={"pauses": {"paragraph": 0.0}})
    assert store.get(with_params.job_id).params == {"pauses": {"paragraph": 0.0}}


def test_warmup_kind_exists(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    job = store.create("engine:kokoro", JobKind.WARMUP, params={"engine_id": "kokoro"})
    assert job.kind is JobKind.WARMUP


def test_migration_from_pre_m6b_schema(tmp_path) -> None:
    db = tmp_path / "jobs.db"
    conn = sqlite3.connect(db)
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO jobs (job_id, book_id, kind, state, created_at) VALUES (?,?,?,?,?)",
        ("old1", "bk", "render", "succeeded", "2026-07-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    store = JobStore(db)  # idempotent ALTER runs here
    assert store.get("old1").params is None  # pre-M6b row reads clean
    new = store.create("bk", "assemble", params={"a": 1})
    assert store.get(new.job_id).params == {"a": 1}
    JobStore(db)  # second open: migration must be a no-op, not an error


# -- JobOut cost-token redaction ----------------------------------------------------------


def test_jobout_redacts_cost_token(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    job = store.create(
        "bk", "render", params={"cost_token": "cq1.payload.sigsigsig", "mode": "single"}
    )
    out = JobOut.from_job(job)
    assert out.params["cost_token"] == {"present": True, "sig_suffix": "igsigsig"}
    assert out.params["mode"] == "single"
    assert "cq1." not in str(out.model_dump())
    # the stored row keeps the live token for the handler
    assert store.get(job.job_id).params["cost_token"] == "cq1.payload.sigsigsig"


# -- /api/jobs ----------------------------------------------------------------------------


def test_list_jobs_filters(client) -> None:
    store: JobStore = client.app.state.store
    a1 = store.create("bk-a", "attribute")
    store.create("bk-a", "assemble")
    a2 = store.create("bk-b", "attribute")
    store.mark_running(a2.job_id)

    all_jobs = client.get("/api/jobs").json()["jobs"]
    assert [j["job_id"] for j in all_jobs][:1] == [a2.job_id]  # newest-first
    assert len(all_jobs) == 3

    by_book = client.get("/api/jobs", params={"book_id": "bk-a"}).json()["jobs"]
    assert {j["book_id"] for j in by_book} == {"bk-a"}

    running = client.get("/api/jobs", params={"state": "running"}).json()["jobs"]
    assert [j["job_id"] for j in running] == [a2.job_id]

    attrs = client.get("/api/jobs", params=[("kind", "attribute"), ("limit", 1)]).json()["jobs"]
    assert [j["job_id"] for j in attrs] == [a2.job_id]  # limit applies AFTER kind filter
    attrs_all = client.get("/api/jobs", params={"kind": "attribute"}).json()["jobs"]
    assert {j["job_id"] for j in attrs_all} == {a1.job_id, a2.job_id}


def test_list_jobs_bad_state_is_422(client) -> None:
    resp = client.get("/api/jobs", params={"state": "exploded"})
    assert resp.status_code == 422
    assert _error(resp)["code"] == "invalid"


def test_get_job_and_unknown_404(client) -> None:
    store: JobStore = client.app.state.store
    job = store.create("bk", "master")
    body = client.get(f"/api/jobs/{job.job_id}").json()
    assert body["state"] == "queued"
    assert body["is_terminal"] is False

    resp = client.get("/api/jobs/nope")
    assert resp.status_code == 404
    assert _error(resp)["code"] == "not_found"


def test_cancel_queued_running_terminal(client) -> None:
    store: JobStore = client.app.state.store
    queued = store.create("bk", "attribute")
    resp = client.post(f"/api/jobs/{queued.job_id}/cancel")
    assert resp.status_code == 202
    assert resp.json()["state"] == "canceled"

    running = store.create("bk2", "attribute")
    store.mark_running(running.job_id)
    body = client.post(f"/api/jobs/{running.job_id}/cancel").json()
    assert body["state"] == "running"  # flag only; settles at the next checkpoint
    assert body["cancel_requested"] is True

    # idempotent on terminal: the canceled job stays canceled, still 202
    again = client.post(f"/api/jobs/{queued.job_id}/cancel")
    assert again.status_code == 202
    assert again.json()["state"] == "canceled"


# -- warmup job end-to-end ----------------------------------------------------------------


class _WarmGate:
    """Controllable fake engine factory: warm() blocks until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.warmed = threading.Event()

    def make_engine(self, engine_id: str, **kwargs):
        gate = self

        class _Warmable(FakeEngine):
            def warm(self) -> None:
                gate.warmed.set()
                if not gate.release.wait(timeout=5.0):
                    raise RuntimeError("warm release never set")

        return _Warmable()


def test_warmup_flow(client, monkeypatch) -> None:
    gate = _WarmGate()
    monkeypatch.setattr("seiyuu.api.registry.get_engine", gate.make_engine)
    store: JobStore = client.app.state.store

    resp = client.post("/api/engines/kokoro/warmup")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.headers["location"] == f"/api/jobs/{job_id}"
    assert resp.json()["book_id"] == "engine:kokoro"
    assert resp.json()["kind"] == "warmup"

    assert gate.warmed.wait(timeout=5.0)  # handler is inside warm(), job is running
    dup = client.post("/api/engines/kokoro/warmup")
    assert dup.status_code == 409
    err = _error(dup)
    assert err["code"] == "duplicate_job"
    assert err["detail"]["job_id"] == job_id  # full Job in detail: UI links straight to it

    gate.release.set()
    done = _wait_terminal(store, job_id)
    assert done.state is JobState.SUCCEEDED
    assert "resident" in done.progress_text

    engines = {e["engine_id"]: e for e in client.get("/api/engines").json()["engines"]}
    assert engines["kokoro"]["resident"] is True

    # not duplicate anymore: a finished warmup can be re-run
    gate.release.set()
    assert client.post("/api/engines/kokoro/warmup").status_code == 202


def test_warmup_failure_never_reads_resident(client, monkeypatch) -> None:
    # A failed load must not leave the manager holding a half-loaded engine — identity
    # residency would then report warm and the cold-engine refusal would be skipped.
    def make_broken(engine_id: str, **kwargs):
        class _Broken(FakeEngine):
            def warm(self) -> None:
                raise RuntimeError("download died")

        return _Broken()

    monkeypatch.setattr("seiyuu.api.registry.get_engine", make_broken)
    job_id = client.post("/api/engines/kokoro/warmup").json()["job_id"]
    done = _wait_terminal(client.app.state.store, job_id)
    assert done.state is JobState.FAILED
    assert "download died" in done.error

    from seiyuu.gpu import get_gpu_manager

    assert get_gpu_manager().resident is None  # freed on failure, not left acquired
    engines = {e["engine_id"]: e for e in client.get("/api/engines").json()["engines"]}
    assert engines["kokoro"]["resident"] is False


def test_warmup_cloud_engine_409(client) -> None:
    resp = client.post("/api/engines/elevenlabs/warmup")
    assert resp.status_code == 409
    assert _error(resp)["code"] == "nothing_to_warm"


def test_warmup_unknown_engine_404(client) -> None:
    assert client.post("/api/engines/fish/warmup").status_code == 404


# -- handlers: params conversion + service wiring ----------------------------------------


def test_pause_profile_honors_explicit_zero() -> None:
    profile = _pause_profile(PauseWrite(paragraph=0.0, scene_break=2.5))
    assert profile.paragraph == 0.0  # the CLI's `or default` bug, deliberately fixed
    assert profile.scene_break == 2.5
    assert profile.dialogue == 0.35  # None -> default
    defaults = _pause_profile(None)
    assert defaults.paragraph == 0.6


def test_loudness_target_null_semantics(tmp_path) -> None:
    cfg = make_settings(tmp_path)
    assert _loudness_target(cfg, LoudnessWrite(enabled=False)) is None
    on = _loudness_target(cfg, None)  # settings default: enabled, -18 LUFS
    assert on is not None and on.i == -18.0
    zero = _loudness_target(cfg, LoudnessWrite(target_lufs=0.0))
    assert zero is not None and zero.i == 0.0  # explicit 0.0 honored


def _run_handler(client, kind: JobKind, book_id: str, params: dict) -> Job:
    runner = client.app.state.runner
    job = runner.enqueue(book_id, kind, params=params)
    return _wait_terminal(client.app.state.store, job.job_id)


def test_attribute_handler_wires_params(client, monkeypatch, tmp_path) -> None:
    cfg = client.app.state.settings
    book_dir = cfg.books_dir / "bk"
    book_dir.mkdir(parents=True)
    (book_dir / "normalized.json").write_text(
        '{"book_meta": {"book_id": "bk", "title": "T", "authors": [], '
        '"source_path": "t.epub", "source_sha256": "0" }, "chapters": []}',
        encoding="utf-8",
    )
    seen: dict = {}

    def fake_run(book, bdir, **kwargs) -> None:
        seen["book_id"] = book.book_meta.book_id
        seen["book_dir"] = bdir
        seen.update(kwargs)

    monkeypatch.setattr("seiyuu.api.handlers.run_attribution", fake_run)
    done = _run_handler(
        client,
        JobKind.ATTRIBUTE,
        "bk",
        {"chapters": [2, 3], "provider": "local", "use_hybrid": False},
    )
    assert done.state is JobState.SUCCEEDED
    assert seen["book_id"] == "bk"
    assert seen["book_dir"] == book_dir
    assert seen["chapters"] == (2, 3)
    assert seen["provider_id"] == "local"
    assert seen["use_hybrid"] is False
    assert callable(seen["check_cancel"]) and callable(seen["progress"])


def test_attribute_handler_missing_book_fails_loudly(client) -> None:
    done = _run_handler(client, JobKind.ATTRIBUTE, "ghost", {})
    assert done.state is JobState.FAILED
    assert "run ingest first" in done.error


def test_master_handler_wires_params_and_cover(client, monkeypatch) -> None:
    cfg = client.app.state.settings
    out_dir = cfg.output_dir / "bk"
    out_dir.mkdir(parents=True)
    (out_dir / "cover.png").write_bytes(b"\x89PNG fake")
    seen: dict = {}

    def fake_master(book_output_dir, **kwargs) -> None:
        seen["dir"] = book_output_dir
        seen.update(kwargs)

    monkeypatch.setattr("seiyuu.assemble.master_book", fake_master)
    done = _run_handler(
        client,
        JobKind.MASTER,
        "bk",
        {"bitrate": "96k", "target_minutes": 2.0, "pauses": {"paragraph": 0.0}},
    )
    assert done.state is JobState.SUCCEEDED
    assert seen["dir"] == out_dir
    assert seen["bitrate"] == "96k"
    assert seen["target_seconds"] == 120.0
    assert seen["cover"] == out_dir / "cover.png"  # use_cover default finds the upload
    assert seen["pauses"].paragraph == 0.0
    assert seen["tempo_bounds"] == (cfg.tempo_min, cfg.tempo_max)


def test_handler_rejects_garbage_params(client) -> None:
    done = _run_handler(client, JobKind.MASTER, "bk", {"target_minutes": -3})
    assert done.state is JobState.FAILED
    assert "ValidationError" in done.error


def test_build_handlers_covers_all_but_render_and_ingest(tmp_path) -> None:
    from seiyuu.api.concurrency import HeavyWorkGate
    from seiyuu.api.registry import EngineRegistry

    cfg = make_settings(tmp_path)
    handlers = build_handlers(cfg, EngineRegistry(cfg), HeavyWorkGate())
    assert set(handlers) == {
        JobKind.WARMUP,
        JobKind.ATTRIBUTE,
        JobKind.ASSEMBLE,
        JobKind.MASTER,
    }  # RENDER lands with the money gate (M6b-5); INGEST stays synchronous by design
