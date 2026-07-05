"""F1 — auditions borrow a running render's resident engine between segments.

Three layers, all fixtures/fakes (no real TTS):
  * admission verdict table (_refuse_conflicts -> REFUSE / EXCLUSIVE / BORROW),
  * render-loop hooks (publish once / serve at each yield after check_cancel / close before
    free_all),
  * route integration (an audition borrows during a simulated render without evicting the
    resident model; mismatches and non-render jobs fall back to the pre-F1 refusals).
"""

import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.api.concurrency import BorrowBroker
from seiyuu.api.errors import ApiError
from seiyuu.api.main import create_app
from seiyuu.api.routes.voices import Verdict, _refuse_conflicts
from seiyuu.gpu import GpuResourceManager, get_gpu_manager
from seiyuu.render import render_book, render_book_multivoice
from seiyuu.repository import JobState, JobStore
from seiyuu.voices import VoiceAssignment
from test_api_m6b1 import make_settings
from test_render_multivoice import _library, _patch_engine, _report

ENGINE = FakeEngine()  # sentinel instance; the broker never calls its methods


def _wait(pred, timeout: float = 3.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.001)
    raise AssertionError("condition not met within timeout")


# -- admission verdict table --------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


def _live(store: JobStore) -> list:
    return store.list_jobs(states=[JobState.QUEUED, JobState.RUNNING])


_REG = SimpleNamespace(is_resident=lambda _e: True)  # warm -> engine_cold never fires
_GPU_CLS = SimpleNamespace(uses_gpu=True)
_CLOUD_CLS = SimpleNamespace(uses_gpu=False)


def test_verdict_no_gpu_job_is_exclusive(store) -> None:
    broker = BorrowBroker()
    meta = SimpleNamespace(engine="kokoro")
    assert _refuse_conflicts(meta, _GPU_CLS, _live(store), _REG, broker) is Verdict.EXCLUSIVE


def test_verdict_render_lending_same_engine_borrows(store) -> None:
    store.mark_running(store.create("bk", "render").job_id)
    broker = BorrowBroker()
    broker.publish("kokoro", ENGINE)  # the render advertises its resident engine
    meta = SimpleNamespace(engine="kokoro")
    assert _refuse_conflicts(meta, _GPU_CLS, _live(store), _REG, broker) is Verdict.BORROW


def test_verdict_render_lending_survives_a_queued_attribute(store) -> None:
    # a running render lending kokoro + a NEWER queued attribute (which sorts first in the
    # newest-first live list) must STILL borrow — a coincidental queued GPU job never defeats
    # a valid borrow (the pre-fix "first GPU job must be the render" check refused this).
    store.mark_running(store.create("bk", "render").job_id)
    store.create("other", "attribute")  # queued, newer -> ahead of the render in `live`
    broker = BorrowBroker()
    broker.publish("kokoro", ENGINE)
    meta = SimpleNamespace(engine="kokoro")
    assert _refuse_conflicts(meta, _GPU_CLS, _live(store), _REG, broker) is Verdict.BORROW


def test_verdict_render_different_engine_refuses(store) -> None:
    store.mark_running(store.create("bk", "render").job_id)
    broker = BorrowBroker()
    broker.publish("chatterbox", ENGINE)  # lending a DIFFERENT engine
    meta = SimpleNamespace(engine="kokoro")
    with pytest.raises(ApiError) as ei:
        _refuse_conflicts(meta, _GPU_CLS, _live(store), _REG, broker)
    assert ei.value.code == "gpu_busy"


def test_verdict_render_refusal_names_resident_engine(store) -> None:
    store.mark_running(store.create("bk", "render").job_id)
    broker = BorrowBroker()  # not lending kokoro
    meta = SimpleNamespace(engine="kokoro")
    gpu = get_gpu_manager()
    try:
        with gpu.acquire(ENGINE, "engine:chatterbox"):  # some other model is resident
            with pytest.raises(ApiError) as ei:
                _refuse_conflicts(meta, _GPU_CLS, _live(store), _REG, broker)
        assert ei.value.code == "gpu_busy"
        assert "engine:chatterbox" in ei.value.message  # resident surfaced to the UI
    finally:
        gpu.free_all()


def test_verdict_attribute_refuses_even_if_broker_lending(store) -> None:
    store.mark_running(store.create("bk", "attribute").job_id)
    broker = BorrowBroker()
    broker.publish("kokoro", ENGINE)  # irrelevant: ATTRIBUTE is not a RENDER
    meta = SimpleNamespace(engine="kokoro")
    with pytest.raises(ApiError) as ei:
        _refuse_conflicts(meta, _GPU_CLS, _live(store), _REG, broker)
    assert ei.value.code == "gpu_busy"


def test_verdict_warmup_refuses(store) -> None:
    store.mark_running(store.create("engine:kokoro", "warmup").job_id)
    broker = BorrowBroker()
    broker.publish("kokoro", ENGINE)
    meta = SimpleNamespace(engine="kokoro")
    with pytest.raises(ApiError) as ei:
        _refuse_conflicts(meta, _GPU_CLS, _live(store), _REG, broker)
    assert ei.value.code == "gpu_busy"


def test_verdict_cloud_engine_unaffected_by_borrow(store) -> None:
    # no GPU job -> a cloud audition is EXCLUSIVE (never borrows; uses_gpu False)
    meta = SimpleNamespace(engine="elevenlabs")
    verdict = _refuse_conflicts(meta, _CLOUD_CLS, _live(store), _REG, BorrowBroker())
    assert verdict is Verdict.EXCLUSIVE
    # a live render holding cloud slots -> the pre-F1 cloud_busy refusal, not a borrow
    store.mark_running(store.create("bk", "render").job_id)
    with pytest.raises(ApiError) as ei:
        _refuse_conflicts(meta, _CLOUD_CLS, _live(store), _REG, BorrowBroker())
    assert ei.value.code == "cloud_busy"


# -- render-loop hooks --------------------------------------------------------------------


class _RecordingBroker:
    """Records the render's publish/serve/close calls into a shared ordered log."""

    def __init__(self, log: list) -> None:
        self._log = log

    def publish(self, engine_id: str, engine) -> None:
        self._log.append(("publish", engine_id))

    def serve(self, engine_id: str, engine) -> None:
        self._log.append(("serve", engine_id))

    def close(self) -> None:
        self._log.append("close")


class _SpyGpu(GpuResourceManager):
    def __init__(self, log: list) -> None:
        super().__init__()
        self._log = log

    def free_all(self) -> None:
        self._log.append("free_all")
        super().free_all()


def _assert_check_precedes_each_serve(log: list) -> None:
    """At no point may a serve have run without a preceding (unconsumed) check."""
    checks = serves = 0
    for entry in log:
        if entry == "check":
            checks += 1
        elif isinstance(entry, tuple) and entry[0] == "serve":
            serves += 1
            assert checks >= serves, f"serve before check at {entry}: {log}"


def test_single_voice_render_hooks(tmp_path) -> None:
    log: list = []
    engine = FakeEngine()  # engine_id == "fake", uses_gpu True
    broker = _RecordingBroker(log)
    gpu = _SpyGpu(log)
    render_book(
        make_book(),
        engine,
        "test_voice",
        tmp_path / "book",
        gpu=gpu,
        broker=broker,
        check_cancel=lambda: log.append("check"),
    )
    publishes = [e for e in log if isinstance(e, tuple) and e[0] == "publish"]
    assert publishes == [("publish", "fake")]  # published exactly once, before the loop
    assert log.index(("publish", "fake")) < log.index("check")
    # a serve rides every yield point (2 chapter + 6 block checks), always after the check
    serves = [e for e in log if isinstance(e, tuple) and e[0] == "serve"]
    assert len(serves) == 8
    _assert_check_precedes_each_serve(log)
    # teardown lends-off BEFORE the GPU is freed
    assert log.index("close") < log.index("free_all")


def test_multivoice_render_hooks(tmp_path, monkeypatch) -> None:
    log: list = []
    engine = FakeEngine()
    _patch_engine(monkeypatch, engine)
    broker = _RecordingBroker(log)
    gpu = _SpyGpu(log)
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "alice_v"},
    )
    render_book_multivoice(
        _report(),
        make_book(),
        _library(tmp_path),
        assignment,
        tmp_path / "out",
        gpu=gpu,
        broker=broker,
        check_cancel=lambda: log.append("check"),
    )
    # one publish per segment (5 attributed segments), all keyed on the voice's engine
    publishes = [e for e in log if isinstance(e, tuple) and e[0] == "publish"]
    assert publishes == [("publish", "kokoro")] * 5
    assert any(isinstance(e, tuple) and e[0] == "serve" for e in log)
    _assert_check_precedes_each_serve(log)
    assert log.index("close") < log.index("free_all")


def test_close_runs_before_free_all_on_cancel(tmp_path) -> None:
    """A canceled render must also lend-off before freeing (finally order holds)."""
    from seiyuu.jobs import JobCanceled

    log: list = []

    def cancel_after_first() -> None:
        log.append("check")
        if sum(1 for e in log if e == "check") > 1:
            raise JobCanceled("stop")

    with pytest.raises(JobCanceled):
        render_book(
            make_book(),
            FakeEngine(),
            "test_voice",
            tmp_path / "book",
            gpu=_SpyGpu(log),
            broker=_RecordingBroker(log),
            check_cancel=cancel_after_first,
        )
    assert log.index("close") < log.index("free_all")


# -- route integration --------------------------------------------------------------------


class _UnloadCountingEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.unloads = 0

    def unload(self) -> None:
        self.unloads += 1


class PaidCloudFake(FakeEngine):
    engine_id = "elevenlabs-fake"
    uses_gpu = False

    def cost_estimate(self, text: str) -> float:
        return len(text) / 1000 * 0.30


class _RenderSim(threading.Thread):
    """Stands in for a running render: publishes its resident engine and serves the broker
    at a tight cadence, exactly like the render loop's check() yield points."""

    def __init__(self, broker: BorrowBroker, engine_id: str, engine, gpu) -> None:
        super().__init__(daemon=True)
        self._broker = broker
        self._engine_id = engine_id
        self._engine = engine
        self._gpu = gpu
        self._stop_flag = threading.Event()

    def run(self) -> None:
        self._broker.publish(self._engine_id, self._engine)
        with self._gpu.acquire(self._engine, f"engine:{self._engine_id}"):
            pass  # establish residency like the render's first segment
        while not self._stop_flag.is_set():
            self._broker.serve(self._engine_id, self._engine)
            time.sleep(0.002)

    def stop(self) -> None:
        self._stop_flag.set()


def _patch_audition(monkeypatch, engine=None, cls_by_id=None) -> None:
    engine = engine or FakeEngine()
    monkeypatch.setattr("seiyuu.api.registry.get_engine", lambda eid, **kw: engine)
    classes = cls_by_id or {}
    monkeypatch.setattr(
        "seiyuu.api.routes.voices.get_engine_class",
        lambda eid: classes.get(eid, FakeEngine),
    )
    monkeypatch.setattr("seiyuu.api.routes.voices.weights_cached", lambda eid: True)


@pytest.fixture
def client(tmp_path):
    app = create_app(settings=make_settings(tmp_path))
    with TestClient(app) as c:
        c.app = app
        yield c


@pytest.fixture
def keyed_client(tmp_path):
    app = create_app(settings=make_settings(tmp_path, elevenlabs_api_key="k-test-not-real"))
    with TestClient(app) as c:
        c.app = app
        yield c


def _error(resp) -> dict:
    return resp.json()["error"]


def _make_preset(client, voice_id="v1", engine="kokoro", preset_id="test_voice") -> None:
    resp = client.post(
        "/api/voices",
        json={
            "kind": "preset",
            "name": "Test Voice",
            "engine": engine,
            "preset_id": preset_id,
            "voice_id": voice_id,
        },
    )
    assert resp.status_code == 201, resp.text


def test_audition_borrows_resident_engine_during_render(client, monkeypatch) -> None:
    engine = _UnloadCountingEngine()
    _patch_audition(monkeypatch, engine)
    _make_preset(client)
    store: JobStore = client.app.state.store
    store.mark_running(store.create("bk", "render").job_id)
    broker: BorrowBroker = client.app.state.borrow_broker
    gpu = get_gpu_manager()
    sim = _RenderSim(broker, "kokoro", engine, gpu)
    sim.start()
    try:
        _wait(lambda: broker.eligible("kokoro") and gpu.holds(engine))
        resp = client.post("/api/voices/v1/audition", json={})
        assert resp.status_code == 200, resp.text
        assert engine.calls  # the borrowed instance actually synthesized
        assert gpu.holds(engine)  # resident model UNCHANGED — a true borrow
        assert engine.unloads == 0  # no evict+reload of the render's engine
    finally:
        sim.stop()
        sim.join(3.0)
        gpu.free_all()


def test_audition_different_engine_409s_during_render(client, monkeypatch) -> None:
    engine = FakeEngine()
    _patch_audition(monkeypatch, engine)
    _make_preset(client)
    store: JobStore = client.app.state.store
    store.mark_running(store.create("bk", "render").job_id)
    client.app.state.borrow_broker.publish("chatterbox", engine)  # lending a different engine
    resp = client.post("/api/voices/v1/audition", json={})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "gpu_busy"


def test_audition_during_attribute_still_409(client, monkeypatch) -> None:
    _patch_audition(monkeypatch)
    _make_preset(client)
    store: JobStore = client.app.state.store
    store.mark_running(store.create("bk", "attribute").job_id)
    client.app.state.borrow_broker.publish("kokoro", FakeEngine())  # ATTRIBUTE never borrows
    resp = client.post("/api/voices/v1/audition", json={})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "gpu_busy"


def test_audition_during_assemble_still_allowed(client, monkeypatch) -> None:
    engine = FakeEngine()
    _patch_audition(monkeypatch, engine)
    _make_preset(client)
    store: JobStore = client.app.state.store
    store.create("bk", "assemble")  # holds no GPU -> auditions run exclusively
    resp = client.post("/api/voices/v1/audition", json={})
    assert resp.status_code == 200, resp.text


def test_cloud_audition_during_render_still_cloud_busy(keyed_client, monkeypatch) -> None:
    _patch_audition(monkeypatch, PaidCloudFake(), cls_by_id={"elevenlabs": PaidCloudFake})
    _make_preset(keyed_client, voice_id="cloudv", engine="elevenlabs", preset_id="stock1")
    store: JobStore = keyed_client.app.state.store
    store.mark_running(store.create("bk", "render").job_id)
    resp = keyed_client.post("/api/voices/cloudv/audition", json={"confirm_paid": True})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "cloud_busy"  # cloud path unchanged by F1
