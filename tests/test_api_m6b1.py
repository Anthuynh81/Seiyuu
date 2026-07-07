"""M6b-1: app skeleton — lifespan wiring, error envelope, system/settings/engines routes."""

import pytest
from fastapi.testclient import TestClient

from seiyuu import __version__
from seiyuu.api.concurrency import HeavyWorkGate
from seiyuu.api.main import create_app
from seiyuu.api.registry import EngineRegistry
from seiyuu.repository import JobState, JobStore
from seiyuu.repository.jobs import JOBS_DB_NAME
from seiyuu.settings import Settings


def make_settings(tmp_path, **overrides) -> Settings:
    """Isolated Settings: no .env, no cloud keys, all roots under tmp_path."""
    defaults = dict(
        books_dir=tmp_path / "books",
        output_dir=tmp_path / "output",
        voices_dir=tmp_path / "voices",
        data_dir=tmp_path / "data",
        anthropic_api_key=None,
        elevenlabs_api_key=None,
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        yield c


def _error(resp) -> dict:
    body = resp.json()
    assert set(body) == {"error"}, body
    assert set(body["error"]) == {"code", "message", "detail"}
    return body["error"]


# -- health / system / settings --------------------------------------------------------


def test_health(client) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}


def test_system_shape_and_defaults(client) -> None:
    body = client.get("/api/system").json()
    assert body["active_job"] is None
    assert body["queued_jobs"] == 0
    assert body["audition_in_flight"] is False
    assert body["reconciled_at_startup"] == 0
    assert isinstance(body["ffmpeg_available"], bool)
    assert body["ollama"]["reachable"] is None  # default poll is network-free
    assert body["keys"] == {"anthropic_configured": False, "elevenlabs_configured": False}
    assert body["limits"]["full_render_confirm_blocks"] == 300
    assert body["limits"]["render_max_usd"] == 25.0
    assert body["limits"]["max_upload_bytes"] == 100 * 1024 * 1024
    # the attribution defaults the UI shows (and lets the user override per job)
    assert body["attribution"]["provider"] == "local"
    assert body["attribution"]["model"] == "qwen2.5:7b"
    assert body["attribution"]["prompt_version"]
    assert body["attribution"]["hybrid"] is False
    assert body["engines"] == ["chatterbox", "elevenlabs", "indextts2", "kokoro"]
    assert body["version"] == __version__


def test_system_shows_running_and_queued_jobs(client) -> None:
    store: JobStore = client.app.state.store
    running = store.create("bk-1", "render")
    store.mark_running(running.job_id)
    store.create("bk-1", "assemble")

    body = client.get("/api/system").json()
    assert body["active_job"]["job_id"] == running.job_id
    assert body["active_job"]["state"] == "running"
    assert body["active_job"]["is_terminal"] is False
    assert body["queued_jobs"] == 1


def test_settings_redacted(client, tmp_path) -> None:
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["anthropic_key_configured"] is False
    assert body["elevenlabs_key_configured"] is False
    assert "api_key" not in resp.text
    assert body["books_dir"] == str(tmp_path / "books")
    assert body["tts_engine"] == "kokoro"


def test_settings_reports_configured_keys(tmp_path) -> None:
    settings = make_settings(tmp_path, elevenlabs_api_key="k-not-a-real-key")
    with TestClient(create_app(settings=settings)) as c:
        body = c.get("/api/settings").json()
        assert body["elevenlabs_key_configured"] is True
        assert "k-not-a-real-key" not in c.get("/api/settings").text
        assert c.get("/api/system").json()["keys"]["elevenlabs_configured"] is True


# -- startup reconcile ------------------------------------------------------------------


def test_startup_reconciles_orphaned_jobs(tmp_path) -> None:
    settings = make_settings(tmp_path)
    pre = JobStore(settings.data_dir / JOBS_DB_NAME)
    orphan = pre.create("bk-dead", "render")
    pre.mark_running(orphan.job_id)

    with TestClient(create_app(settings=settings)) as c:
        assert c.get("/api/system").json()["reconciled_at_startup"] == 1
        assert c.get("/api/system").json()["active_job"] is None
    assert pre.get(orphan.job_id).state is JobState.FAILED


# -- engines catalog --------------------------------------------------------------------


def test_engines_catalog(client) -> None:
    body = client.get("/api/engines").json()
    by_id = {e["engine_id"]: e for e in body["engines"]}
    assert set(by_id) == {"kokoro", "chatterbox", "indextts2", "elevenlabs"}

    assert by_id["kokoro"]["uses_gpu"] is True
    assert by_id["kokoro"]["requires_validation"] is False
    assert by_id["kokoro"]["paid"] is False
    assert by_id["kokoro"]["supports_cloning"] is False

    assert by_id["chatterbox"]["uses_gpu"] is True
    assert by_id["chatterbox"]["requires_validation"] is True
    assert by_id["chatterbox"]["supports_cloning"] is True

    # IndexTTS-2 (M7): local, autoregressive (validated), cloning, free. weights_cached is False
    # here (no checkpoints dir configured in the test settings), never None.
    assert by_id["indextts2"]["uses_gpu"] is True
    assert by_id["indextts2"]["requires_validation"] is True
    assert by_id["indextts2"]["supports_cloning"] is True
    assert by_id["indextts2"]["paid"] is False
    assert by_id["indextts2"]["weights_cached"] is False

    assert by_id["elevenlabs"]["uses_gpu"] is False
    assert by_id["elevenlabs"]["paid"] is True
    assert by_id["elevenlabs"]["weights_cached"] is None  # cloud: no local weights

    assert all(e["resident"] is False for e in body["engines"])


def test_kokoro_voices(client) -> None:
    body = client.get("/api/engines/kokoro/voices").json()
    assert body["engine_id"] == "kokoro"
    ids = [v["id"] for v in body["voices"]]
    assert len(ids) == 28
    assert "af_heart" in ids
    heart = next(v for v in body["voices"] if v["id"] == "af_heart")
    assert heart["language"] == "en-US"
    assert heart["gender"] == "female"
    # every preset carries an editorial note — the mixer's "what am I blending?"
    assert all(v["description"] for v in body["voices"])


def test_chatterbox_voices_empty(client) -> None:
    body = client.get("/api/engines/chatterbox/voices").json()
    assert body["voices"] == []


def test_elevenlabs_voices_without_key_is_503(client) -> None:
    resp = client.get("/api/engines/elevenlabs/voices")
    assert resp.status_code == 503
    err = _error(resp)
    assert err["code"] == "not_ready"
    assert "ELEVENLABS_API_KEY" in err["message"]


# -- error envelope ---------------------------------------------------------------------


def test_unknown_engine_is_enveloped_404(client) -> None:
    resp = client.get("/api/engines/fish/voices")
    assert resp.status_code == 404
    assert _error(resp)["code"] == "not_found"


def test_validation_error_is_enveloped_422(client) -> None:
    resp = client.get("/api/system", params={"probe": "not-a-bool"})
    assert resp.status_code == 422
    err = _error(resp)
    assert err["code"] == "invalid"
    assert err["detail"]  # pydantic error list, JSON-serializable


def test_unknown_path_is_enveloped_404(client) -> None:
    # Framework-raised 404s (unmatched path — e.g. an M6b-2 route during version skew)
    # must keep the envelope, not FastAPI's default {"detail": ...} shape.
    resp = client.get("/api/nonexistent")
    assert resp.status_code == 404
    assert _error(resp)["code"] == "not_found"


def test_wrong_method_is_enveloped_405(client) -> None:
    resp = client.post("/api/health")
    assert resp.status_code == 405
    assert _error(resp)["code"] == "method_not_allowed"
    assert "GET" in resp.headers["allow"]  # exc.headers pass through the envelope


def test_unhandled_exception_is_enveloped_500(tmp_path) -> None:
    app = create_app(settings=make_settings(tmp_path))

    @app.get("/api/boom")
    def boom() -> None:
        raise RuntimeError("secret-internal-detail")

    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/api/boom")
    assert resp.status_code == 500
    err = _error(resp)
    assert err["code"] == "internal"
    assert "secret-internal-detail" not in resp.text  # exception text never leaks


# -- engine registry / heavy-work gate --------------------------------------------------


def test_registry_builds_chatterbox_with_settings_voices_dir(tmp_path) -> None:
    # Consent invariant: the registry's chatterbox must resolve references from the SAME
    # voices root as the VoiceLibrary, or clone consent checks diverge from what renders.
    settings = make_settings(tmp_path)
    registry = EngineRegistry(settings)
    engine = registry.get("chatterbox")
    assert engine._voices_dir == settings.voices_dir
    assert registry.get("chatterbox") is engine  # process-lifetime instance, not a new one


def test_registry_residency_is_identity_with_gpu_manager(tmp_path) -> None:
    from seiyuu.gpu import GpuResourceManager

    gpu = GpuResourceManager()
    registry = EngineRegistry(make_settings(tmp_path), gpu_manager=gpu)
    engine = registry.get("chatterbox")
    assert not registry.is_resident("chatterbox")  # constructed != loaded

    with gpu.acquire(engine, "engine:chatterbox"):
        pass
    assert registry.is_resident("chatterbox")  # lazy release keeps it resident

    # Eviction truth (the stale-flag bug): a competitor acquire (an attribution run's
    # LLM) unloads the engine — residency must flip to cold WITHOUT anyone telling the
    # registry, or the M6b-6 cold-engine refusal would skip the warmup job.
    class _Llm:
        def unload(self) -> None:
            pass

    with gpu.acquire(_Llm(), "llm:qwen"):
        pass
    assert not registry.is_resident("chatterbox")


def test_registry_invalidate_drops_instance_and_residency(tmp_path) -> None:
    from seiyuu.gpu import GpuResourceManager

    gpu = GpuResourceManager()
    registry = EngineRegistry(make_settings(tmp_path), gpu_manager=gpu)
    first = registry.get("chatterbox")
    with gpu.acquire(first, "engine:chatterbox"):
        pass
    assert registry.is_resident("chatterbox")
    registry.invalidate("chatterbox")
    # the OLD instance is still the manager's resident, but it is no longer this
    # registry's instance — the new one reads cold until warmed
    assert not registry.is_resident("chatterbox")
    assert registry.get("chatterbox") is not first


def test_registry_unknown_engine_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown TTS engine"):
        EngineRegistry(make_settings(tmp_path)).get("fish")


def test_registry_elevenlabs_never_reads_global_settings(tmp_path, monkeypatch) -> None:
    # Injected-settings-governs invariant: a keyless injected Settings must yield an
    # UNCONFIGURED engine — the adapter's api_key=None fallback would silently adopt the
    # dev machine's real .env key (a paid-capable client under a config that reports
    # elevenlabs_configured=false). Review finding, reproduced live before the fix.
    monkeypatch.setattr(
        "seiyuu.settings.get_settings",
        lambda: pytest.fail("EngineRegistry read the global get_settings()"),
    )
    engine = EngineRegistry(make_settings(tmp_path)).get("elevenlabs")
    assert engine._api_key == ""  # falsy: _get_client() refuses loudly on first use


def test_heavy_work_gate_nonblocking_refusal() -> None:
    gate = HeavyWorkGate()
    with gate.hold("job"):
        assert gate.holder == "job"
        assert gate.audition_in_flight is False
        with gate.try_hold("audition") as acquired:
            assert acquired is False
            assert gate.audition_in_flight is False  # refused: never became the holder
    with gate.try_hold("audition") as acquired:
        assert acquired is True
        assert gate.audition_in_flight is True
    assert gate.holder == ""
