"""Ollama pre-flight: a stopped server or an unpulled model fails fast with the fix.

The conftest autouse stub keeps the rest of the suite offline; every test here restores
the REAL ``ollama_preflight`` and fakes only the HTTP GET (``local._urllib_get``), so the
tags parsing, name matching, and error mapping are exercised end to end — no network.
"""

import time
import urllib.error

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import seiyuu.attribute.providers.local as local
import seiyuu.services.attribution as svc
from conftest import build_synthetic_epub
from factories import make_book
from fake_provider import FakeProvider
from seiyuu.api.main import create_app
from seiyuu.attribute.cache import AttributionCache
from seiyuu.attribute.models import ChunkAttribution, Segment, SegmentType
from seiyuu.attribute.providers.base import AttributionError
from seiyuu.attribute.providers.local import (
    OllamaModelMissingError,
    OllamaPreflightError,
    OllamaUnreachableError,
    canonical_model_name,
    ollama_preflight,
)
from seiyuu.cli import main
from seiyuu.repository.jobs import JobState
from test_api_m6b1 import make_settings

BASE = "http://localhost:11434/v1"


class FakeGet:
    """Records GETs; returns a canned /api/tags payload or raises a connection error."""

    def __init__(self, names: list[str] | None = None, *, down: bool = False) -> None:
        self.names = names or []
        self.down = down
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> dict:
        self.calls.append((url, timeout))
        if self.down:
            raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))
        return {"models": [{"name": n, "model": n} for n in self.names]}


def _quoted_book():
    """make_book() with real quoted dialogue so chunks reach the provider: a quote-free book
    takes the narration fast path and never consults it (or the chunk cache)."""
    book = make_book()
    book.chapters[0].blocks[1].text = '"Hello world."'
    book.chapters[1].blocks[1].text = '"Second chapter."'
    return book


def _narrate(chunk, registry, attempt) -> ChunkAttribution:
    return ChunkAttribution(
        segments=[
            Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text)
            for b in chunk.owned_blocks
        ],
        characters=[],
    )


@pytest.fixture
def real_preflight(monkeypatch):
    """Undo the conftest stub; install a fake GET the test configures."""
    monkeypatch.setattr(svc, "ollama_preflight", ollama_preflight)

    def install(fake: FakeGet) -> FakeGet:
        monkeypatch.setattr(local, "_urllib_get", fake)
        return fake

    return install


# -- the provider-module function ---------------------------------------------------------


def test_reachable_and_model_present_passes_with_one_cheap_tags_read() -> None:
    fake = FakeGet(["qwen2.5:7b", "nomic-embed-text:latest"])
    ollama_preflight(BASE, ["qwen2.5:7b"], get=fake)
    # GET /api/tags at the server ROOT (not the /v1 shim), short timeout, nothing else —
    # never a generation call (that would load the model onto the GPU)
    assert fake.calls == [("http://localhost:11434/api/tags", 3.0)]


def test_unreachable_raises_typed_error_with_fix() -> None:
    with pytest.raises(OllamaUnreachableError) as info:
        ollama_preflight(BASE, ["qwen2.5:7b"], get=FakeGet(down=True))
    err = info.value
    assert err.code == "ollama_unreachable"
    assert "isn't reachable at http://localhost:11434" in str(err)
    assert "'ollama serve'" in str(err)
    assert isinstance(err, OllamaPreflightError)


def test_model_missing_raises_typed_error_with_pull_command() -> None:
    with pytest.raises(OllamaModelMissingError) as info:
        ollama_preflight(BASE, ["qwen3.5:9b"], get=FakeGet(["qwen2.5:7b"]))
    err = info.value
    assert err.code == "model_missing"
    assert err.model == "qwen3.5:9b"
    assert "'ollama pull qwen3.5:9b'" in str(err)


@pytest.mark.parametrize(
    ("configured", "tags", "ok"),
    [
        ("llama3", ["llama3:latest"], True),  # bare name == :latest
        ("llama3:latest", ["llama3"], True),  # and the other way round
        ("Qwen2.5:7B", ["qwen2.5:7b"], True),  # Ollama names are case-insensitive
        ("qwen2.5", ["qwen2.5:7b"], False),  # bare = :latest, NOT any tag
        ("qwen2.5:7b", ["qwen2.5:14b"], False),
        ("registry.ollama.ai/library/mistral:7b", ["mistral:7b"], True),
        ("me/custom", ["me/custom:latest"], True),
        ("localhost:5000/m", ["localhost:5000/m:latest"], True),  # port is not a tag
    ],
)
def test_model_name_matching_follows_ollama_naming(configured, tags, ok) -> None:
    fake = FakeGet(tags)
    if ok:
        ollama_preflight(BASE, [configured], get=fake)
    else:
        with pytest.raises(OllamaModelMissingError):
            ollama_preflight(BASE, [configured], get=fake)


def test_canonical_name() -> None:
    assert canonical_model_name("qwen2.5") == "qwen2.5:latest"
    assert canonical_model_name("qwen2.5:7b") == "qwen2.5:7b"


def test_matches_on_model_field_when_name_differs() -> None:
    def get(url, timeout):
        return {"models": [{"name": "alias-name", "model": "qwen2.5:7b"}]}

    ollama_preflight(BASE, ["qwen2.5:7b"], get=get)


# -- service seam: which runs pre-flight --------------------------------------------------


def test_service_checks_local_primary_model(tmp_path, real_preflight) -> None:
    cfg = make_settings(tmp_path)
    fake = real_preflight(FakeGet([]))
    with pytest.raises(OllamaModelMissingError) as info:
        svc.preflight_attribution(cfg)
    assert info.value.model == cfg.attribution_model
    assert len(fake.calls) == 1


def test_service_skips_anthropic_provider(tmp_path, real_preflight) -> None:
    fake = real_preflight(FakeGet(down=True))
    svc.preflight_attribution(make_settings(tmp_path), provider_id="anthropic")
    assert fake.calls == []  # no Ollama call at all


def test_service_hybrid_still_checks_local(tmp_path, real_preflight) -> None:
    real_preflight(FakeGet(down=True))
    cfg = make_settings(tmp_path, attribution_hybrid=True)
    with pytest.raises(OllamaUnreachableError):
        svc.preflight_attribution(cfg, provider_id="local")


def test_service_checks_local_adjudicator_on_full_book_only(tmp_path, real_preflight) -> None:
    real_preflight(FakeGet(["qwen2.5:7b"]))
    cfg = make_settings(tmp_path, adjudication_provider="local", adjudication_model="judge:1b")
    with pytest.raises(OllamaModelMissingError) as info:
        svc.preflight_attribution(cfg, use_adjudicate=True)
    assert info.value.model == "judge:1b"
    # a --chapter subset never adjudicates (run_attribution), so its model isn't needed
    svc.preflight_attribution(cfg, use_adjudicate=True, chapters=(1,))
    # anthropic primary + local adjudicator: only the adjudicator is checked
    with pytest.raises(OllamaModelMissingError):
        svc.preflight_attribution(cfg, provider_id="anthropic", use_adjudicate=True)


def test_run_attribution_fails_fast_before_building_provider(
    tmp_path, real_preflight, monkeypatch
) -> None:
    real_preflight(FakeGet(down=True))

    def no_build(*_a, **_k):
        raise AssertionError("provider must not be built when Ollama is down")

    monkeypatch.setattr(svc, "build_provider", no_build)
    book = make_book()
    with pytest.raises(OllamaUnreachableError):
        svc.run_attribution(book, tmp_path / "book", cfg=make_settings(tmp_path))
    assert not (tmp_path / "book" / "attribution.db").exists()


def test_run_attribution_with_injected_provider_skips_preflight(tmp_path, real_preflight) -> None:
    fake = real_preflight(FakeGet(down=True))
    provider = FakeProvider(_narrate)
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    report = svc.run_attribution(
        _quoted_book(), book_dir, cfg=make_settings(tmp_path), provider=provider
    )
    assert report.chapters
    assert provider.calls
    assert fake.calls == []


def test_cached_rerun_only_warns_when_ollama_down(tmp_path, real_preflight, monkeypatch) -> None:
    # A re-run (attribution.db exists) can be served wholly from the chunk cache, as before
    # the pre-flight existed — e.g. `convert --multivoice` again after recasting voices.
    book, cfg, book_dir = _quoted_book(), make_settings(tmp_path), tmp_path / "book"
    book_dir.mkdir()
    svc.run_attribution(book, book_dir, cfg=cfg, provider=FakeProvider(_narrate))  # warm

    fake = real_preflight(FakeGet(down=True))

    def must_not_call(*_a):
        raise AssertionError("every chunk is cached; no LLM call may happen")

    replay = FakeProvider(must_not_call)  # same provider/model/prompt ids -> cache hits
    monkeypatch.setattr(svc, "build_provider", lambda *_a, **_k: replay)
    said: list[str] = []
    report = svc.run_attribution(book, book_dir, cfg=cfg, progress=said.append)
    assert report.chapters
    assert len(fake.calls) == 1  # the check still ran ...
    assert any("warning:" in m and "ollama serve" in m for m in said)  # ... and only warned
    assert replay.calls == []


def test_rerun_cache_miss_still_fails_loudly(tmp_path, real_preflight, monkeypatch) -> None:
    real_preflight(FakeGet(down=True))
    book_dir = tmp_path / "book"
    AttributionCache(book_dir / "attribution.db").close()  # a prior run's (empty) cache

    def offline(*_a):
        raise AttributionError("cannot reach Ollama: is `ollama serve` running?")

    monkeypatch.setattr(svc, "build_provider", lambda *_a, **_k: FakeProvider(offline))
    with pytest.raises(AttributionError, match="ollama serve"):
        svc.run_attribution(_quoted_book(), book_dir, cfg=make_settings(tmp_path))


# -- CLI ----------------------------------------------------------------------------------


def test_cli_attribute_prints_the_fix(tmp_path, real_preflight, monkeypatch) -> None:
    real_preflight(FakeGet(["something-else:latest"]))
    book = make_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    (book_dir / "normalized.json").write_text(book.model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(
        main, ["attribute", "test-book", "--books-dir", str(tmp_path / "books")]
    )
    assert result.exit_code != 0
    assert "ollama pull" in result.output


# -- POST /api/books/{id}/attribute -------------------------------------------------------


@pytest.fixture(scope="module")
def epub_bytes(tmp_path_factory) -> bytes:
    return build_synthetic_epub(tmp_path_factory.mktemp("epub") / "s.epub").read_bytes()


def _app_with_book(tmp_path, epub_bytes, **overrides):
    app = create_app(settings=make_settings(tmp_path, **overrides))
    client = TestClient(app)
    client.__enter__()
    resp = client.post("/api/books", files={"file": ("s.epub", epub_bytes, "application/epub+zip")})
    assert resp.status_code == 201, resp.text
    return client, resp.json()["book"]["book_id"]


@pytest.mark.parametrize(
    ("fake", "code", "fix"),
    [
        (FakeGet(down=True), "ollama_unreachable", "ollama serve"),
        (FakeGet(["other:latest"]), "model_missing", "ollama pull qwen2.5:7b"),
    ],
)
def test_route_returns_503_before_enqueuing(
    tmp_path, epub_bytes, real_preflight, fake, code, fix
) -> None:
    real_preflight(FakeGet(fake.names, down=fake.down))
    client, book_id = _app_with_book(tmp_path, epub_bytes)
    try:
        resp = client.post(f"/api/books/{book_id}/attribute", json={})
        assert resp.status_code == 503
        error = resp.json()["error"]
        assert error["code"] == code
        assert fix in error["message"]
        assert client.get("/api/jobs").json()["jobs"] == []  # nothing enqueued
    finally:
        client.__exit__(None, None, None)


def test_route_anthropic_provider_skips_preflight(
    tmp_path, epub_bytes, real_preflight, monkeypatch
) -> None:
    fake = real_preflight(FakeGet(down=True))
    monkeypatch.setattr("seiyuu.api.handlers.run_attribution", lambda *a, **k: None)
    client, book_id = _app_with_book(tmp_path, epub_bytes, anthropic_api_key="sk-test")
    try:
        resp = client.post(
            f"/api/books/{book_id}/attribute",
            json={"provider": "anthropic", "confirm_paid": True},
        )
        assert resp.status_code == 202, resp.text
        assert fake.calls == []
    finally:
        client.__exit__(None, None, None)


def test_route_enqueues_when_ollama_ready(
    tmp_path, epub_bytes, real_preflight, monkeypatch
) -> None:
    fake = real_preflight(FakeGet(["qwen2.5:7b"]))
    monkeypatch.setattr("seiyuu.api.handlers.run_attribution", lambda *a, **k: None)
    client, book_id = _app_with_book(tmp_path, epub_bytes)
    try:
        resp = client.post(f"/api/books/{book_id}/attribute", json={})
        assert resp.status_code == 202, resp.text
        assert len(fake.calls) == 1
    finally:
        client.__exit__(None, None, None)


def test_job_runs_the_adjudication_setting_the_route_checked(
    tmp_path, epub_bytes, real_preflight, monkeypatch
) -> None:
    # Server default adjudicates with an UNPULLED local judge; a request opting out must
    # pass the route AND the job must honor the opt-out (it used to fall back to the server
    # default and fail on the unpulled model in the background).
    real_preflight(FakeGet(["qwen2.5:7b"]))
    seen: dict = {}
    monkeypatch.setattr("seiyuu.api.handlers.run_attribution", lambda *a, **k: seen.update(k))
    client, book_id = _app_with_book(
        tmp_path,
        epub_bytes,
        attribution_adjudicate=True,
        adjudication_provider="local",
        adjudication_model="judge:1b",
    )
    try:
        refused = client.post(f"/api/books/{book_id}/attribute", json={})
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "model_missing"
        resp = client.post(f"/api/books/{book_id}/attribute", json={"use_adjudicate": False})
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]
        store = client.app.state.store
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not store.get(job_id).is_terminal:
            time.sleep(0.02)
        assert store.get(job_id).state == JobState.SUCCEEDED
        assert seen["use_adjudicate"] is False
    finally:
        client.__exit__(None, None, None)


# -- /api/system?probe=true ---------------------------------------------------------------


def test_system_probe_reports_model_installed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("seiyuu.api.routes.system._probe_ollama", lambda _url: True)
    fake = FakeGet(["qwen2.5:7b"])
    monkeypatch.setattr(local, "_urllib_get", fake)
    with TestClient(create_app(settings=make_settings(tmp_path))) as client:
        assert client.get("/api/system").json()["ollama"]["model_installed"] is None
        assert fake.calls == []  # default poll stays network-free
        body = client.get("/api/system", params={"probe": True}).json()["ollama"]
        assert body == {
            "base_url": BASE,
            "reachable": True,
            "model_installed": True,
        }
        fake.names = []
        body = client.get("/api/system", params={"probe": True}).json()["ollama"]
        assert body["model_installed"] is False


def test_system_probe_unreachable_leaves_model_installed_unknown(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("seiyuu.api.routes.system._probe_ollama", lambda _url: False)
    fake = FakeGet(["qwen2.5:7b"])
    monkeypatch.setattr(local, "_urllib_get", fake)
    with TestClient(create_app(settings=make_settings(tmp_path))) as client:
        body = client.get("/api/system", params={"probe": True}).json()["ollama"]
        assert body["reachable"] is False
        assert body["model_installed"] is None
        assert fake.calls == []
