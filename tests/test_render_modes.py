"""Per-mode render manifests + the manifest.json active pointer (render-modes feature).

A completed render archives its manifest as manifest.single.json / manifest.multi.json AND
promotes the same content to manifest.json (rendering a mode activates it); switching modes
is a pure atomic archive copy through the service/API/CLI. Fixture-based and offline: fake
engines only, multivoice inputs borrowed from test_render_multivoice.
"""

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.api.main import create_app
from seiyuu.cli import main
from seiyuu.gpu import GpuResourceManager
from seiyuu.render import (
    MANIFEST_NAME,
    RenderError,
    manifest_name_for_mode,
    render_book,
    render_book_multivoice,
)
from seiyuu.repository import JobStore
from seiyuu.repository.jobs import JOBS_DB_NAME
from seiyuu.services.render_mode import (
    RenderModeConflict,
    RenderModeUnavailable,
    activate_render_mode,
    available_render_modes,
)
from seiyuu.voices import VoiceAssignment
from test_api_m6b1 import make_settings
from test_render_multivoice import _library, _patch_engine, _report

BOOK = "test-book-00000000"


def _assignment() -> VoiceAssignment:
    return VoiceAssignment(
        book_id=BOOK, narrator_voice_id="narrator_v", assignments={"alice": "alice_v"}
    )


def _render_multi(tmp_path, monkeypatch, out, chapters=()) -> None:
    _patch_engine(monkeypatch, FakeEngine())
    render_book_multivoice(
        _report(), make_book(), _library(tmp_path), _assignment(), out,
        chapters=chapters, gpu=GpuResourceManager(),
    )  # fmt: skip


def _read(path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "data" / JOBS_DB_NAME)


# -- pipeline: archive + promote ------------------------------------------------------------


def test_render_both_modes_coexist_and_last_render_activates(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    render_book(make_book(), FakeEngine(), "test_voice", out)
    single_raw = _read(out / manifest_name_for_mode("single"))
    assert _read(out / MANIFEST_NAME) == single_raw  # rendering single activates single

    _render_multi(tmp_path, monkeypatch, out)
    multi_raw = _read(out / manifest_name_for_mode("multi"))
    # both archives coexist; the LAST render's mode is active (Listen reads manifest.json)
    assert _read(out / manifest_name_for_mode("single")) == single_raw
    assert _read(out / MANIFEST_NAME) == multi_raw
    assert single_raw != multi_raw


def test_subset_merges_into_same_mode_archive_not_the_active_pointer(tmp_path, monkeypatch) -> None:
    # single ch1, then a full multi render flips the active pointer to multi; a single
    # SUBSET render of ch2 must still merge into the SINGLE archive (ch1 carried over) —
    # never refuse against, or merge into, the multi manifest the pointer names.
    out = tmp_path / "out"
    render_book(make_book(), FakeEngine(), "test_voice", out, chapters=(1,))
    _render_multi(tmp_path, monkeypatch, out)
    multi_raw = _read(out / manifest_name_for_mode("multi"))

    result = render_book(make_book(), FakeEngine(), "test_voice", out, chapters=(2,))

    assert [c.index for c in result.manifest.chapters] == [1, 2]  # merged with archived ch1
    single_raw = _read(out / manifest_name_for_mode("single"))
    assert _read(out / MANIFEST_NAME) == single_raw  # rendering single re-activated single
    assert _read(out / manifest_name_for_mode("multi")) == multi_raw  # multi untouched


def test_inconsistent_mode_archive_refused_before_synthesis(tmp_path, monkeypatch) -> None:
    # Unreachable via normal flows (renders always archive under their own mode): a
    # hand-edited manifest.multi.json holding a single-voice manifest. The old cross-mode
    # refusal survives as this internal consistency guard.
    out = tmp_path / "out"
    render_book(make_book(), FakeEngine(), "test_voice", out)
    (out / manifest_name_for_mode("multi")).write_text(
        _read(out / manifest_name_for_mode("single")), encoding="utf-8"
    )
    fake = FakeEngine()
    _patch_engine(monkeypatch, fake)
    with pytest.raises(RenderError, match="inconsistent"):
        render_book_multivoice(
            _report(), make_book(), _library(tmp_path), _assignment(), out,
            chapters=(2,), gpu=GpuResourceManager(),
        )  # fmt: skip
    assert fake.calls == []


def test_corrupt_same_mode_archive_fails_loudly(tmp_path) -> None:
    out = tmp_path / "out"
    render_book(make_book(), FakeEngine(), "test_voice", out)
    (out / manifest_name_for_mode("single")).write_text("{not json", encoding="utf-8")
    with pytest.raises(RenderError, match="unreadable"):
        render_book(make_book(), FakeEngine(), "test_voice", out, chapters=(2,))


# -- service: activate_render_mode -----------------------------------------------------------


def test_switch_restores_the_other_modes_manifest(tmp_path, monkeypatch, store) -> None:
    out_root = tmp_path / "output"
    out = out_root / BOOK
    render_book(make_book(), FakeEngine(), "test_voice", out)
    _render_multi(tmp_path, monkeypatch, out)
    single_raw = _read(out / manifest_name_for_mode("single"))
    assert _read(out / MANIFEST_NAME) != single_raw  # multi is active

    result = activate_render_mode(out_root, BOOK, "single", store=store)
    assert (result.mode, result.changed, result.chapters) == ("single", True, 2)
    assert _read(out / MANIFEST_NAME) == single_raw  # instant fallback, no synthesis

    again = activate_render_mode(out_root, BOOK, "single", store=store)
    assert again.changed is False  # already active


def test_switch_refuses_never_rendered_mode(tmp_path, store) -> None:
    out_root = tmp_path / "output"
    out = out_root / BOOK
    render_book(make_book(), FakeEngine(), "test_voice", out)
    before = _read(out / MANIFEST_NAME)
    with pytest.raises(RenderModeUnavailable, match="render that mode first"):
        activate_render_mode(out_root, BOOK, "multi", store=store)
    assert _read(out / MANIFEST_NAME) == before  # pointer untouched


@pytest.mark.parametrize("kind", ["render", "assemble", "master"])
def test_switch_refused_while_manifest_job_live(tmp_path, store, kind) -> None:
    out_root = tmp_path / "output"
    render_book(make_book(), FakeEngine(), "test_voice", out_root / BOOK)
    store.mark_running(store.create(BOOK, kind).job_id)
    with pytest.raises(RenderModeConflict, match=kind):
        activate_render_mode(out_root, BOOK, "single", store=store)


def test_attribute_job_does_not_block_switch(tmp_path, store) -> None:
    # attribution never touches output/{id}/manifest.json — it must not block the switch
    out_root = tmp_path / "output"
    render_book(make_book(), FakeEngine(), "test_voice", out_root / BOOK)
    store.mark_running(store.create(BOOK, "attribute").job_id)
    assert activate_render_mode(out_root, BOOK, "single", store=store).mode == "single"


def test_switch_prefeature_active_mode_is_graceful(tmp_path, store) -> None:
    # pre-feature book: manifest.json only, no archives. Activating ITS mode is a no-op
    # that materializes the archive; the other mode stays unavailable.
    out_root = tmp_path / "output"
    out = out_root / BOOK
    render_book(make_book(), FakeEngine(), "test_voice", out)
    (out / manifest_name_for_mode("single")).unlink()
    raw = _read(out / MANIFEST_NAME)

    result = activate_render_mode(out_root, BOOK, "single", store=store)
    assert (result.changed, result.chapters) == (False, 2)
    assert _read(out / manifest_name_for_mode("single")) == raw
    assert _read(out / MANIFEST_NAME) == raw
    with pytest.raises(RenderModeUnavailable):
        activate_render_mode(out_root, BOOK, "multi", store=store)


def test_switch_away_from_prefeature_active_preserves_it_as_archive(
    tmp_path, monkeypatch, store
) -> None:
    out_root = tmp_path / "output"
    out = out_root / BOOK
    _render_multi(tmp_path, monkeypatch, out)
    render_book(make_book(), FakeEngine(), "test_voice", out)  # single archived + active
    (out / manifest_name_for_mode("single")).unlink()  # make single pre-feature (pointer only)
    single_raw = _read(out / MANIFEST_NAME)

    activate_render_mode(out_root, BOOK, "multi", store=store)
    # the switch preserved the only copy of the single render before moving the pointer
    assert _read(out / manifest_name_for_mode("single")) == single_raw
    assert _read(out / MANIFEST_NAME) == _read(out / manifest_name_for_mode("multi"))
    assert available_render_modes(out) == ["single", "multi"]


def test_available_modes_counts_archives_and_prefeature_pointer(tmp_path) -> None:
    out = tmp_path / "output" / BOOK
    assert available_render_modes(out) == []
    render_book(make_book(), FakeEngine(), "test_voice", out)
    assert available_render_modes(out) == ["single"]
    (out / manifest_name_for_mode("single")).unlink()
    assert available_render_modes(out) == ["single"]  # lazy migration: the pointer counts


# -- API: summary fields + POST /render/mode --------------------------------------------------


def test_api_summary_reports_modes_and_switch_round_trips(tmp_path, monkeypatch) -> None:
    cfg = make_settings(tmp_path)
    out = cfg.output_dir / BOOK
    render_book(make_book(), FakeEngine(), "test_voice", out)
    _render_multi(tmp_path, monkeypatch, out)
    single_raw = _read(out / manifest_name_for_mode("single"))

    app = create_app(settings=cfg)
    with TestClient(app) as client:
        body = client.get(f"/api/books/{BOOK}/render").json()
        assert body["mode"] == "multivoice"
        assert body["active_mode"] == "multi"
        assert body["available_modes"] == ["single", "multi"]

        switched = client.post(f"/api/books/{BOOK}/render/mode", json={"mode": "single"})
        assert switched.status_code == 200, switched.text
        body = switched.json()
        assert body["mode"] == "single"
        assert body["active_mode"] == "single"
        assert body["available_modes"] == ["single", "multi"]
        assert body["voice_id"] == "test_voice"
        assert _read(out / MANIFEST_NAME) == single_raw


def test_api_switch_unrendered_mode_404(tmp_path) -> None:
    cfg = make_settings(tmp_path)
    render_book(make_book(), FakeEngine(), "test_voice", cfg.output_dir / BOOK)
    app = create_app(settings=cfg)
    with TestClient(app) as client:
        resp = client.post(f"/api/books/{BOOK}/render/mode", json={"mode": "multi"})
        assert resp.status_code == 404
        err = resp.json()["error"]
        assert err["code"] == "not_found"
        assert "render that mode first" in err["message"]


def test_api_switch_conflicting_job_409(tmp_path) -> None:
    cfg = make_settings(tmp_path)
    render_book(make_book(), FakeEngine(), "test_voice", cfg.output_dir / BOOK)
    app = create_app(settings=cfg)
    with TestClient(app) as client:
        store: JobStore = app.state.store
        job = store.create(BOOK, "assemble")
        store.mark_running(job.job_id)
        resp = client.post(f"/api/books/{BOOK}/render/mode", json={"mode": "single"})
        assert resp.status_code == 409
        err = resp.json()["error"]
        assert err["code"] == "conflicting_job"
        assert err["detail"]["job_id"] == job.job_id


# -- CLI ---------------------------------------------------------------------------------------


def test_cli_render_mode_switch_and_refusal(tmp_path, monkeypatch) -> None:
    cfg = make_settings(tmp_path)
    monkeypatch.setattr("seiyuu.settings.get_settings", lambda: cfg)
    render_book(make_book(), FakeEngine(), "test_voice", cfg.output_dir / BOOK)

    ok = CliRunner().invoke(main, ["render-mode", BOOK, "single"])
    assert ok.exit_code == 0, ok.output
    assert "single-voice render already active" in ok.output

    refused = CliRunner().invoke(main, ["render-mode", BOOK, "multi"])
    assert refused.exit_code != 0
    assert "render that mode first" in refused.output
