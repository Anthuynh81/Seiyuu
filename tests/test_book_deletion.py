"""F3 — guarded book deletion: the DELETE route, the two-root purge helper, terminal-only
jobs-row reaping, paid-artifact detection + the 402 second-confirm gate, and CLI parity.

Fixture-based and offline: no live LLM/TTS. Books are scaffolded directly across both roots
(books/{id} + output/{id}); paid renders are faked as a manifest + SegmentKey cache sidecars.
"""

import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from seiyuu.cli import main
from seiyuu.render.cache import SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest
from seiyuu.repository import JobState, JobStore
from seiyuu.repository import books as books_mod
from seiyuu.repository.books import RepositoryError, delete_book_trees
from seiyuu.repository.jobs import JOBS_DB_NAME
from seiyuu.settings import Settings


def make_settings(tmp_path, **overrides) -> Settings:
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
def cfg(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def client(cfg):
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        c.app = app
        yield c


def _error(resp) -> dict:
    body = resp.json()
    assert set(body) == {"error"}, body
    return body["error"]


# -- scaffolding helpers ------------------------------------------------------------------


def _seed_books_root(cfg: Settings, book_id: str) -> None:
    d = cfg.books_dir / book_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "normalized.json").write_text(
        json.dumps({"book_meta": {"title": "T", "authors": ["A"]}, "chapters": []}),
        encoding="utf-8",
    )


def _seed_output_marker(cfg: Settings, book_id: str) -> None:
    o = cfg.output_dir / book_id
    o.mkdir(parents=True, exist_ok=True)
    (o / "manifest.json").write_text("{}", encoding="utf-8")


def _seed_render(cfg: Settings, book_id: str, *, engine: str, voice_id: str, n: int = 2) -> None:
    """A manifest + n matching SegmentKey cache sidecars for one engine/voice."""
    o = cfg.output_dir / book_id
    cache = o / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    segs = []
    for i in range(n):
        key = SegmentKey(
            engine=engine,
            engine_model_version=f"{engine}-x",
            voice_id=voice_id,
            settings_hash="s",
            seed=1,
            normalized_text_hash=f"h{i}",
        )
        stem = key.key_hash
        (cache / f"{stem}.wav").write_bytes(b"RIFFfake")
        (cache / f"{stem}.json").write_text(key.model_dump_json(), encoding="utf-8")
        segs.append(
            RenderedSegment(
                block_id=f"ch001_b{i:04d}",
                type="paragraph",
                wav=f"cache/{stem}.wav",
                duration_seconds=1.0,
                voice_id=voice_id,
            )
        )
    manifest = RenderManifest(
        book_id=book_id,
        engine=engine,
        engine_model_version=f"{engine}-x",
        voice_id=voice_id,
        seed=1,
        chapters=[RenderedChapter(index=1, title="Chapter 1", segments=segs)],
    )
    (o / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")


# -- live-job guard (D7, kind-agnostic) ---------------------------------------------------


@pytest.mark.parametrize("running", [False, True])
def test_delete_refused_while_job_live(client, cfg, running) -> None:
    _seed_books_root(cfg, "bk")
    store: JobStore = client.app.state.store
    job = store.create("bk", "attribute")
    if running:
        store.mark_running(job.job_id)
    resp = client.delete("/api/books/bk")
    assert resp.status_code == 409
    err = _error(resp)
    assert err["code"] == "conflicting_job"
    assert err["detail"]["job_id"] == job.job_id
    assert (cfg.books_dir / "bk").is_dir()  # nothing removed


def test_other_books_job_and_warmup_do_not_block(client, cfg) -> None:
    _seed_books_root(cfg, "bk")
    store: JobStore = client.app.state.store
    store.mark_running(store.create("other", "render").job_id)  # different book
    store.mark_running(store.create("engine:elevenlabs", "warmup").job_id)  # engine, not a book
    resp = client.delete("/api/books/bk")
    assert resp.status_code == 200, resp.json()
    assert not (cfg.books_dir / "bk").exists()


# -- two-root ghost removal + preservation ------------------------------------------------


def test_books_only_book_removed(client, cfg) -> None:
    _seed_books_root(cfg, "bk")
    body = client.delete("/api/books/bk").json()
    assert body["books_removed"] is True
    assert body["output_removed"] is False
    assert not (cfg.books_dir / "bk").exists()


def test_output_only_ghost_removed(client, cfg) -> None:
    _seed_output_marker(cfg, "bk")
    body = client.delete("/api/books/bk").json()
    assert body["output_removed"] is True
    assert body["books_removed"] is False
    assert not (cfg.output_dir / "bk").exists()


def test_both_roots_removed_and_neighbors_preserved(client, cfg) -> None:
    _seed_books_root(cfg, "bk")
    _seed_output_marker(cfg, "bk")
    _seed_books_root(cfg, "keep")
    _seed_output_marker(cfg, "keep")
    (cfg.voices_dir / "v1").mkdir(parents=True)
    (cfg.voices_dir / "v1" / "meta.json").write_text("{}", encoding="utf-8")
    jobs_db = cfg.data_dir / JOBS_DB_NAME

    body = client.delete("/api/books/bk").json()
    assert body["output_removed"] and body["books_removed"]
    assert not (cfg.books_dir / "bk").exists()
    assert not (cfg.output_dir / "bk").exists()
    # the shared voice library, the jobs.db FILE, and OTHER books survive
    assert (cfg.voices_dir / "v1" / "meta.json").is_file()
    assert jobs_db.is_file()
    assert (cfg.books_dir / "keep").is_dir()
    assert (cfg.output_dir / "keep").is_dir()


def test_delete_unknown_book_404(client) -> None:
    assert client.delete("/api/books/nope").status_code == 404


# -- terminal-only jobs-row reaping -------------------------------------------------------


def test_delete_jobs_for_book_only_terminal(tmp_path) -> None:
    store = JobStore(tmp_path / JOBS_DB_NAME)
    queued = store.create("bk", "render")
    running = store.create("bk", "attribute")
    store.mark_running(running.job_id)
    succeeded = store.create("bk", "assemble")
    store.mark_running(succeeded.job_id)
    store.finish(succeeded.job_id, JobState.SUCCEEDED)
    failed = store.create("bk", "master")
    store.mark_running(failed.job_id)
    store.finish(failed.job_id, JobState.FAILED, error="boom")
    canceled = store.create("bk", "render")
    store.request_cancel(canceled.job_id)  # queued -> canceled immediately
    store.create("other", "render")  # different book, terminal set untouched

    other_succeeded = store.create("other", "assemble")
    store.mark_running(other_succeeded.job_id)
    store.finish(other_succeeded.job_id, JobState.SUCCEEDED)

    removed = store.delete_jobs_for_book("bk")
    assert removed == 3  # succeeded + failed + canceled only
    remaining = {j.job_id for j in store.list_jobs(book_id="bk")}
    assert remaining == {queued.job_id, running.job_id}
    assert len(store.list_jobs(book_id="other")) == 2  # other book untouched


def test_delete_reaps_terminal_rows_and_returns_count(client, cfg) -> None:
    _seed_books_root(cfg, "bk")
    store: JobStore = client.app.state.store
    done = store.create("bk", "attribute")
    store.mark_running(done.job_id)
    store.finish(done.job_id, JobState.SUCCEEDED)
    body = client.delete("/api/books/bk").json()
    assert body["jobs_rows_deleted"] == 1
    assert store.list_jobs(book_id="bk") == []


# -- paid gate (D2) -----------------------------------------------------------------------


def test_paid_book_gated_then_confirmed(client, cfg) -> None:
    _seed_books_root(cfg, "bk")
    _seed_render(cfg, "bk", engine="elevenlabs", voice_id="v-el", n=2)

    gated = client.delete("/api/books/bk")
    assert gated.status_code == 402
    err = _error(gated)
    assert err["code"] == "payment_confirmation_required"
    assert err["detail"]["paid_segment_count"] == 2
    assert err["detail"]["paid_voice_ids"] == ["v-el"]
    assert err["detail"]["engines"] == ["elevenlabs"]
    assert (cfg.books_dir / "bk").is_dir()  # nothing removed without confirmation

    ok = client.delete("/api/books/bk", params={"confirm_paid": True})
    assert ok.status_code == 200
    body = ok.json()
    assert body["paid_segments_discarded"] == 2
    assert not (cfg.books_dir / "bk").exists()
    assert not (cfg.output_dir / "bk").exists()


def test_free_kokoro_book_deletes_in_one_call(client, cfg) -> None:
    _seed_books_root(cfg, "bk")
    _seed_render(cfg, "bk", engine="kokoro", voice_id="af_heart", n=3)
    resp = client.delete("/api/books/bk")  # no confirm_paid
    assert resp.status_code == 200
    assert resp.json()["paid_segments_discarded"] == 0
    assert not (cfg.output_dir / "bk").exists()


def test_stale_paid_cache_gates_after_free_rerender(client, cfg) -> None:
    """A paid render followed by a free (kokoro) re-render overwrites the manifest to kokoro
    but leaves the elevenlabs SegmentKey sidecars behind (the content-addressed cache is never
    pruned). Deletion must STILL gate on those stale paid sidecars — the cache is authoritative,
    not the current manifest."""
    _seed_books_root(cfg, "bk")
    _seed_render(cfg, "bk", engine="elevenlabs", voice_id="v-el", n=2)  # paid render
    _seed_render(cfg, "bk", engine="kokoro", voice_id="af_heart", n=3)  # free re-render
    # manifest now names kokoro, but the elevenlabs sidecars coexist in cache/
    cache = cfg.output_dir / "bk" / "cache"
    assert sum(1 for p in cache.glob("*.json")) == 5  # 2 paid + 3 free sidecars

    gated = client.delete("/api/books/bk")  # no confirm_paid
    assert gated.status_code == 402
    err = _error(gated)
    assert err["code"] == "payment_confirmation_required"
    assert err["detail"]["paid_segment_count"] == 2
    assert err["detail"]["paid_voice_ids"] == ["v-el"]
    assert err["detail"]["engines"] == ["elevenlabs"]
    assert (cfg.books_dir / "bk").is_dir()  # nothing removed without confirmation

    ok = client.delete("/api/books/bk", params={"confirm_paid": True})
    assert ok.status_code == 200
    assert ok.json()["paid_segments_discarded"] == 2
    assert not (cfg.output_dir / "bk").exists()


def test_paid_manifest_gates_when_cache_pruned(cfg) -> None:
    """Defensive fallback: if the cache dir is gone entirely, the manifest is the last proof
    of paid work and must still gate (count from its paid segments)."""
    from seiyuu.services.deletion import detect_paid_artifacts

    _seed_render(cfg, "bk", engine="elevenlabs", voice_id="v-el", n=2)
    import shutil as _shutil

    _shutil.rmtree(cfg.output_dir / "bk" / "cache")  # cache evicted, manifest remains

    paid = detect_paid_artifacts(cfg, "bk")
    assert paid.paid_segment_count == 2
    assert paid.paid_voice_ids == ["v-el"]
    assert paid.engines == ["elevenlabs"]


def test_paid_mode_archive_gates_when_cache_pruned_and_active_is_free(cfg) -> None:
    """Per-mode archives are paid-work proof too: cache gone, active pointer on a free
    multivoice render — the archived paid single render must still gate."""
    import shutil as _shutil

    from seiyuu.render import manifest_name_for_mode
    from seiyuu.services.deletion import detect_paid_artifacts

    # the fallback scan mirrors the archive names as literals; assert they stay in sync
    assert manifest_name_for_mode("single") == "manifest.single.json"
    assert manifest_name_for_mode("multi") == "manifest.multi.json"

    _seed_render(cfg, "bk", engine="elevenlabs", voice_id="v-el", n=2)
    o = cfg.output_dir / "bk"
    (o / "manifest.single.json").write_text(
        (o / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    # a free MULTIVOICE render then took over the active pointer (and its own archive)
    free = RenderManifest(
        book_id="bk",
        chapters=[
            RenderedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    RenderedSegment(
                        block_id="ch001_b0000",
                        type="paragraph",
                        wav="cache/x.wav",
                        duration_seconds=1.0,
                        voice_id="af_heart",
                    )
                ],
            )
        ],
        voices_used={
            "af_heart": {"engine": "kokoro", "engine_model_version": "k-x", "kind": "preset"}
        },
    )
    (o / "manifest.multi.json").write_text(free.model_dump_json(), encoding="utf-8")
    (o / "manifest.json").write_text(free.model_dump_json(), encoding="utf-8")
    _shutil.rmtree(o / "cache")  # cache evicted: the manifests are the last proof

    paid = detect_paid_artifacts(cfg, "bk")
    assert paid.paid_segment_count == 2  # the archived paid render still gates
    assert paid.engines == ["elevenlabs"]
    assert paid.paid_voice_ids == ["v-el"]


def test_paid_fallback_does_not_double_count_the_active_pointer(cfg) -> None:
    """manifest.json is a byte copy of the single archive (the normal post-render state);
    the fallback scan must count that render once, not twice."""
    import shutil as _shutil

    from seiyuu.services.deletion import detect_paid_artifacts

    _seed_render(cfg, "bk", engine="elevenlabs", voice_id="v-el", n=2)
    o = cfg.output_dir / "bk"
    (o / "manifest.single.json").write_text(
        (o / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _shutil.rmtree(o / "cache")

    paid = detect_paid_artifacts(cfg, "bk")
    assert paid.paid_segment_count == 2  # once, not doubled by the pointer copy


def test_delete_removes_mode_archives_with_output(client, cfg) -> None:
    _seed_books_root(cfg, "bk")
    _seed_render(cfg, "bk", engine="kokoro", voice_id="af_heart", n=1)
    o = cfg.output_dir / "bk"
    (o / "manifest.single.json").write_text(
        (o / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (o / "manifest.multi.json").write_text('{"book_id": "bk", "chapters": []}', encoding="utf-8")
    resp = client.delete("/api/books/bk")
    assert resp.status_code == 200, resp.json()
    assert not o.exists()  # archives went with the whole output tree


def test_paid_scan_skips_validation_and_tolerates_torn_sidecar(cfg) -> None:
    from seiyuu.services.deletion import detect_paid_artifacts

    _seed_render(cfg, "bk", engine="elevenlabs", voice_id="v-el", n=2)
    cache = cfg.output_dir / "bk" / "cache"
    # a validation verdict (must be skipped), a torn sidecar (tolerated), and a future
    # .words.json sibling lacking engine/voice_id (ignored) — none inflate the count
    (cache / "deadbeef.validation.json").write_text('{"ok": true}', encoding="utf-8")
    (cache / "torn.json").write_text("{not json", encoding="utf-8")
    (cache / "cafe.words.json").write_text('{"words": []}', encoding="utf-8")

    paid = detect_paid_artifacts(cfg, "bk")
    assert paid.paid_segment_count == 2
    assert paid.paid_voice_ids == ["v-el"]
    assert paid.estimated_usd is None


# -- partial delete + idempotent retry ----------------------------------------------------


def test_partial_delete_500_then_retry_succeeds(client, cfg, monkeypatch) -> None:
    _seed_books_root(cfg, "bk")
    _seed_output_marker(cfg, "bk")
    store: JobStore = client.app.state.store
    done = store.create("bk", "assemble")
    store.mark_running(done.job_id)
    store.finish(done.job_id, JobState.SUCCEEDED)

    real_rmtree = books_mod.shutil.rmtree

    def fake_rmtree(path, onerror=None, **kwargs):
        # simulate a Windows sharing violation: record a survivor, remove nothing
        if onerror is not None:
            onerror(fake_rmtree, str(path), (OSError, OSError("in use"), None))

    monkeypatch.setattr(books_mod.shutil, "rmtree", fake_rmtree)
    failed = client.delete("/api/books/bk")
    assert failed.status_code == 500
    assert _error(failed)["code"] == "partial_delete"
    # jobs rows NOT deleted, trees still present -> retry is idempotent
    assert store.list_jobs(book_id="bk", states=[JobState.SUCCEEDED])
    assert (cfg.books_dir / "bk").is_dir()

    monkeypatch.setattr(books_mod.shutil, "rmtree", real_rmtree)
    ok = client.delete("/api/books/bk")
    assert ok.status_code == 200
    assert ok.json()["jobs_rows_deleted"] == 1
    assert not (cfg.books_dir / "bk").exists()
    assert not (cfg.output_dir / "bk").exists()


def test_partial_output_failure_leaves_books_resolvable(client, cfg, monkeypatch) -> None:
    """The realistic partial failure: output loses its markers but a locked leaf survives.
    delete_book_trees must NOT then remove the books root — otherwise the book resolves via
    neither root and the survivor leaks as an unreachable ghost. Books intact => retry works."""
    _seed_books_root(cfg, "bk")
    _seed_render(cfg, "bk", engine="kokoro", voice_id="af_heart", n=1)  # free: no 402
    odir = cfg.output_dir / "bk"
    bdir = cfg.books_dir / "bk"
    real_rmtree = books_mod.shutil.rmtree
    calls: list[str] = []

    def fake_rmtree(path, onerror=None, **kwargs):
        from pathlib import Path as _P

        calls.append(str(path))
        if _P(path) == odir:  # markers gone, one leaf held open (Windows sharing violation)
            (odir / "manifest.json").unlink(missing_ok=True)
            leaf = odir / "cache" / "locked.wav"
            leaf.parent.mkdir(parents=True, exist_ok=True)
            leaf.write_bytes(b"x")
            if onerror is not None:
                onerror(fake_rmtree, str(leaf), (OSError, OSError("in use"), None))
        else:
            real_rmtree(path, onerror=onerror, **kwargs)

    monkeypatch.setattr(books_mod.shutil, "rmtree", fake_rmtree)
    failed = client.delete("/api/books/bk")
    assert failed.status_code == 500
    assert _error(failed)["code"] == "partial_delete"
    assert bdir.is_dir()  # books root untouched -> the book still resolves
    assert str(bdir) not in calls  # we stopped before attempting the books root

    monkeypatch.setattr(books_mod.shutil, "rmtree", real_rmtree)
    ok = client.delete("/api/books/bk")  # lock cleared -> retry resolves and completes
    assert ok.status_code == 200
    assert not odir.exists()
    assert not bdir.exists()


# -- path safety --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["..", ".", "a/b", "a\\b", "../evil", "a/../b", ""])
def test_delete_book_trees_refuses_traversal(tmp_path, bad) -> None:
    with pytest.raises(RepositoryError):
        delete_book_trees(bad, books_dir=tmp_path / "books", output_dir=tmp_path / "output")


def test_delete_book_trees_missing_roots_not_an_error(tmp_path) -> None:
    result = delete_book_trees(
        "ghost", books_dir=tmp_path / "books", output_dir=tmp_path / "output"
    )
    assert result.output_removed is False
    assert result.books_removed is False
    assert result.survivors == []


# -- CLI parity ---------------------------------------------------------------------------


def _iso_cli(monkeypatch, cfg: Settings) -> None:
    monkeypatch.setattr("seiyuu.settings.get_settings", lambda: cfg)


def test_cli_delete_happy_path(cfg, monkeypatch) -> None:
    _iso_cli(monkeypatch, cfg)
    _seed_books_root(cfg, "bk")
    _seed_output_marker(cfg, "bk")
    result = CliRunner().invoke(main, ["delete", "bk", "--yes"])
    assert result.exit_code == 0, result.output
    assert "deleted bk" in result.output
    assert not (cfg.books_dir / "bk").exists()
    assert not (cfg.output_dir / "bk").exists()


def test_cli_delete_refused_while_job_live(cfg, monkeypatch) -> None:
    _iso_cli(monkeypatch, cfg)
    _seed_books_root(cfg, "bk")
    store = JobStore(cfg.data_dir / JOBS_DB_NAME)
    store.mark_running(store.create("bk", "render").job_id)
    result = CliRunner().invoke(main, ["delete", "bk", "--yes"])
    assert result.exit_code != 0
    assert "render job for 'bk'" in result.output
    assert (cfg.books_dir / "bk").is_dir()


def test_cli_delete_paid_requires_confirm(cfg, monkeypatch) -> None:
    _iso_cli(monkeypatch, cfg)
    _seed_books_root(cfg, "bk")
    _seed_render(cfg, "bk", engine="elevenlabs", voice_id="v-el", n=2)

    refused = CliRunner().invoke(main, ["delete", "bk", "--yes"])
    assert refused.exit_code != 0
    assert "--confirm-paid" in refused.output
    assert (cfg.books_dir / "bk").is_dir()

    ok = CliRunner().invoke(main, ["delete", "bk", "--yes", "--confirm-paid"])
    assert ok.exit_code == 0, ok.output
    assert "paid segments discarded=2" in ok.output
    assert not (cfg.books_dir / "bk").exists()
