"""M6b-5: cost estimate (pure read), quote minting, token-gated render job, render reads.

Fake engines are patched into BOTH construction sites (the API registry and the render
pipeline) so no test touches a real model or a paid API. The paid fake prices by text
length like ElevenLabs so estimates are non-zero and fingerprints are real.
"""

import time

import pytest
from fastapi.testclient import TestClient

from fake_engine import FakeEngine
from seiyuu.api.main import create_app
from seiyuu.repository import Job, JobState, JobStore
from test_api_m6b1 import make_settings
from test_api_m6b3 import _write_attribution, build_synthetic_epub


class PaidFakeEngine(FakeEngine):
    engine_id = "fakepaid"
    uses_gpu = False  # like elevenlabs: cloud, no local weights

    def cost_estimate(self, text: str) -> float:
        return len(text) / 1000 * 0.30


@pytest.fixture(scope="module")
def epub_bytes(tmp_path_factory) -> bytes:
    path = build_synthetic_epub(tmp_path_factory.mktemp("epub") / "synthetic.epub")
    return path.read_bytes()


def _patch_engines(monkeypatch, engine) -> None:
    monkeypatch.setattr("seiyuu.api.registry.get_engine", lambda eid, **kw: engine)
    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda eid, **kw: engine)


@pytest.fixture
def client(tmp_path, epub_bytes, monkeypatch):
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        resp = c.post(
            "/api/books",
            files={"file": ("synthetic.epub", epub_bytes, "application/epub+zip")},
        )
        assert resp.status_code == 201, resp.text
        c.book_id = resp.json()["book"]["book_id"]
        yield c


def _error(resp) -> dict:
    return resp.json()["error"]


def _wait_terminal(store: JobStore, job_id: str, timeout: float = 15.0) -> Job:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job.is_terminal:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} not terminal within {timeout}s: {store.get(job_id)}")


# -- cost estimate ------------------------------------------------------------------------


def test_estimate_single_free(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, FakeEngine())
    body = client.get(
        f"/api/books/{client.book_id}/cost-estimate", params={"mode": "single"}
    ).json()
    assert body["total_usd"] == 0
    assert body["paid_segments"] == 0
    assert body["free_segments"] > 0
    assert body["assignment_hash"] is None
    assert body["edit_warnings"] == []


def test_estimate_single_paid(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, PaidFakeEngine())
    body = client.get(
        f"/api/books/{client.book_id}/cost-estimate",
        params={"mode": "single", "engine": "fakepaid", "voice": "test_voice"},
    ).json()
    assert body["total_usd"] > 0
    assert body["paid_segments"] == body["free_segments"] + body["paid_segments"]
    assert body["fingerprint"]


def test_estimate_multivoice_requires_stages(client) -> None:
    resp = client.get(f"/api/books/{client.book_id}/cost-estimate")
    assert resp.status_code == 404  # pure read: missing attribute/assign -> 404
    assert "attribute" in _error(resp)["message"]


def test_estimate_unknown_engine_422(client) -> None:
    resp = client.get(
        f"/api/books/{client.book_id}/cost-estimate",
        params={"mode": "single", "engine": "fish"},
    )
    assert resp.status_code == 422


# -- quotes -------------------------------------------------------------------------------


def test_quote_free_render_nothing_to_quote(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, FakeEngine())
    resp = client.post(f"/api/books/{client.book_id}/quotes", json={"mode": "single", "single": {}})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "nothing_to_quote"


def test_quote_mint_and_shape(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, PaidFakeEngine())
    resp = client.post(
        f"/api/books/{client.book_id}/quotes",
        json={"mode": "single", "single": {"engine": "fakepaid", "voice": "test_voice"}},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["token"].startswith("cq1.")
    assert body["total_usd"] > 0
    assert body["expires_at"] > body["issued_at"]
    assert body["ttl_seconds"] == 900
    assert body["max_usd_ceiling"] == 25.0


def test_quote_over_ceiling_402(tmp_path, epub_bytes, monkeypatch) -> None:
    settings = make_settings(tmp_path, render_max_usd=0.0001)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        up = c.post(
            "/api/books",
            files={"file": ("synthetic.epub", epub_bytes, "application/epub+zip")},
        )
        book_id = up.json()["book"]["book_id"]
        _patch_engines(monkeypatch, PaidFakeEngine())
        resp = c.post(
            f"/api/books/{book_id}/quotes",
            json={"mode": "single", "single": {"engine": "fakepaid", "voice": "test_voice"}},
        )
    assert resp.status_code == 402
    assert _error(resp)["code"] == "ceiling_exceeded"


def test_quote_single_spec_required_iff(client) -> None:
    assert (
        client.post(f"/api/books/{client.book_id}/quotes", json={"mode": "single"}).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/books/{client.book_id}/quotes",
            json={"mode": "multivoice", "single": {}},
        ).status_code
        == 422
    )


# -- render job: free path end-to-end ------------------------------------------------------


def test_free_single_render_end_to_end(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, FakeEngine())
    resp = client.post(f"/api/books/{client.book_id}/render", json={"mode": "single", "single": {}})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    done = _wait_terminal(client.app.state.store, job_id)
    assert done.state is JobState.SUCCEEDED, done.error

    summary = client.get(f"/api/books/{client.book_id}/render").json()
    assert summary["mode"] == "single"
    assert summary["engine"] == "fake"
    assert summary["total_seconds"] > 0
    assert len(summary["chapters"]) == 3
    assert summary["validation_failures"] == 0
    assert summary["assignment_present"] is False

    validation = client.get(f"/api/books/{client.book_id}/validation").json()
    assert validation == {
        "validated_segments": 0,
        "validation_failures": 0,
        "results": [],
    }  # FakeEngine.requires_validation is False — whisper never ran

    assert summary["chapters"][0]["segments"] > 0
    # play the first rendered block
    from seiyuu.render.models import RenderManifest

    cfg = client.app.state.settings
    manifest = RenderManifest.model_validate_json(
        (cfg.output_dir / client.book_id / "manifest.json").read_text(encoding="utf-8")
    )
    first = next(s for ch in manifest.chapters for s in ch.segments if s.wav)
    audio = client.get(f"/api/books/{client.book_id}/segments/{first.block_id}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert audio.content[:4] == b"RIFF"

    missing = client.get(f"/api/books/{client.book_id}/segments/ch999_b9999/audio")
    assert missing.status_code == 404


def test_subset_renders_accumulate_in_summary(client, monkeypatch) -> None:
    # Regression: the frontend's "continue — next N chapters" preset enqueues subset
    # renders; the second one used to clobber manifest.json so chapter 1 vanished from
    # the summary (and from Listen/assembly) even though its WAVs sat in cache.
    _patch_engines(monkeypatch, FakeEngine())
    for chapters in ([1], [2, 3]):
        resp = client.post(
            f"/api/books/{client.book_id}/render",
            json={"mode": "single", "single": {}, "chapters": chapters},
        )
        assert resp.status_code == 202, resp.text
        done = _wait_terminal(client.app.state.store, resp.json()["job_id"])
        assert done.state is JobState.SUCCEEDED, done.error

    summary = client.get(f"/api/books/{client.book_id}/render").json()
    assert [c["index"] for c in summary["chapters"]] == [1, 2, 3]
    assert all(c["segments"] > 0 for c in summary["chapters"])


def test_render_reads_404_before_render(client) -> None:
    assert client.get(f"/api/books/{client.book_id}/render").status_code == 404
    assert client.get(f"/api/books/{client.book_id}/validation").status_code == 404
    assert client.get(f"/api/books/{client.book_id}/segments/ch001_b0001/audio").status_code == 404


# -- render job: money gate ----------------------------------------------------------------


def test_paid_render_requires_token(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, PaidFakeEngine())
    resp = client.post(
        f"/api/books/{client.book_id}/render",
        json={"mode": "single", "single": {"engine": "fakepaid", "voice": "test_voice"}},
    )
    assert resp.status_code == 402
    err = _error(resp)
    assert err["code"] == "token_required"
    assert err["detail"]["estimated_usd"] > 0


def test_paid_render_full_token_lifecycle(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, PaidFakeEngine())
    token = client.post(
        f"/api/books/{client.book_id}/quotes",
        json={"mode": "single", "single": {"engine": "fakepaid", "voice": "test_voice"}},
    ).json()["token"]

    resp = client.post(
        f"/api/books/{client.book_id}/render",
        json={
            "mode": "single",
            "single": {"engine": "fakepaid", "voice": "test_voice"},
            "cost_token": token,
        },
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    redacted = resp.json()["params"]["cost_token"]
    assert redacted["present"] is True and "sig_suffix" in redacted
    assert "cq1." not in resp.text  # the raw token never crosses HTTP back out

    done = _wait_terminal(client.app.state.store, job_id)
    assert done.state is JobState.SUCCEEDED, done.error

    # single-use: the handler consumed it at job start; replay refuses at the dry-run
    replay = client.post(
        f"/api/books/{client.book_id}/render",
        json={
            "mode": "single",
            "single": {"engine": "fakepaid", "voice": "test_voice"},
            "cost_token": token,
        },
    )
    # everything is cached now, so the re-render is FREE and the stale token is ignored
    assert replay.status_code == 202
    done2 = _wait_terminal(client.app.state.store, replay.json()["job_id"])
    assert done2.state is JobState.SUCCEEDED


def test_used_token_refuses_when_still_paid(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, PaidFakeEngine())
    quote = client.post(
        f"/api/books/{client.book_id}/quotes",
        json={"mode": "single", "single": {"engine": "fakepaid", "voice": "test_voice"}},
    ).json()
    token = quote["token"]

    # consume it directly through the gate (as a finished render job would have)
    from seiyuu.render.gate import CostQuote, verify_quote

    cfg = client.app.state.settings
    decoded = CostQuote.decode(token)
    verify_quote(
        decoded,
        book_id=client.book_id,
        chapters=(),
        fingerprint=decoded.fingerprint,
        assignment_hash=None,
        recomputed_total_usd=decoded.total_usd,
        max_usd=cfg.render_max_usd,
        data_dir=cfg.data_dir,
        consume=True,
    )
    resp = client.post(
        f"/api/books/{client.book_id}/render",
        json={
            "mode": "single",
            "single": {"engine": "fakepaid", "voice": "test_voice"},
            "cost_token": token,
        },
    )
    assert resp.status_code == 402
    assert _error(resp)["code"] == "quote_used"  # dry-run catches it, token state unchanged


def test_token_scope_mismatch_402(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, PaidFakeEngine())
    token = client.post(
        f"/api/books/{client.book_id}/quotes",
        json={
            "mode": "single",
            "single": {"engine": "fakepaid", "voice": "test_voice"},
            "chapters": [1],
        },
    ).json()["token"]
    resp = client.post(
        f"/api/books/{client.book_id}/render",
        json={
            "mode": "single",
            "single": {"engine": "fakepaid", "voice": "test_voice"},
            "chapters": [1, 2],
            "cost_token": token,
        },
    )
    assert resp.status_code == 402
    assert _error(resp)["code"] == "quote_mismatch"


def test_malformed_token_422(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, PaidFakeEngine())
    resp = client.post(
        f"/api/books/{client.book_id}/render",
        json={
            "mode": "single",
            "single": {"engine": "fakepaid", "voice": "test_voice"},
            "cost_token": "garbage",
        },
    )
    assert resp.status_code == 422
    assert _error(resp)["code"] == "invalid"


# -- render job: conflicts + confirm-full ---------------------------------------------------


def test_render_conflicts_with_live_attribute_job(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, FakeEngine())
    store: JobStore = client.app.state.store
    attribute = store.create(client.book_id, "attribute")
    resp = client.post(f"/api/books/{client.book_id}/render", json={"mode": "single", "single": {}})
    assert resp.status_code == 409
    err = _error(resp)
    assert err["code"] == "conflicting_job"
    assert err["detail"]["job_id"] == attribute.job_id
    store.request_cancel(attribute.job_id)


def test_full_render_confirmation(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, FakeEngine())
    monkeypatch.setattr("seiyuu.api.routes.render.FULL_RENDER_CONFIRM_BLOCKS", 1)
    resp = client.post(f"/api/books/{client.book_id}/render", json={"mode": "single", "single": {}})
    assert resp.status_code == 409
    err = _error(resp)
    assert err["code"] == "full_render_confirmation_required"
    assert err["detail"]["speakable_blocks"] > 1
    assert err["detail"]["runtime_estimate_seconds"] > 0

    confirmed = client.post(
        f"/api/books/{client.book_id}/render",
        json={"mode": "single", "single": {}, "confirm_full": True},
    )
    assert confirmed.status_code == 202
    _wait_terminal(client.app.state.store, confirmed.json()["job_id"])

    # an explicit chapter subset never needs the confirm
    subset = client.post(
        f"/api/books/{client.book_id}/render",
        json={"mode": "single", "single": {}, "chapters": [1]},
    )
    assert subset.status_code == 202
    _wait_terminal(client.app.state.store, subset.json()["job_id"])


def test_render_multivoice_missing_stage_is_409(client) -> None:
    resp = client.post(f"/api/books/{client.book_id}/render", json={})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "stage_prerequisite"


# -- multivoice end-to-end with the edits overlay ------------------------------------------


def test_multivoice_render_end_to_end(client, monkeypatch) -> None:
    _patch_engines(monkeypatch, FakeEngine())
    cfg = client.app.state.settings
    book_id = client.book_id
    # attribution fixture aligned to the real ingested block ids
    report = _write_attribution(cfg, book_id)
    from seiyuu.ingest.models import NormalizedBook

    book = NormalizedBook.model_validate_json(
        (cfg.books_dir / book_id / "normalized.json").read_text(encoding="utf-8")
    )
    real_blocks = [b.id for c in book.chapters for b in c.blocks if b.is_speakable]
    fixed = report.model_dump()
    for chapter in fixed["chapters"]:
        for i, seg in enumerate(chapter["segments"]):
            seg["block_id"] = real_blocks[i % len(real_blocks)]
    fixed["chapters"] = [c for c in fixed["chapters"] if c["index"] == 1]
    from seiyuu.attribute.models import AttributionReport

    (cfg.books_dir / book_id / "attribution.json").write_text(
        AttributionReport.model_validate(fixed).model_dump_json(), encoding="utf-8"
    )
    draft = client.post(f"/api/books/{book_id}/assignment/draft", json={})
    assert draft.status_code == 201, draft.text

    resp = client.post(f"/api/books/{book_id}/render", json={"chapters": [1]})
    assert resp.status_code == 202, resp.text
    done = _wait_terminal(client.app.state.store, resp.json()["job_id"])
    assert done.state is JobState.SUCCEEDED, done.error

    summary = client.get(f"/api/books/{book_id}/render").json()
    assert summary["mode"] == "multivoice"
    assert summary["assignment_present"] is True
    assert summary["voices_used"]


# -- review-workflow regression fixes (M6b-5 findings) --------------------------------------


def test_gate_code_immune_to_book_id_shadowing() -> None:
    # Finding: single-word needles were matched by user-controlled book ids embedded in
    # the mismatch message, steering M6c toward the silent re-mint path.
    from seiyuu.api.money import gate_code
    from seiyuu.render.gate import CostGateError

    mismatch = CostGateError(
        "cost token was issued for book 'the-expired-heart-1a2b', not 'glass-ceiling-9f'"
    )
    assert gate_code(mismatch) == "quote_mismatch"
    assert gate_code(CostGateError("cost token expired; re-run estimate-cost")) == "quote_expired"
    assert (
        gate_code(
            CostGateError(
                "estimated $9.00 exceeds the render_max_usd ceiling ($5.00); set RENDER_MAX_USD"
            )
        )
        == "ceiling_exceeded"
    )
    assert (
        gate_code(CostGateError("cost token already used (tokens are single-use); re-run"))
        == "quote_used"
    )


def test_single_nonkokoro_requires_explicit_voice(client, monkeypatch) -> None:
    # Finding: the kokoro default voice leaked into other engines, minting quotes (and
    # later burning tokens) for renders that could never synthesize.
    _patch_engines(monkeypatch, PaidFakeEngine())
    est = client.get(
        f"/api/books/{client.book_id}/cost-estimate",
        params={"mode": "single", "engine": "fakepaid"},
    )
    assert est.status_code == 422
    assert "requires an explicit voice" in _error(est)["message"]
    quote = client.post(
        f"/api/books/{client.book_id}/quotes",
        json={"mode": "single", "single": {"engine": "fakepaid"}},
    )
    assert quote.status_code == 422
    # kokoro still defaults fine
    ok = client.get(f"/api/books/{client.book_id}/cost-estimate", params={"mode": "single"})
    assert ok.status_code == 200


def test_estimate_multivoice_rejects_single_params(client) -> None:
    # Finding: the GET silently discarded single-voice params in multivoice mode while
    # the POST bodies 422 the same combination — the money dialog could show an
    # estimate for a different scope than the client asked about.
    resp = client.get(f"/api/books/{client.book_id}/cost-estimate", params={"engine": "fakepaid"})
    assert resp.status_code == 422


def test_handler_burn_survives_cache_wipe_replay(client, monkeypatch) -> None:
    # Finding: nothing pinned the handler's verify_quote(consume=True). Wipe the segment
    # cache after a paid render (same fingerprint, paid again, within TTL) and replay the
    # SAME token: only the handler's burn makes this a 402 quote_used.
    import shutil

    _patch_engines(monkeypatch, PaidFakeEngine())
    spec = {"engine": "fakepaid", "voice": "test_voice"}
    token = client.post(
        f"/api/books/{client.book_id}/quotes", json={"mode": "single", "single": spec}
    ).json()["token"]
    first = client.post(
        f"/api/books/{client.book_id}/render",
        json={"mode": "single", "single": spec, "cost_token": token},
    )
    assert first.status_code == 202, first.text
    done = _wait_terminal(client.app.state.store, first.json()["job_id"])
    assert done.state is JobState.SUCCEEDED, done.error

    cfg = client.app.state.settings
    shutil.rmtree(cfg.output_dir / client.book_id / "cache")  # eviction/cleanup/restore
    replay = client.post(
        f"/api/books/{client.book_id}/render",
        json={"mode": "single", "single": spec, "cost_token": token},
    )
    assert replay.status_code == 402
    assert _error(replay)["code"] == "quote_used"  # one approval can never bill twice


def test_paid_elevenlabs_without_key_refuses_before_token(client, monkeypatch) -> None:
    # Finding: a keyless elevenlabs render was quotable and its token burned on a job
    # that deterministically fails at the first synthesis.
    _patch_engines(monkeypatch, PaidFakeEngine())
    body = {"mode": "single", "single": {"engine": "elevenlabs", "voice": "stock-id"}}
    quote = client.post(f"/api/books/{client.book_id}/quotes", json=body)
    assert quote.status_code == 503
    assert "ELEVENLABS_API_KEY" in _error(quote)["message"]
    render = client.post(f"/api/books/{client.book_id}/render", json=body)
    assert render.status_code == 503


def test_consent_preflight_refuses_before_token(client, monkeypatch) -> None:
    # Finding: a stale/missing clone consent surfaced only AFTER the token was consumed.
    import json as jsonlib

    _patch_engines(monkeypatch, FakeEngine())
    cfg = client.app.state.settings
    voice_dir = cfg.voices_dir / "badclone"
    voice_dir.mkdir(parents=True)
    (voice_dir / "meta.json").write_text(
        jsonlib.dumps(
            {
                "voice_id": "badclone",
                "name": "Bad Clone",
                "kind": "cloned",
                "engine": "chatterbox",
                "reference_audio": "reference.wav",
                "consent_attested": False,
            }
        ),
        encoding="utf-8",
    )
    resp = client.post(
        f"/api/books/{client.book_id}/render",
        json={"mode": "single", "single": {"engine": "chatterbox", "voice": "badclone"}},
    )
    assert resp.status_code == 409
    assert _error(resp)["code"] == "consent_invalid"


def test_segment_audio_addresses_multivoice_segments(client) -> None:
    # Finding: only a block's FIRST wav was reachable — a failing dialogue span behind a
    # passing narration span could never be auditioned.
    from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest

    cfg = client.app.state.settings
    odir = cfg.output_dir / client.book_id
    (odir / "cache").mkdir(parents=True, exist_ok=True)
    (odir / "cache" / "a.wav").write_bytes(b"RIFFnarration")
    (odir / "cache" / "b.wav").write_bytes(b"RIFFdialogue")
    manifest = RenderManifest(
        book_id=client.book_id,
        chapters=[
            RenderedChapter(
                index=1,
                title="Ch",
                segments=[
                    RenderedSegment(
                        block_id="ch001_b0001",
                        type="paragraph",
                        wav="cache/a.wav",
                        duration_seconds=1.0,
                        validation={"ok": True, "score": 0.99, "transcript": "x", "expected": "x"},
                    ),
                    RenderedSegment(
                        block_id="ch001_b0001",
                        type="paragraph",
                        wav="cache/b.wav",
                        duration_seconds=1.0,
                        validation={"ok": False, "score": 0.31, "transcript": "y", "expected": "z"},
                    ),
                ],
            )
        ],
        validation_failures=1,
    )
    (odir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    rows = client.get(f"/api/books/{client.book_id}/validation").json()["results"]
    assert len(rows) == 1
    assert rows[0]["segment_index"] == 1  # the FAILING span, not the block

    first = client.get(f"/api/books/{client.book_id}/segments/ch001_b0001/audio")
    assert first.content == b"RIFFnarration"
    failing = client.get(
        f"/api/books/{client.book_id}/segments/ch001_b0001/audio", params={"segment": 1}
    )
    assert failing.content == b"RIFFdialogue"  # the validation row's index plays ITS audio
    out_of_range = client.get(
        f"/api/books/{client.book_id}/segments/ch001_b0001/audio", params={"segment": 2}
    )
    assert out_of_range.status_code == 404
