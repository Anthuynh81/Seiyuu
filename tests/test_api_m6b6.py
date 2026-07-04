"""M6b-6: voices API — CRUD, clone (purge-on-reclone), audition refusals, slots, cover."""

import json

import pytest
from fastapi.testclient import TestClient

from fake_engine import FakeEngine
from seiyuu.api.main import create_app
from seiyuu.repository import JobStore
from test_api_m6b1 import make_settings
from test_api_m6b3 import _write_attribution


class PaidCloudFake(FakeEngine):
    engine_id = "elevenlabs-fake"
    uses_gpu = False

    def cost_estimate(self, text: str) -> float:
        return len(text) / 1000 * 0.30


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        yield c


@pytest.fixture
def keyed_client(tmp_path):
    """Paid-audition tests: the keyless-503 preflight correctly fires before the
    confirm_paid 402, so exercising the money prompts needs a configured key."""
    settings = make_settings(tmp_path, elevenlabs_api_key="k-test-not-real")
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        yield c


def _error(resp) -> dict:
    return resp.json()["error"]


def _patch_audition(monkeypatch, engine=None, cls_by_id=None) -> None:
    engine = engine or FakeEngine()
    monkeypatch.setattr("seiyuu.api.registry.get_engine", lambda eid, **kw: engine)
    classes = cls_by_id or {}
    monkeypatch.setattr(
        "seiyuu.api.routes.voices.get_engine_class",
        lambda eid: classes.get(eid, FakeEngine),
    )
    # pin the cold-engine probe: tests must not depend on this machine's HF cache
    monkeypatch.setattr("seiyuu.api.routes.voices.weights_cached", lambda eid: True)


def _make_preset(client, voice_id="v1", engine="kokoro", preset_id="test_voice") -> dict:
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
    return resp.json()


def _clone(client, voice_id="clone1", data=b"RIFF-fake-reference", **form):
    payload = {
        "name": "My Clone",
        "consent": "true",
        "attested_by": "cyberfang",
        "voice_id": voice_id,
        **form,
    }
    return client.post(
        "/api/voices/clone",
        files={"file": ("ref.wav", data, "audio/wav")},
        data=payload,
    )


# -- library CRUD -------------------------------------------------------------------------


def test_list_voices_tolerant_scan(client) -> None:
    assert client.get("/api/voices").json() == {"voices": [], "unreadable": []}
    _make_preset(client)
    broken = client.app.state.settings.voices_dir / "broken"
    broken.mkdir(parents=True)
    (broken / "meta.json").write_text("{nope", encoding="utf-8")

    body = client.get("/api/voices").json()
    assert [v["voice_id"] for v in body["voices"]] == ["v1"]
    assert body["voices"][0]["has_audition"] is False
    assert [u["voice_id"] for u in body["unreadable"]] == ["broken"]


def test_create_preset_blend_and_duplicate(client) -> None:
    created = _make_preset(client)
    assert created["kind"] == "preset"
    assert created["seed"] == 41172

    dup = client.post(
        "/api/voices",
        json={"kind": "preset", "name": "X", "preset_id": "af_heart", "voice_id": "v1"},
    )
    assert dup.status_code == 409
    assert _error(dup)["code"] == "voice_exists"

    auto = client.post(
        "/api/voices",
        json={"kind": "blend", "name": "Mara", "gender": "female", "voice_id": "mara"},
    )
    assert auto.status_code == 201, auto.text
    assert auto.json()["kind"] == "blend"
    assert len(auto.json()["blend"]) >= 2
    assert auto.json()["engine"] == "kokoro"

    manual = client.post(
        "/api/voices",
        json={
            "kind": "blend",
            "name": "Duo",
            "voice_id": "duo",
            "components": [
                {"preset_id": "af_heart", "weight": 2},
                {"preset_id": "af_bella", "weight": 1},
            ],
        },
    )
    assert manual.status_code == 201, manual.text

    short = client.post(
        "/api/voices",
        json={
            "kind": "blend",
            "name": "Solo",
            "components": [{"preset_id": "af_heart", "weight": 1}],
        },
    )
    assert short.status_code == 422  # >= 2 components


# -- clone --------------------------------------------------------------------------------


def test_clone_records_hash_bound_consent(client) -> None:
    resp = _clone(client)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["kind"] == "cloned"
    assert body["consent_attested"] is True
    assert body["consent"]["attested_by"] == "cyberfang"

    from seiyuu.voices.library import sha256_file

    cfg = client.app.state.settings
    ref = cfg.voices_dir / "clone1" / "reference.wav"
    assert ref.read_bytes() == b"RIFF-fake-reference"
    assert body["consent"]["reference_sha256"] == sha256_file(ref)  # binds to THE bytes
    assert not (cfg.voices_dir / "clone1" / "reference.wav.part").exists()


def test_clone_requires_consent_and_attestor(client) -> None:
    no_consent = _clone(client, consent="false")
    assert no_consent.status_code == 422
    assert "consent" in _error(no_consent)["message"]
    no_by = _clone(client, attested_by="  ")
    assert no_by.status_code == 422
    assert "attested_by" in _error(no_by)["message"]


def test_reclone_blocked_then_replace_purges(client) -> None:
    assert _clone(client).status_code == 201
    cfg = client.app.state.settings

    blocked = _clone(client, data=b"RIFF-new-take")
    assert blocked.status_code == 409
    assert _error(blocked)["code"] == "reclone_blocked"

    # seed cached segments for clone1 and an unrelated voice, plus an audition clip
    cache = cfg.output_dir / "bk" / "cache"
    cache.mkdir(parents=True)
    (cache / "aaa.json").write_text(json.dumps({"voice_id": "clone1"}), encoding="utf-8")
    (cache / "aaa.wav").write_bytes(b"old-audio")
    (cache / "aaa.validation.json").write_text("{}", encoding="utf-8")
    (cache / "bbb.json").write_text(json.dumps({"voice_id": "other"}), encoding="utf-8")
    (cache / "bbb.wav").write_bytes(b"other-audio")
    (cfg.voices_dir / "clone1" / "audition.wav").write_bytes(b"old-audition")
    (cfg.voices_dir / "clone1" / "conds_v1_deadbeef.pt").write_bytes(b"conds")

    replaced = _clone(client, data=b"RIFF-new-take", replace="true")
    assert replaced.status_code == 201, replaced.text
    assert (cfg.voices_dir / "clone1" / "reference.wav").read_bytes() == b"RIFF-new-take"

    assert not (cache / "aaa.wav").exists()  # Q1: stale audio can never replay
    assert not (cache / "aaa.json").exists()
    assert not (cache / "aaa.validation.json").exists()
    assert (cache / "bbb.wav").exists()  # other voices untouched
    assert not (cfg.voices_dir / "clone1" / "audition.wav").exists()
    assert not (cfg.voices_dir / "clone1" / "conds_v1_deadbeef.pt").exists()


# -- detail / references / delete -----------------------------------------------------------


def test_detail_references_delete_lifecycle(client) -> None:
    assert client.get("/api/voices/ghost").status_code == 404
    assert client.get("/api/voices/a:b").status_code == 422  # traversal-shaped id

    _make_preset(client)
    detail = client.get("/api/voices/v1").json()
    assert detail["audition_url"] is None

    refs = client.get("/api/voices/v1/references").json()
    assert refs == {"voice_id": "v1", "references": []}

    # referenced -> refuse deletion, list the roles
    cfg = client.app.state.settings
    odir = cfg.output_dir / "bk"
    odir.mkdir(parents=True)
    (odir / "assignments.json").write_text(
        json.dumps({"book_id": "bk", "narrator_voice_id": "v1", "assignments": {"alice": "v1"}}),
        encoding="utf-8",
    )
    refs = client.get("/api/voices/v1/references").json()["references"]
    assert {r["role"] for r in refs} == {"narrator", "character:alice"}
    refused = client.delete("/api/voices/v1")
    assert refused.status_code == 409
    assert _error(refused)["code"] == "voice_referenced"

    (odir / "assignments.json").unlink()
    gone = client.delete("/api/voices/v1")
    assert gone.status_code == 200
    assert gone.json() == {"deleted": "v1"}
    assert not (cfg.voices_dir / "v1").exists()
    assert client.delete("/api/voices/v1").status_code == 404


def test_unreadable_assignment_fails_closed(client) -> None:
    _make_preset(client)
    odir = client.app.state.settings.output_dir / "bk"
    odir.mkdir(parents=True)
    (odir / "assignments.json").write_text("{broken", encoding="utf-8")
    assert client.get("/api/voices/v1/references").status_code == 500
    refused = client.delete("/api/voices/v1")
    assert refused.status_code == 500
    assert _error(refused)["code"] == "corrupt_artifact"


# -- audition -----------------------------------------------------------------------------


def test_audition_happy_path_free(client, monkeypatch) -> None:
    engine = FakeEngine()
    _patch_audition(monkeypatch, engine)
    _make_preset(client)

    resp = client.post("/api/voices/v1/audition", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cost_usd"] == 0.0
    assert body["duration_seconds"] > 0
    assert engine.calls  # the fake actually synthesized

    wav = client.get("/api/voices/v1/audition.wav")
    assert wav.status_code == 200
    assert wav.headers["content-type"].startswith("audio/wav")
    assert client.get("/api/voices/v1").json()["audition_url"] == "/api/voices/v1/audition.wav"

    from seiyuu.gpu import get_gpu_manager

    assert get_gpu_manager().holds(engine)  # stays lazily resident — no free_all
    assert client.app.state.registry.is_resident("kokoro") is True


def test_audition_wav_404_before_first_audition(client) -> None:
    _make_preset(client)
    assert client.get("/api/voices/v1/audition.wav").status_code == 404


def test_audition_refused_while_gpu_job_live(client, monkeypatch) -> None:
    _patch_audition(monkeypatch)
    _make_preset(client)
    store: JobStore = client.app.state.store
    job = store.create("bk", "attribute")

    resp = client.post("/api/voices/v1/audition", json={})
    assert resp.status_code == 409
    err = _error(resp)
    assert err["code"] == "gpu_busy"
    assert err["detail"]["job_id"] == job.job_id

    store.request_cancel(job.job_id)
    assert client.post("/api/voices/v1/audition", json={}).status_code == 200

    # assemble/master jobs hold no GPU: auditions stay allowed
    other = store.create("bk", "assemble")
    assert client.post("/api/voices/v1/audition", json={}).status_code == 200
    store.request_cancel(other.job_id)


def test_cloud_audition_refused_while_render_live(keyed_client, monkeypatch) -> None:
    _patch_audition(monkeypatch, PaidCloudFake(), cls_by_id={"elevenlabs": PaidCloudFake})
    _make_preset(keyed_client, voice_id="cloudv", engine="elevenlabs", preset_id="stock1")
    store: JobStore = keyed_client.app.state.store
    render = store.create("bk", "render")

    resp = keyed_client.post("/api/voices/cloudv/audition", json={"confirm_paid": True})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "cloud_busy"  # eviction-race closure (Q6)

    store.request_cancel(render.job_id)
    # an attribute job does NOT block a cloud audition (no GPU, no slots)
    attr = store.create("bk", "attribute")
    ok = keyed_client.post("/api/voices/cloudv/audition", json={"confirm_paid": True})
    assert ok.status_code == 200, ok.text
    assert ok.json()["cost_usd"] > 0
    store.request_cancel(attr.job_id)


def test_paid_audition_requires_confirmation(keyed_client, monkeypatch) -> None:
    _patch_audition(monkeypatch, PaidCloudFake(), cls_by_id={"elevenlabs": PaidCloudFake})
    _make_preset(keyed_client, voice_id="cloudv", engine="elevenlabs", preset_id="stock1")
    resp = keyed_client.post("/api/voices/cloudv/audition", json={})
    assert resp.status_code == 402
    err = _error(resp)
    assert err["code"] == "payment_confirmation_required"
    assert err["detail"]["estimated_usd"] > 0


def test_audition_in_flight_refusal(client, monkeypatch) -> None:
    _patch_audition(monkeypatch)
    _make_preset(client)
    with client.app.state.audition_slot.try_hold() as held:
        assert held
        resp = client.post("/api/voices/v1/audition", json={})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "audition_in_flight"


def test_cold_engine_refused_toward_warmup(client, monkeypatch) -> None:
    _patch_audition(monkeypatch)
    monkeypatch.setattr("seiyuu.api.routes.voices.weights_cached", lambda eid: False)
    _make_preset(client)
    resp = client.post("/api/voices/v1/audition", json={})
    assert resp.status_code == 409
    err = _error(resp)
    assert err["code"] == "engine_cold"
    assert err["detail"]["warmup"] == "/api/engines/kokoro/warmup"


def test_audition_consent_invalid(client, monkeypatch) -> None:
    _patch_audition(monkeypatch)
    cfg = client.app.state.settings
    voice_dir = cfg.voices_dir / "badclone"
    voice_dir.mkdir(parents=True)
    (voice_dir / "meta.json").write_text(
        json.dumps(
            {
                "voice_id": "badclone",
                "name": "Bad",
                "kind": "cloned",
                "engine": "chatterbox",
                "reference_audio": "reference.wav",
                "consent_attested": False,
            }
        ),
        encoding="utf-8",
    )
    resp = client.post("/api/voices/badclone/audition", json={})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "consent_invalid"


def test_audition_text_length_capped(client, monkeypatch) -> None:
    _patch_audition(monkeypatch)
    _make_preset(client)
    resp = client.post("/api/voices/v1/audition", json={"text": "x" * 501})
    assert resp.status_code == 422


# -- cloud slots --------------------------------------------------------------------------


def test_cloud_slots_read_only_view(client) -> None:
    empty = client.get("/api/cloud-slots").json()
    assert empty == {"max_slots": 10, "count": 0, "slots": []}

    cfg = client.app.state.settings
    cfg.voices_dir.mkdir(parents=True, exist_ok=True)
    (cfg.voices_dir / "cloud_voices.json").write_text(
        json.dumps(
            {
                "next_seq": 3,
                "voices": {
                    "v1": {"cloud_id": "c1", "seq": 1},
                    "v2": {"cloud_id": "c2", "seq": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    body = client.get("/api/cloud-slots").json()
    assert body["count"] == 2
    assert [s["voice_id"] for s in body["slots"]] == ["v2", "v1"]  # MRU-first

    (cfg.voices_dir / "cloud_voices.json").write_text("{broken", encoding="utf-8")
    assert client.get("/api/cloud-slots").status_code == 500


# -- cover art ----------------------------------------------------------------------------


def test_cover_upload_replace_delete(client) -> None:
    _write_attribution(client.app.state.settings, "bk")  # the book must exist

    assert client.get("/api/books/bk/cover").status_code == 404  # nothing uploaded yet
    png = client.put(
        "/api/books/bk/cover",
        files={"file": ("c.png", b"\x89PNG\r\n\x1a\nrest", "image/png")},
    )
    assert png.status_code == 200, png.text
    assert png.json()["content_type"] == "image/png"
    served = client.get("/api/books/bk/cover")
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("image/png")
    assert served.content.startswith(b"\x89PNG")
    cfg = client.app.state.settings
    assert (cfg.output_dir / "bk" / "cover.png").is_file()
    assert client.get("/api/books/bk").json()["cover"]["content_type"] == "image/png"

    jpg = client.put(
        "/api/books/bk/cover",
        files={"file": ("c.jpg", b"\xff\xd8\xff\xe0rest", "image/jpeg")},
    )
    assert jpg.status_code == 200
    assert (cfg.output_dir / "bk" / "cover.jpg").is_file()
    assert not (cfg.output_dir / "bk" / "cover.png").exists()  # never two covers

    wrong_type = client.put(
        "/api/books/bk/cover", files={"file": ("c.txt", b"hello", "text/plain")}
    )
    assert wrong_type.status_code == 415
    forged = client.put(
        "/api/books/bk/cover", files={"file": ("c.png", b"\xff\xd8\xffjpeg", "image/png")}
    )
    assert forged.status_code == 415  # magic bytes must match the claimed type

    assert client.delete("/api/books/bk/cover").status_code == 204
    assert not (cfg.output_dir / "bk" / "cover.jpg").exists()
    assert client.delete("/api/books/bk/cover").status_code == 204  # idempotent
    assert (
        client.put(
            "/api/books/ghost/cover", files={"file": ("c.png", b"\x89PNG", "image/png")}
        ).status_code
        == 404
    )


# -- cumulative-review regression fixes ------------------------------------------------------


def test_route_surface_is_complete() -> None:
    # Finding (x5): assemble/master routes were missing — the API dead-ended at
    # rendered=true. Pin the full documented surface.
    from fastapi.routing import APIRoute

    from seiyuu.api.main import create_app as make

    app = make()
    paths = {
        (r.path, m) for r in app.routes if isinstance(r, APIRoute) for m in r.methods if m != "HEAD"
    }
    assert ("/api/books/{book_id}/assemble", "POST") in paths
    assert ("/api/books/{book_id}/master", "POST") in paths
    assert ("/api/books/{book_id}/cover", "GET") in paths
    # the scoping doc's 44 rows + GET cover (M6c-5b: the shelf shows books by cover art)
    assert len(paths) == 45


def test_assemble_and_master_routes(client, monkeypatch) -> None:
    import time as time_mod

    seen: dict = {}
    monkeypatch.setattr(
        "seiyuu.assemble.assemble_book", lambda d, **kw: seen.setdefault("assemble", kw)
    )
    monkeypatch.setattr(
        "seiyuu.assemble.master_book", lambda d, **kw: seen.setdefault("master", kw)
    )
    _write_attribution(client.app.state.settings, "bk")

    premature = client.post("/api/books/bk/assemble", json={})
    assert premature.status_code == 409
    assert _error(premature)["code"] == "stage_prerequisite"

    odir = client.app.state.settings.output_dir / "bk"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "manifest.json").write_text("{}", encoding="utf-8")  # rendered marker

    resp = client.post("/api/books/bk/assemble", json={"pauses": {"paragraph": 0.0}})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    assert resp.headers["location"] == f"/api/jobs/{job_id}"

    master = client.post("/api/books/bk/master", json={"bitrate": "96k"})
    assert master.status_code == 202, master.text

    store: JobStore = client.app.state.store
    deadline = time_mod.monotonic() + 5.0
    while time_mod.monotonic() < deadline:
        if store.get(job_id).is_terminal and store.get(master.json()["job_id"]).is_terminal:
            break
        time_mod.sleep(0.02)
    assert store.get(job_id).state.value == "succeeded", store.get(job_id).error
    assert seen["assemble"]["pauses"].paragraph == 0.0  # explicit zero honored end-to-end
    assert seen["master"]["bitrate"] == "96k"


def test_replace_clone_and_delete_refused_while_render_live(client) -> None:
    # Finding (x4): the purge deletes cache files a RUNNING render references; a
    # single-voice render's voice never appears in assignments.json.
    assert _clone(client).status_code == 201
    store: JobStore = client.app.state.store
    render = store.create("some-book", "render")

    blocked = _clone(client, data=b"RIFF-new", replace="true")
    assert blocked.status_code == 409
    assert _error(blocked)["code"] == "render_active"
    assert _error(blocked)["detail"]["job_id"] == render.job_id

    deleted = client.delete("/api/voices/clone1")
    assert deleted.status_code == 409
    assert _error(deleted)["code"] == "render_active"

    store.request_cancel(render.job_id)
    assert _clone(client, data=b"RIFF-new", replace="true").status_code == 201
    assert client.delete("/api/voices/clone1").status_code == 200


def test_reclone_and_delete_drop_cloud_handle(client) -> None:
    # Finding (x3, worst Q1 violation): the IVC handle trained on the OLD reference
    # stayed in cloud_voices.json — paid synthesis kept the previously-attested speaker.
    assert _clone(client).status_code == 201
    cfg = client.app.state.settings
    (cfg.voices_dir / "cloud_voices.json").write_text(
        json.dumps(
            {
                "next_seq": 2,
                "voices": {
                    "clone1": {"cloud_id": "old-ivc", "seq": 0},
                    "other": {"cloud_id": "keep", "seq": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    assert _clone(client, data=b"RIFF-new-speaker", replace="true").status_code == 201
    slots = {s["voice_id"] for s in client.get("/api/cloud-slots").json()["slots"]}
    assert slots == {"other"}  # the stale handle is gone, the unrelated one remains

    (cfg.voices_dir / "cloud_voices.json").write_text(
        json.dumps({"next_seq": 3, "voices": {"clone1": {"cloud_id": "new-ivc", "seq": 2}}}),
        encoding="utf-8",
    )
    assert client.delete("/api/voices/clone1").status_code == 200
    assert client.get("/api/cloud-slots").json()["count"] == 0  # slot entry freed


def test_cloud_audition_allowed_while_gate_held_by_job(keyed_client, monkeypatch) -> None:
    # Finding (x5, one repro'd): any gate-holding job produced a phantom 409
    # audition_in_flight. Cloud auditions (no GPU) must not touch the gate at all...
    _patch_audition(monkeypatch, PaidCloudFake(), cls_by_id={"elevenlabs": PaidCloudFake})
    _make_preset(keyed_client, voice_id="cloudv", engine="elevenlabs", preset_id="stock1")
    with keyed_client.app.state.gate.hold("job"):  # e.g. a running attribute handler
        resp = keyed_client.post("/api/voices/cloudv/audition", json={"confirm_paid": True})
    assert resp.status_code == 200, resp.text


def test_gpu_audition_gets_gpu_busy_not_phantom_audition(client, monkeypatch) -> None:
    # ...and a GPU audition losing the gate reports the TRUE holder (a job), never
    # "another audition is already synthesizing".
    _patch_audition(monkeypatch)
    _make_preset(client)
    with client.app.state.gate.hold("job"):  # e.g. a running assemble/master handler
        resp = client.post("/api/voices/v1/audition", json={})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "gpu_busy"
    assert client.get("/api/system").json()["audition_in_flight"] is False  # consistent


def test_create_voice_unknown_engine_422_and_location(client) -> None:
    bad = client.post(
        "/api/voices",
        json={"kind": "preset", "name": "X", "engine": "Kokoro", "preset_id": "af_heart"},
    )
    assert bad.status_code == 422  # was a 201 that 500'd on every later use
    assert "unknown engine" in _error(bad)["message"]

    ok = client.post(
        "/api/voices",
        json={"kind": "preset", "name": "X", "preset_id": "af_heart", "voice_id": "vx"},
    )
    assert ok.headers["location"] == "/api/voices/vx"
    clone = _clone(client, voice_id="cl2")
    assert clone.headers["location"] == "/api/voices/cl2"


def test_keyless_paid_audition_is_503_not_502(client, monkeypatch) -> None:
    _patch_audition(monkeypatch, PaidCloudFake(), cls_by_id={"elevenlabs": PaidCloudFake})
    _make_preset(client, voice_id="cloudv", engine="elevenlabs", preset_id="stock1")
    resp = client.post("/api/voices/cloudv/audition", json={"confirm_paid": True})
    assert resp.status_code == 503  # config fault, not an upstream failure
    assert _error(resp)["code"] == "not_ready"
    assert "ELEVENLABS_API_KEY" in _error(resp)["message"]


def test_edit_conflict_with_corrupt_looking_id_stays_409(client) -> None:
    # Finding: the bare "corrupt" sniff — a character id like "corrupt-one" in an
    # anchor-conflict message crossed into 500 corrupt_artifact.
    _write_attribution(client.app.state.settings, "bk")
    resp = client.post(
        "/api/books/bk/edits",
        json={
            "op": "reassign",
            "block_id": "ch001_b0001",
            "segment_index": 0,
            "speaker": "corrupt-one",
        },
    )
    assert resp.status_code == 409
    assert _error(resp)["code"] == "edit_conflict"


def test_ingest_refused_while_book_job_live(client, tmp_path) -> None:
    from conftest import build_synthetic_epub

    data = build_synthetic_epub(tmp_path / "s.epub").read_bytes()
    first = client.post("/api/books", files={"file": ("s.epub", data, "application/epub+zip")})
    assert first.status_code == 201
    book_id = first.json()["book"]["book_id"]

    job = client.app.state.store.create(book_id, "attribute")
    again = client.post("/api/books", files={"file": ("s.epub", data, "application/epub+zip")})
    assert again.status_code == 409
    assert _error(again)["code"] == "conflicting_job"
    client.app.state.store.request_cancel(job.job_id)


def test_cover_writes_refused_while_master_live(client) -> None:
    _write_attribution(client.app.state.settings, "bk")
    job = client.app.state.store.create("bk", "master")
    put = client.put("/api/books/bk/cover", files={"file": ("c.png", b"\x89PNG\r\n", "image/png")})
    assert put.status_code == 409
    assert _error(put)["code"] == "conflicting_job"
    assert client.delete("/api/books/bk/cover").status_code == 409
    client.app.state.store.request_cancel(job.job_id)
    assert client.delete("/api/books/bk/cover").status_code == 204
