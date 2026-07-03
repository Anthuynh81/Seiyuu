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


def _make_preset(client, voice_id="v1", engine="fake", preset_id="test_voice") -> dict:
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
    engines = {e["engine_id"]: e for e in client.get("/api/engines").json()["engines"]}
    assert "fake" not in engines  # sanity: catalog stays real; residency read next
    assert client.app.state.registry.is_resident("fake") is True


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


def test_cloud_audition_refused_while_render_live(client, monkeypatch) -> None:
    _patch_audition(monkeypatch, PaidCloudFake(), cls_by_id={"elevenlabs": PaidCloudFake})
    _make_preset(client, voice_id="cloudv", engine="elevenlabs", preset_id="stock1")
    store: JobStore = client.app.state.store
    render = store.create("bk", "render")

    resp = client.post("/api/voices/cloudv/audition", json={"confirm_paid": True})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "cloud_busy"  # eviction-race closure (Q6)

    store.request_cancel(render.job_id)
    # an attribute job does NOT block a cloud audition (no GPU, no slots)
    attr = store.create("bk", "attribute")
    ok = client.post("/api/voices/cloudv/audition", json={"confirm_paid": True})
    assert ok.status_code == 200, ok.text
    assert ok.json()["cost_usd"] > 0
    store.request_cancel(attr.job_id)


def test_paid_audition_requires_confirmation(client, monkeypatch) -> None:
    _patch_audition(monkeypatch, PaidCloudFake(), cls_by_id={"elevenlabs": PaidCloudFake})
    _make_preset(client, voice_id="cloudv", engine="elevenlabs", preset_id="stock1")
    resp = client.post("/api/voices/cloudv/audition", json={})
    assert resp.status_code == 402
    err = _error(resp)
    assert err["code"] == "payment_confirmation_required"
    assert err["detail"]["estimated_usd"] > 0


def test_audition_in_flight_refusal(client, monkeypatch) -> None:
    _patch_audition(monkeypatch)
    _make_preset(client)
    with client.app.state.gate.hold("audition"):
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
    assert err["detail"]["warmup"] == "/api/engines/fake/warmup"


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

    png = client.put(
        "/api/books/bk/cover",
        files={"file": ("c.png", b"\x89PNG\r\n\x1a\nrest", "image/png")},
    )
    assert png.status_code == 200, png.text
    assert png.json()["content_type"] == "image/png"
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
