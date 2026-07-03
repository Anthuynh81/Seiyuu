"""M6b-3: books routes — ingest upload, library/detail, attribute job, attribution reads."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from conftest import build_synthetic_epub
from seiyuu.api.main import create_app
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    Segment,
)
from seiyuu.repository import JobState, JobStore
from test_api_m6b1 import make_settings


@pytest.fixture(scope="module")
def epub_bytes(tmp_path_factory) -> bytes:
    path = build_synthetic_epub(tmp_path_factory.mktemp("epub") / "synthetic.epub")
    return path.read_bytes()


@pytest.fixture
def client(tmp_path):
    app = create_app(settings=make_settings(tmp_path))
    with TestClient(app) as c:
        c.app = app
        yield c


def _error(resp) -> dict:
    return resp.json()["error"]


def _upload(client, epub_bytes: bytes, **form):
    return client.post(
        "/api/books",
        files={"file": ("synthetic.epub", epub_bytes, "application/epub+zip")},
        data=form or None,
    )


def _ingested(client, epub_bytes: bytes) -> str:
    resp = _upload(client, epub_bytes)
    assert resp.status_code == 201, resp.text
    return resp.json()["book"]["book_id"]


def _write_attribution(cfg, book_id: str) -> AttributionReport:
    report = AttributionReport(
        book_id=book_id,
        provider_id="local",
        model_id="test-model",
        prompt_version="v3",
        registry=CharacterRegistry(
            characters=[
                Character(id="alice", canonical_name="Alice"),
                Character(id="bob", canonical_name="Bob"),
            ]
        ),
        chapters=[
            AttributedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    Segment(
                        block_id="ch001_b0001", type="narration", speaker=None, text="He waited."
                    ),
                    Segment(
                        block_id="ch001_b0001",
                        type="dialogue",
                        speaker="alice",
                        text='"Hello."',
                        confidence=0.95,
                    ),
                    Segment(
                        block_id="ch001_b0002",
                        type="dialogue",
                        speaker="bob",
                        text='"Hi."',
                        confidence=0.4,
                    ),
                ],
            ),
            AttributedChapter(
                index=2,
                title="Chapter 2",
                segments=[
                    Segment(
                        block_id="ch002_b0001", type="narration", speaker=None, text="Later on."
                    ),
                ],
            ),
        ],
    )
    book_dir = cfg.books_dir / book_id
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / "attribution.json").write_text(report.model_dump_json(), encoding="utf-8")
    return report


# -- ingest -------------------------------------------------------------------------------


def test_ingest_upload_and_idempotency(client, epub_bytes) -> None:
    resp = _upload(client, epub_bytes)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    book_id = body["book"]["book_id"]
    assert resp.headers["location"] == f"/api/books/{book_id}"
    assert body["book"]["ingested"] is True
    assert body["chapters"] == 3
    assert body["blocks"] > 0

    again = _upload(client, epub_bytes)  # identical bytes -> same id, 200 not 201
    assert again.status_code == 200
    assert again.json()["book"]["book_id"] == book_id

    # the temp upload dir is always cleaned up
    uploads = client.app.state.settings.data_dir / "uploads"
    assert not any(uploads.iterdir()) if uploads.is_dir() else True


def test_ingest_bad_epub_is_422(client) -> None:
    resp = _upload(client, b"this is not an epub at all")
    assert resp.status_code == 422
    assert _error(resp)["code"] == "invalid"


def test_ingest_oversize_is_413(tmp_path, epub_bytes) -> None:
    app = create_app(settings=make_settings(tmp_path, max_upload_bytes=1000))
    with TestClient(app) as c:
        resp = _upload(c, epub_bytes)
    assert resp.status_code == 413
    assert _error(resp)["code"] == "payload_too_large"


# -- library / detail ---------------------------------------------------------------------


def test_library_cards_and_active_job(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    store: JobStore = client.app.state.store

    books = client.get("/api/books").json()["books"]
    assert [b["book_id"] for b in books] == [book_id]
    assert books[0]["active_job"] is None

    job = store.create(book_id, "attribute")
    store.mark_running(job.job_id)
    card = client.get("/api/books").json()["books"][0]
    assert card["active_job"] == {"job_id": job.job_id, "kind": "attribute", "state": "running"}
    # polling discipline: the card must be useless as a progress poll


def test_book_detail(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    body = client.get(f"/api/books/{book_id}").json()
    assert body["status"]["ingested"] is True
    assert len(body["chapters"]) == 3
    assert body["chapters"][0]["index"] == 1
    assert body["chapters"][0]["speakable_blocks"] > 0
    assert body["runtime_estimate_seconds"] > 0
    assert body["downloads"] == {"m4b": None, "chapter_mp3s": []}
    assert body["cover"] is None
    assert body["recent_jobs"] == []


def test_book_detail_unknown_404_and_bad_id_422(client) -> None:
    resp = client.get("/api/books/ghost")
    assert resp.status_code == 404
    assert _error(resp)["code"] == "not_found"
    assert client.get("/api/books/a:b").status_code == 422


def test_downloads_listed_and_served(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    odir = client.app.state.settings.output_dir / book_id
    (odir / "chapters").mkdir(parents=True)
    (odir / f"{book_id}.m4b").write_bytes(b"m4b-bytes")
    (odir / "chapters" / "ch001.mp3").write_bytes(b"mp3-bytes")

    downloads = client.get(f"/api/books/{book_id}").json()["downloads"]
    assert downloads["m4b"]["url"] == f"/api/books/{book_id}/files/m4b"
    assert downloads["chapter_mp3s"][0]["index"] == 1

    m4b = client.get(f"/api/books/{book_id}/files/m4b")
    assert m4b.status_code == 200
    assert m4b.headers["content-type"].startswith("audio/mp4")
    assert m4b.content == b"m4b-bytes"
    mp3 = client.get(f"/api/books/{book_id}/files/chapters/1")
    assert mp3.status_code == 200
    assert mp3.content == b"mp3-bytes"
    assert client.get(f"/api/books/{book_id}/files/chapters/2").status_code == 404


# -- runtime estimate ---------------------------------------------------------------------


def test_runtime_estimate(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    full = client.get(f"/api/books/{book_id}/runtime-estimate").json()
    assert full["seconds"] > 0
    assert full["wpm_used"] == 150.0
    one = client.get(f"/api/books/{book_id}/runtime-estimate", params={"chapters": 1}).json()
    assert 0 < one["seconds"] < full["seconds"]
    fast = client.get(f"/api/books/{book_id}/runtime-estimate", params={"wpm": 300}).json()
    assert fast["seconds"] == pytest.approx(full["seconds"] / 2)

    resp = client.get(f"/api/books/{book_id}/runtime-estimate", params={"chapters": 99})
    assert resp.status_code == 422
    assert "out of range" in _error(resp)["message"]


# -- attribute job ------------------------------------------------------------------------


def test_attribute_enqueue_and_duplicate(client, epub_bytes, monkeypatch) -> None:
    book_id = _ingested(client, epub_bytes)
    release = threading.Event()
    monkeypatch.setattr(
        "seiyuu.api.handlers.run_attribution",
        lambda *a, **kw: release.wait(timeout=5.0),
    )

    resp = client.post(f"/api/books/{book_id}/attribute", json={})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.headers["location"] == f"/api/jobs/{job_id}"

    dup = client.post(f"/api/books/{book_id}/attribute", json={"chapters": [1]})
    assert dup.status_code == 409
    assert _error(dup)["code"] == "duplicate_job"

    release.set()
    store: JobStore = client.app.state.store
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not store.get(job_id).is_terminal:
        time.sleep(0.02)
    assert store.get(job_id).state is JobState.SUCCEEDED


def test_attribute_paid_gate_on_effective_values(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)

    resp = client.post(f"/api/books/{book_id}/attribute", json={"use_hybrid": True})
    assert resp.status_code == 402
    assert _error(resp)["code"] == "payment_confirmation_required"

    resp = client.post(f"/api/books/{book_id}/attribute", json={"provider": "anthropic"})
    assert resp.status_code == 402

    confirmed = client.post(
        f"/api/books/{book_id}/attribute", json={"use_hybrid": True, "confirm_paid": True}
    )
    assert confirmed.status_code == 503  # confirmed, but no key configured
    assert _error(confirmed)["code"] == "not_ready"


def test_attribute_hybrid_settings_default_needs_confirm(tmp_path, epub_bytes) -> None:
    # A .env default of attribution_hybrid=true must not silently run paid attribution.
    app = create_app(settings=make_settings(tmp_path, attribution_hybrid=True))
    with TestClient(app) as c:
        c.app = app
        book_id = _ingested(c, epub_bytes)
        resp = c.post(f"/api/books/{book_id}/attribute", json={})
        assert resp.status_code == 402
        # and an explicit opt-OUT overrides the paid default without confirmation
        release_free = c.post(f"/api/books/{book_id}/attribute", json={"use_hybrid": False})
        assert release_free.status_code == 202


def test_attribute_not_ingested_is_409(client) -> None:
    odir = client.app.state.settings.output_dir / "bk-output-only"
    odir.mkdir(parents=True)
    (odir / "manifest.json").write_text("{}", encoding="utf-8")
    resp = client.post("/api/books/bk-output-only/attribute", json={})
    assert resp.status_code == 409
    assert _error(resp)["code"] == "stage_prerequisite"


def test_attribute_chapter_out_of_range_422(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    resp = client.post(f"/api/books/{book_id}/attribute", json={"chapters": [42]})
    assert resp.status_code == 422


# -- attribution reads --------------------------------------------------------------------


def test_attribution_read_and_filter(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    missing = client.get(f"/api/books/{book_id}/attribution")
    assert missing.status_code == 404

    _write_attribution(client.app.state.settings, book_id)
    body = client.get(f"/api/books/{book_id}/attribution").json()
    assert body["edit_warnings"] == []
    assert [c["index"] for c in body["report"]["chapters"]] == [1, 2]
    assert {c["id"] for c in body["report"]["registry"]["characters"]} == {"alice", "bob"}

    filtered = client.get(f"/api/books/{book_id}/attribution", params={"chapters": 2}).json()[
        "report"
    ]
    assert [c["index"] for c in filtered["chapters"]] == [2]
    assert len(filtered["registry"]["characters"]) == 2  # registry never trimmed


def test_attribution_corrupt_is_500(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    cfg = client.app.state.settings
    (cfg.books_dir / book_id / "attribution.json").write_text("{broken", encoding="utf-8")
    resp = client.get(f"/api/books/{book_id}/attribution")
    assert resp.status_code == 500
    assert _error(resp)["code"] == "corrupt_artifact"


def test_segment_browser(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    _write_attribution(client.app.state.settings, book_id)

    body = client.get(f"/api/books/{book_id}/chapters/1/segments").json()
    assert body["title"] == "Chapter 1"
    rows = body["segments"]
    assert [(r["block_id"], r["segment_index"]) for r in rows] == [
        ("ch001_b0001", 0),
        ("ch001_b0001", 1),
        ("ch001_b0002", 0),
    ]
    assert rows[1]["speaker_name"] == "Alice"
    assert all(r["has_audio"] is False for r in rows)

    narration = client.get(
        f"/api/books/{book_id}/chapters/1/segments", params={"speaker": "narration"}
    ).json()["segments"]
    assert len(narration) == 1 and narration[0]["speaker"] is None

    low = client.get(
        f"/api/books/{book_id}/chapters/1/segments", params={"low_confidence": True}
    ).json()["segments"]
    assert [r["speaker"] for r in low] == ["bob"]
    assert low[0]["segment_index"] == 0  # index counts pre-filter, within the block

    assert client.get(f"/api/books/{book_id}/chapters/9/segments").status_code == 404


def test_segment_browser_has_audio_from_manifest(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    _write_attribution(client.app.state.settings, book_id)
    from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest

    manifest = RenderManifest(
        book_id=book_id,
        chapters=[
            RenderedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    RenderedSegment(
                        block_id="ch001_b0001",
                        type="paragraph",
                        wav="cache/x.wav",
                        duration_seconds=1.0,
                    )
                ],
            )
        ],
    )
    odir = client.app.state.settings.output_dir / book_id
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    rows = client.get(f"/api/books/{book_id}/chapters/1/segments").json()["segments"]
    by_block = {r["block_id"]: r["has_audio"] for r in rows}
    assert by_block["ch001_b0001"] is True
    assert by_block["ch001_b0002"] is False


def test_characters_route(client, epub_bytes) -> None:
    book_id = _ingested(client, epub_bytes)
    missing = client.get(f"/api/books/{book_id}/characters")
    assert missing.status_code == 404

    _write_attribution(client.app.state.settings, book_id)
    body = client.get(f"/api/books/{book_id}/characters").json()
    assert body["book_id"] == book_id
    assert body["narration_segments"] == 2
    assert body["low_confidence_segments"] == 1  # bob at 0.4 < 0.7
    by_id = {c["id"]: c for c in body["characters"]}
    assert by_id["alice"]["line_count"] == 1
    assert by_id["alice"]["sample_lines"] == ['"Hello."']

    bare = client.get(f"/api/books/{book_id}/characters", params={"sample_lines": 0}).json()
    assert all(c["sample_lines"] == [] for c in bare["characters"])
