"""M6b-4: edits overlay endpoints + assignment draft/full-replace, with write guards."""

import pytest
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from seiyuu.repository import JobStore
from test_api_m6b1 import make_settings
from test_api_m6b3 import _write_attribution

BOOK = "test-book"


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        _write_attribution(settings, BOOK)  # attributed marker = the book exists
        yield c


def _error(resp) -> dict:
    return resp.json()["error"]


def _drafted(client) -> dict:
    resp = client.post(f"/api/books/{BOOK}/assignment/draft", json={})
    assert resp.status_code == 201, resp.text
    return resp.json()


# -- edits: read + record -----------------------------------------------------------------


def test_edits_empty_log_and_unknown_book(client) -> None:
    assert client.get(f"/api/books/{BOOK}/edits").json() == {"version": 1, "ops": []}
    assert client.get("/api/books/ghost/edits").status_code == 404


def test_record_rename_returns_anchored_op(client) -> None:
    resp = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "rename", "character_id": "alice", "new_name": "Alicia"},
    )
    assert resp.status_code == 201, resp.text
    op = resp.json()
    assert op["expected_name"] == "Alice"  # server-filled anchor: what the user saw

    log = client.get(f"/api/books/{BOOK}/edits").json()
    assert len(log["ops"]) == 1

    report = client.get(f"/api/books/{BOOK}/attribution").json()["report"]
    alice = next(c for c in report["registry"]["characters"] if c["id"] == "alice")
    assert alice["canonical_name"] == "Alicia"
    assert "Alice" in alice["aliases"]


def test_record_reassign_and_anchor_staleness(client) -> None:
    resp = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "reassign", "block_id": "ch001_b0002", "segment_index": 0, "speaker": None},
    )
    assert resp.status_code == 201
    assert resp.json()["text_anchor"] == '"Hi."'

    rows = client.get(f"/api/books/{BOOK}/chapters/1/segments").json()["segments"]
    reassigned = next(r for r in rows if r["block_id"] == "ch001_b0002")
    assert reassigned["speaker"] is None  # null speaker -> narration
    assert reassigned["type"] == "narration"
    assert reassigned["confidence"] == 1.0  # manual = ground truth; review queue drains

    # simulate re-attribution that re-splits the block into DIFFERENT text: the op must
    # skip with a warning, never silently retarget
    _write_attribution(client.app.state.settings, BOOK)
    cfg = client.app.state.settings
    raw = (cfg.books_dir / BOOK / "attribution.json").read_text(encoding="utf-8")
    (cfg.books_dir / BOOK / "attribution.json").write_text(
        raw.replace('\\"Hi.\\"', '\\"Completely new words.\\"'), encoding="utf-8"
    )
    body = client.get(f"/api/books/{BOOK}/attribution").json()
    assert any("reassign skipped" in w for w in body["edit_warnings"])


def test_record_merge_and_self_merge_rejected(client) -> None:
    resp = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "merge", "loser_id": "bob", "winner_id": "alice"},
    )
    assert resp.status_code == 201
    op = resp.json()
    assert (op["expected_loser_name"], op["expected_winner_name"]) == ("Bob", "Alice")

    report = client.get(f"/api/books/{BOOK}/attribution").json()["report"]
    assert {c["id"] for c in report["registry"]["characters"]} == {"alice"}
    speakers = {s["speaker"] for ch in report["chapters"] for s in ch["segments"] if s["speaker"]}
    assert speakers == {"alice"}

    self_merge = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "merge", "loser_id": "alice", "winner_id": "alice"},
    )
    assert self_merge.status_code == 422  # shape error, refused before any service call


def test_edit_conflict_is_409(client) -> None:
    resp = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "rename", "character_id": "nobody", "new_name": "X"},
    )
    assert resp.status_code == 409
    assert _error(resp)["code"] == "edit_conflict"

    out_of_range = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "reassign", "block_id": "ch001_b0001", "segment_index": 9, "speaker": None},
    )
    assert out_of_range.status_code == 409
    assert "out of range" in _error(out_of_range)["message"]


def test_client_supplied_anchors_are_structurally_rejected(client) -> None:
    resp = client.post(
        f"/api/books/{BOOK}/edits",
        json={
            "op": "rename",
            "character_id": "alice",
            "new_name": "X",
            "expected_name": "Forged",
        },
    )
    assert resp.status_code == 422  # extra="forbid": anchoring is server-authoritative
    resp = client.post(
        f"/api/books/{BOOK}/edits",
        json={
            "op": "reassign",
            "block_id": "ch001_b0001",
            "segment_index": 0,
            "speaker": None,
            "text_anchor": "forged",
        },
    )
    assert resp.status_code == 422


def test_reassign_requires_explicit_speaker_field(client) -> None:
    resp = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "reassign", "block_id": "ch001_b0001", "segment_index": 0},
    )
    assert resp.status_code == 422  # speaker is required-nullable, not defaulted


def test_undo_last_edit(client) -> None:
    client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "rename", "character_id": "alice", "new_name": "Alicia"},
    )
    resp = client.request("DELETE", f"/api/books/{BOOK}/edits/last")
    assert resp.status_code == 200
    assert resp.json()["removed"]["op"] == "rename"

    report = client.get(f"/api/books/{BOOK}/attribution").json()["report"]
    alice = next(c for c in report["registry"]["characters"] if c["id"] == "alice")
    assert alice["canonical_name"] == "Alice"  # back to the raw report

    empty = client.request("DELETE", f"/api/books/{BOOK}/edits/last")
    assert empty.status_code == 404
    assert "no manual edits" in _error(empty)["message"]


def test_corrupt_edits_file_is_500(client) -> None:
    cfg = client.app.state.settings
    (cfg.books_dir / BOOK / "edits.json").write_text("{broken", encoding="utf-8")
    resp = client.get(f"/api/books/{BOOK}/edits")
    assert resp.status_code == 500
    assert _error(resp)["code"] == "corrupt_artifact"


# -- render_active write guards -----------------------------------------------------------


def test_writes_refused_while_render_active(client) -> None:
    store: JobStore = client.app.state.store
    render = store.create(BOOK, "render")

    edit = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "rename", "character_id": "alice", "new_name": "X"},
    )
    assert edit.status_code == 409
    err = _error(edit)
    assert err["code"] == "render_active"
    assert err["detail"]["job_id"] == render.job_id  # the job to cancel, right in detail

    assert client.request("DELETE", f"/api/books/{BOOK}/edits/last").status_code == 409
    assert client.post(f"/api/books/{BOOK}/assignment/draft", json={}).status_code == 409
    put = client.put(
        f"/api/books/{BOOK}/assignment",
        json={"stage": "draft", "narrator_voice_id": "n", "assignments": {}},
    )
    assert put.status_code == 409

    # a non-render job does NOT block edits (attribution jobs conflict at render-enqueue
    # time instead, per the scoping doc's conflict matrix)
    store.request_cancel(render.job_id)
    other = store.create(BOOK, "attribute")
    ok = client.post(
        f"/api/books/{BOOK}/edits",
        json={"op": "rename", "character_id": "alice", "new_name": "Y"},
    )
    assert ok.status_code == 201, ok.text
    store.request_cancel(other.job_id)


# -- assignment ---------------------------------------------------------------------------


def test_assignment_404_before_draft(client) -> None:
    resp = client.get(f"/api/books/{BOOK}/assignment")
    assert resp.status_code == 404


def test_draft_creates_deterministic_voices(client) -> None:
    body = _drafted(client)
    assignment = body["assignment"]
    assert assignment["book_id"] == BOOK
    assert assignment["stage"] == "draft"
    assert assignment["narrator_voice_id"] == "narrator_af_heart"
    assert assignment["assignments"] == {"alice": "alice_auto", "bob": "bob_auto"}
    assert set(body["created_voice_ids"]) == {"narrator_af_heart", "alice_auto", "bob_auto"}

    again = client.post(f"/api/books/{BOOK}/assignment/draft", json={})
    assert again.status_code == 201
    assert again.json()["created_voice_ids"] == []  # deterministic: nothing new

    read = client.get(f"/api/books/{BOOK}/assignment")
    assert read.status_code == 200
    assert read.json()["assignments"] == {"alice": "alice_auto", "bob": "bob_auto"}


def test_draft_unknown_narrator_is_422(client) -> None:
    resp = client.post(f"/api/books/{BOOK}/assignment/draft", json={"narrator_voice_id": "ghost"})
    assert resp.status_code == 422
    assert "not in the library" in _error(resp)["message"]


def test_put_full_replace_validation(client) -> None:
    _drafted(client)

    incomplete = client.put(
        f"/api/books/{BOOK}/assignment",
        json={
            "stage": "final",
            "narrator_voice_id": "narrator_af_heart",
            "assignments": {"alice": "alice_auto"},  # bob speaks but is missing
        },
    )
    assert incomplete.status_code == 422
    assert "bob" in _error(incomplete)["message"]

    unknown_char = client.put(
        f"/api/books/{BOOK}/assignment",
        json={
            "stage": "final",
            "narrator_voice_id": "narrator_af_heart",
            "assignments": {"alice": "alice_auto", "bob": "bob_auto", "eve": "alice_auto"},
        },
    )
    assert unknown_char.status_code == 422
    assert "eve" in _error(unknown_char)["message"]

    unknown_voice = client.put(
        f"/api/books/{BOOK}/assignment",
        json={
            "stage": "final",
            "narrator_voice_id": "narrator_af_heart",
            "assignments": {"alice": "alice_auto", "bob": "ghost_voice"},
        },
    )
    assert unknown_voice.status_code == 422

    ok = client.put(
        f"/api/books/{BOOK}/assignment",
        json={
            "stage": "final",
            "narrator_voice_id": "narrator_af_heart",
            "assignments": {"alice": "alice_auto", "bob": "bob_auto"},
        },
    )
    assert ok.status_code == 200, ok.text
    saved = ok.json()
    assert saved["stage"] == "final"
    assert saved["schema_version"] == 1  # server-filled
    assert saved["created_at"]
    assert client.get(f"/api/books/{BOOK}/assignment").json()["stage"] == "final"


def test_put_respects_edit_overlay_characters(client) -> None:
    # merge bob into alice, then a map WITHOUT bob is complete — the PUT validates
    # against the EFFECTIVE report, not the raw one
    _drafted(client)
    assert (
        client.post(
            f"/api/books/{BOOK}/edits",
            json={"op": "merge", "loser_id": "bob", "winner_id": "alice"},
        ).status_code
        == 201
    )
    ok = client.put(
        f"/api/books/{BOOK}/assignment",
        json={
            "stage": "draft",
            "narrator_voice_id": "narrator_af_heart",
            "assignments": {"alice": "alice_auto"},
        },
    )
    assert ok.status_code == 200, ok.text


def test_corrupt_assignment_is_500(client) -> None:
    cfg = client.app.state.settings
    (cfg.output_dir / BOOK).mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / BOOK / "assignments.json").write_text('{"nope": 1}', encoding="utf-8")
    resp = client.get(f"/api/books/{BOOK}/assignment")
    assert resp.status_code == 500
    assert _error(resp)["code"] == "corrupt_artifact"
