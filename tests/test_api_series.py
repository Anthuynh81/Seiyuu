"""F5 API — series routes end-to-end: seed, list/get, join, suggestions, inheritance
overrides fed to the draft seam, explicit write-back, unlink, and book-delete membership drop.
"""

import pytest
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from test_api_m6b1 import make_settings
from test_api_m6b3 import _write_attribution


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        yield c


def _preset(client, voice_id: str, preset_id: str = "af_bella") -> None:
    resp = client.post(
        "/api/voices",
        json={
            "kind": "preset",
            "name": voice_id,
            "engine": "kokoro",
            "preset_id": preset_id,
            "voice_id": voice_id,
        },
    )
    assert resp.status_code == 201, resp.text


def _draft(client, book_id: str, **overrides) -> dict:
    resp = client.post(
        f"/api/books/{book_id}/assignment/draft",
        json={"strategy": "smart", **overrides},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["assignment"]


def test_series_full_flow(client) -> None:
    cfg = client.app.state.settings
    # Book 1: attributed with Alice + Bob, then assign Alice an explicit voice.
    _write_attribution(cfg, "book-one")
    _preset(client, "alice_voice", "af_nicole")
    a1 = _draft(client, "book-one", overrides={"alice": "alice_voice"})
    assert a1["assignments"]["alice"] == "alice_voice"

    # Create a series seeded from book 1's cast.
    resp = client.post(
        "/api/series",
        json={"name": "Wonderland", "book_id": "book-one", "series_id": "wl"},
    )
    assert resp.status_code == 201, resp.text
    series = resp.json()
    assert series["book_ids"] == ["book-one"]
    assert series["voice_links"]["alice"] == "alice_voice"

    assert client.get("/api/series").json()["series"][0]["series_id"] == "wl"
    assert client.get("/api/series/wl").json()["name"] == "Wonderland"

    # Book 2: same characters, join the series.
    _write_attribution(cfg, "book-two")
    assert client.post("/api/series/wl/books", json={"book_id": "book-two"}).status_code == 200

    # Suggestions surface Alice (linked voice exists) — for confirmation.
    sugg = client.get("/api/series/wl/books/book-two/link-suggestions").json()["suggestions"]
    assert any(s["character_id"] == "alice" and s["voice_id"] == "alice_voice" for s in sugg)

    # Overrides resolve to book 1's whole seeded cast (Alice's explicit voice + Bob's auto),
    # since both linked voices still exist in the global library.
    overrides = client.get("/api/series/wl/books/book-two/overrides").json()["overrides"]
    assert overrides == {"alice": "alice_voice", "bob": "bob_auto"}

    # ...and feeding them to the draft seam makes book 2 inherit book 1's voices.
    a2 = _draft(client, "book-two", overrides=overrides)
    assert a2["assignments"]["alice"] == "alice_voice"
    assert a2["assignments"]["bob"] == "bob_auto"

    # Explicit write-back: fold book 2's cast (incl. Bob's auto voice) into the series links.
    saved = client.post("/api/series/wl/save-cast", json={"book_id": "book-two"}).json()
    assert "bob" in saved["series"]["voice_links"]

    # Unlink Alice by name (case-insensitive).
    after = client.request("DELETE", "/api/series/wl/links", params={"name": "ALICE"}).json()
    assert "alice" not in after["voice_links"]


def test_deleting_a_book_drops_series_membership(client) -> None:
    cfg = client.app.state.settings
    _write_attribution(cfg, "book-one")
    _draft(client, "book-one")
    client.post("/api/series", json={"name": "S", "book_id": "book-one", "series_id": "s1"})
    assert client.get("/api/series/s1").json()["book_ids"] == ["book-one"]

    assert client.delete("/api/books/book-one").status_code == 200
    # membership is gone — no dangling ghost id in the series
    assert client.get("/api/series/s1").json()["book_ids"] == []


def test_series_404s(client) -> None:
    assert client.get("/api/series/nope").status_code == 404
    assert client.post("/api/series/nope/books", json={"book_id": "x"}).status_code in (404, 422)
    cfg = client.app.state.settings
    _write_attribution(cfg, "book-one")
    # creating a series from an un-ASSIGNED book fails cleanly (no assignments.json yet)
    resp = client.post("/api/series", json={"name": "S", "book_id": "book-one"})
    assert resp.status_code == 409
