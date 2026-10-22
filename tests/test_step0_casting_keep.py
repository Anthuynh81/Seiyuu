"""Re-drafting a cast keeps the user's manual voice picks (keep_existing / --reset).

The draft/suggest routes, ``seiyuu assign`` and ``convert --multivoice`` fold the saved
assignments.json into the request via ``merge_existing_cast``: manual character picks, the
narrator and the thought voice survive; auto picks are recomputed; the request always wins;
stale picks (character gone / voice deleted) are dropped without an error.
"""

import shutil

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from seiyuu.cli import main
from seiyuu.services import (
    load_assignment,
    load_existing_assignment,
    merge_existing_cast,
    save_assignment,
)
from seiyuu.voices import VoiceAssignment, VoiceKind, VoiceLibrary, VoiceMeta
from test_api_m6b1 import make_settings
from test_api_m6b3 import _write_attribution
from test_cli_voices import fake_provider, fake_render_engine  # noqa: F401 (fixtures)

BOOK = "test-book"


def _add_voice(cfg, voice_id: str, preset: str = "am_adam") -> None:
    VoiceLibrary(cfg.voices_dir).save(
        VoiceMeta(
            voice_id=voice_id,
            name=voice_id,
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id=preset,
            source="preset",
        )
    )


def _save(cfg, *, narrator: str, assignments: dict[str, str], thought: str | None = None):
    save_assignment(
        VoiceAssignment(
            book_id=BOOK,
            narrator_voice_id=narrator,
            assignments=assignments,
            thought_voice_id=thought,
        ),
        cfg.output_dir,
    )


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.app = app
        _write_attribution(settings, BOOK)
        yield c


@pytest.fixture
def cfg(client):
    return client.app.state.settings


def _draft(client, **body) -> dict:
    resp = client.post(f"/api/books/{BOOK}/assignment/draft", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["assignment"]


# -- the pure fold ------------------------------------------------------------------------


def test_merge_without_saved_assignment_returns_request_unchanged(cfg) -> None:
    from seiyuu.services import load_report

    report, _ = load_report(cfg.books_dir / BOOK)
    lib = VoiceLibrary(cfg.voices_dir)
    kept = merge_existing_cast(
        None, report, lib, narrator_voice_id=None, thought_voice_id=None, overrides={"a": "b"}
    )
    assert kept == (None, None, {"a": "b"})
    assert load_existing_assignment(cfg.output_dir, BOOK) is None  # missing file = nothing


def test_merge_folds_manual_picks_only(cfg) -> None:
    from seiyuu.services import load_report

    report, _ = load_report(cfg.books_dir / BOOK)
    _add_voice(cfg, "picked")
    _add_voice(cfg, "alice_auto")
    _add_voice(cfg, "bob_auto")
    existing = VoiceAssignment(
        book_id=BOOK,
        narrator_voice_id="picked",
        assignments={"alice": "picked", "bob": "bob_auto"},
    )
    kept = merge_existing_cast(
        existing,
        report,
        VoiceLibrary(cfg.voices_dir),
        narrator_voice_id=None,
        thought_voice_id=None,
        overrides=None,
    )
    assert kept.overrides == {"alice": "picked"}  # bob's auto pick is NOT folded
    assert kept.narrator_voice_id == "picked"
    assert kept.thought_voice_id is None


# -- draft route --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [{}, {"strategy": "smart"}, {"strategy": "smart", "recast": True}],
    ids=["hash", "smart", "smart-recast"],
)
def test_manual_pick_survives_redraft(client, cfg, body) -> None:
    _draft(client)
    _add_voice(cfg, "picked")
    _save(cfg, narrator="narrator_af_heart", assignments={"alice": "picked", "bob": "bob_auto"})

    assignment = _draft(client, **body)
    assert assignment["assignments"] == {"alice": "picked", "bob": "bob_auto"}
    assert load_assignment(cfg.output_dir, BOOK).assignments["alice"] == "picked"


def test_narrator_and_thought_survive_redraft(client, cfg) -> None:
    _draft(client)
    _add_voice(cfg, "my_narrator")
    _add_voice(cfg, "my_thoughts", preset="af_bella")
    _save(
        cfg,
        narrator="my_narrator",
        assignments={"alice": "alice_auto", "bob": "bob_auto"},
        thought="my_thoughts",
    )
    assignment = _draft(client, strategy="smart")
    assert assignment["narrator_voice_id"] == "my_narrator"
    assert assignment["thought_voice_id"] == "my_thoughts"


def test_request_overrides_win_over_saved_picks(client, cfg) -> None:
    _draft(client)
    for vid in ("old_pick", "new_pick", "old_narr", "new_narr", "old_thought", "new_thought"):
        _add_voice(cfg, vid)
    _save(
        cfg,
        narrator="old_narr",
        assignments={"alice": "old_pick", "bob": "bob_auto"},
        thought="old_thought",
    )
    assignment = _draft(
        client,
        narrator_voice_id="new_narr",
        thought_voice_id="new_thought",
        overrides={"alice": "new_pick"},
    )
    assert assignment["assignments"]["alice"] == "new_pick"
    assert assignment["narrator_voice_id"] == "new_narr"
    assert assignment["thought_voice_id"] == "new_thought"


def test_pick_for_character_no_longer_in_registry_is_dropped(client, cfg) -> None:
    _draft(client)
    _add_voice(cfg, "carol_voice")
    _save(
        cfg,
        narrator="narrator_af_heart",
        assignments={"alice": "alice_auto", "bob": "bob_auto", "carol": "carol_voice"},
    )
    assignment = _draft(client)
    assert assignment["assignments"] == {"alice": "alice_auto", "bob": "bob_auto"}


def test_pick_whose_voice_was_deleted_is_dropped_without_error(client, cfg) -> None:
    _draft(client)
    for vid in ("gone_pick", "gone_narr", "gone_thought"):
        _add_voice(cfg, vid)
    _save(
        cfg,
        narrator="gone_narr",
        assignments={"alice": "gone_pick", "bob": "bob_auto"},
        thought="gone_thought",
    )
    for vid in ("gone_pick", "gone_narr", "gone_thought"):
        shutil.rmtree(cfg.voices_dir / vid)

    assignment = _draft(client)
    assert assignment["assignments"] == {"alice": "alice_auto", "bob": "bob_auto"}
    assert assignment["narrator_voice_id"] == "narrator_af_heart"
    assert assignment["thought_voice_id"] is None


def test_keep_existing_false_is_a_full_reset(client, cfg) -> None:
    _draft(client)
    _add_voice(cfg, "picked")
    _save(
        cfg,
        narrator="picked",
        assignments={"alice": "picked", "bob": "bob_auto"},
        thought="picked",
    )
    assignment = _draft(client, keep_existing=False)
    assert assignment["assignments"] == {"alice": "alice_auto", "bob": "bob_auto"}
    assert assignment["narrator_voice_id"] == "narrator_af_heart"
    assert assignment["thought_voice_id"] is None


def test_corrupt_saved_assignment_is_500_unless_resetting(client, cfg) -> None:
    (cfg.output_dir / BOOK).mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / BOOK / "assignments.json").write_text('{"nope": 1}', encoding="utf-8")
    for path in ("draft", "suggest"):
        resp = client.post(f"/api/books/{BOOK}/assignment/{path}", json={"strategy": "smart"})
        assert resp.status_code == 500, resp.text
        assert resp.json()["error"]["code"] == "corrupt_artifact"
    # the explicit reset doesn't read the file, so it is the way out
    assert _draft(client, keep_existing=False)["assignments"]["alice"] == "alice_auto"


def _auto_presets(cfg) -> dict[str, set[str]]:
    lib = VoiceLibrary(cfg.voices_dir)
    out: dict[str, set[str]] = {}
    for char_id in ("alice", "bob"):
        meta = lib.load(f"{char_id}_auto")
        out[char_id] = (
            {meta.preset_id} if meta.preset_id else {c.preset_id for c in meta.blend or []}
        )
    return out


def test_smart_recast_never_casts_onto_kept_narrator_or_thought_preset(client, cfg) -> None:
    # A smart draft hands alice/bob distinct presets P/Q; the user then picks a narrator on P
    # and a thought voice on Q. The kept narrator/thought must be reserved out of the cast, or
    # "Apply cast" (smart + recast) would give a character the narrator's exact voice.
    _draft(client, strategy="smart", keep_existing=False)
    first = _auto_presets(cfg)
    (p,) = first["alice"]
    (q,) = first["bob"]
    _add_voice(cfg, "my_narr", preset=p)
    _add_voice(cfg, "my_thought", preset=q)
    _save(
        cfg,
        narrator="my_narr",
        assignments={"alice": "alice_auto", "bob": "bob_auto"},
        thought="my_thought",
    )

    applied = _draft(client, strategy="smart", recast=True)
    assert applied["narrator_voice_id"] == "my_narr"
    assert applied["thought_voice_id"] == "my_thought"
    recast = _auto_presets(cfg)
    for char_id, presets in recast.items():
        assert p not in presets, (char_id, presets)
        assert q not in presets, (char_id, presets)
    assert recast["alice"].isdisjoint(recast["bob"])  # still a distinct cast


# -- suggest (preview) --------------------------------------------------------------------


def test_suggest_preview_reflects_kept_picks(client, cfg) -> None:
    _add_voice(cfg, "picked")
    _add_voice(cfg, "my_narrator")
    _save(cfg, narrator="my_narrator", assignments={"alice": "picked", "bob": "bob_auto"})

    resp = client.post(f"/api/books/{BOOK}/assignment/suggest", json={"strategy": "smart"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["assignment"]["assignments"] == {"alice": "picked", "bob": "bob_auto"}
    assert body["assignment"]["narrator_voice_id"] == "my_narrator"
    assert body["would_create_voice_ids"] == ["bob_auto"]  # alice is kept, not cast
    assert not (cfg.voices_dir / "bob_auto").exists()  # still a pure preview

    reset = client.post(
        f"/api/books/{BOOK}/assignment/suggest",
        json={"strategy": "smart", "keep_existing": False},
    ).json()
    assert reset["assignment"]["assignments"] == {"alice": "alice_auto", "bob": "bob_auto"}

    # applying matches the preview
    assert _draft(client, strategy="smart")["assignments"] == body["assignment"]["assignments"]


# -- CLI ----------------------------------------------------------------------------------


def _assign(cfg, *extra: str):
    result = CliRunner().invoke(
        main,
        [
            "assign", BOOK, "--books-dir", str(cfg.books_dir),
            "--output-dir", str(cfg.output_dir), "--voices-dir", str(cfg.voices_dir), *extra,
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    return load_assignment(cfg.output_dir, BOOK)


def test_cli_assign_keeps_picks_by_default_and_reset_rebuilds(tmp_path) -> None:
    cfg = make_settings(tmp_path)
    _write_attribution(cfg, BOOK)
    assert _assign(cfg).assignments == {"alice": "alice_auto", "bob": "bob_auto"}

    _add_voice(cfg, "picked")
    _add_voice(cfg, "other")
    _save(
        cfg,
        narrator="picked",
        assignments={"alice": "picked", "bob": "bob_auto"},
        thought="picked",
    )

    kept = _assign(cfg)
    assert kept.assignments == {"alice": "picked", "bob": "bob_auto"}
    assert kept.narrator_voice_id == "picked"
    assert kept.thought_voice_id == "picked"

    mapped = _assign(cfg, "--map", "alice=other")  # --map still wins
    assert mapped.assignments["alice"] == "other"

    reset = _assign(cfg, "--reset")
    assert reset.assignments == {"alice": "alice_auto", "bob": "bob_auto"}
    assert reset.narrator_voice_id == "narrator_af_heart"
    assert reset.thought_voice_id is None


def test_cli_assign_corrupt_saved_assignment_is_loud(tmp_path) -> None:
    cfg = make_settings(tmp_path)
    _write_attribution(cfg, BOOK)
    (cfg.output_dir / BOOK).mkdir(parents=True)
    (cfg.output_dir / BOOK / "assignments.json").write_text("{", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        [
            "assign", BOOK, "--books-dir", str(cfg.books_dir),
            "--output-dir", str(cfg.output_dir), "--voices-dir", str(cfg.voices_dir),
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "--reset" in result.output
    assert _assign(cfg, "--reset").assignments["alice"] == "alice_auto"


def test_cli_assign_by_prefix_keeps_picks(tmp_path) -> None:
    # `assign` resolves an unambiguous book_id PREFIX; the saved cast must be loaded by the
    # resolved id, else the save would silently overwrite every manual pick.
    cfg = make_settings(tmp_path)
    _write_attribution(cfg, BOOK)
    _add_voice(cfg, "picked")
    _save(
        cfg, narrator="picked", assignments={"alice": "picked", "bob": "bob_auto"}, thought="picked"
    )
    result = CliRunner().invoke(
        main,
        [
            "assign", BOOK[:7], "--books-dir", str(cfg.books_dir),
            "--output-dir", str(cfg.output_dir), "--voices-dir", str(cfg.voices_dir),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    kept = load_assignment(cfg.output_dir, BOOK)
    assert kept.assignments["alice"] == "picked"
    assert kept.narrator_voice_id == "picked"
    assert kept.thought_voice_id == "picked"
    assert not (cfg.output_dir / BOOK[:7]).exists()


def test_cli_assign_non_utf8_saved_assignment_suggests_reset(tmp_path) -> None:
    cfg = make_settings(tmp_path)
    _write_attribution(cfg, BOOK)
    (cfg.output_dir / BOOK).mkdir(parents=True)
    (cfg.output_dir / BOOK / "assignments.json").write_bytes(bytes([0xFF, 0xFE]) + b"garbage")
    result = CliRunner().invoke(
        main,
        [
            "assign", BOOK, "--books-dir", str(cfg.books_dir),
            "--output-dir", str(cfg.output_dir), "--voices-dir", str(cfg.voices_dir),
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "--reset" in result.output


def test_convert_multivoice_keeps_manual_picks(
    synthetic_epub,
    tmp_path,
    fake_provider,  # noqa: F811
    fake_render_engine,  # noqa: F811
    monkeypatch,
) -> None:
    # The synthetic EPUB has no quoted dialogue, so the narration fast path would skip the
    # (faked) provider and find no characters; this test is about casting, so turn it off.
    monkeypatch.setenv("ATTRIBUTION_NARRATION_FAST_PATH", "false")
    args = [
        "convert", str(synthetic_epub), "--multivoice",
        "--books-dir", str(tmp_path / "books"), "--output-dir", str(tmp_path / "out"),
        "--voices-dir", str(tmp_path / "voices"),
    ]  # fmt: skip
    first = CliRunner().invoke(main, args)
    assert first.exit_code == 0, first.output
    out_book = next(d for d in (tmp_path / "out").iterdir() if d.is_dir())
    saved = load_assignment(tmp_path / "out", out_book.name)
    assert saved.assignments == {"alice": "alice_auto"}

    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(
            voice_id="picked",
            name="picked",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="am_adam",
            source="preset",
        )
    )
    saved.assignments["alice"] = "picked"
    save_assignment(saved, tmp_path / "out")

    second = CliRunner().invoke(main, args)
    assert second.exit_code == 0, second.output
    assert load_assignment(tmp_path / "out", out_book.name).assignments == {"alice": "picked"}
