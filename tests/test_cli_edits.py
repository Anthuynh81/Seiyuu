"""CLI: the `edit` overlay group and `voice delete` — end to end, including THE overlay
property: manual edits survive a full re-attribution."""

import pytest
from click.testing import CliRunner

from factories import make_book
from fake_provider import FakeProvider
from seiyuu.attribute.models import CharacterMention, ChunkAttribution, Segment, SegmentType
from seiyuu.cli import main


def _alice(chunk, registry, attempt):
    return ChunkAttribution(
        segments=[
            Segment(block_id=b.id, type=SegmentType.DIALOGUE, text=b.text, speaker="Alice")
            for b in chunk.owned_blocks
        ],
        characters=[CharacterMention(name="Alice", gender="female")],
    )


@pytest.fixture
def books_dir(tmp_path):
    book = make_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    (book_dir / "normalized.json").write_text(book.model_dump_json(), encoding="utf-8")
    return tmp_path / "books"


@pytest.fixture
def fake_provider(monkeypatch):
    import seiyuu.attribute.providers

    provider = FakeProvider(_alice)
    monkeypatch.setattr(seiyuu.attribute.providers, "get_provider", lambda *a, **k: provider)
    return provider


def test_edits_survive_reattribution(books_dir, fake_provider):
    """The approved overlay property, end to end: rename, re-run attribution, the
    rename is still there (and the raw file was regenerated underneath)."""
    runner = CliRunner()
    args = ["--books-dir", str(books_dir)]
    assert runner.invoke(main, ["attribute", "test-book", *args]).exit_code == 0

    r = runner.invoke(main, ["edit", "rename", "test-book", "alice", "Alicia", *args])
    assert r.exit_code == 0, r.output
    chars = runner.invoke(main, ["characters", "test-book", *args])
    assert "Alicia [alice]" in chars.output
    assert "aka Alice" in chars.output  # old name kept as an alias

    # re-attribute from scratch — the overlay must re-apply
    assert runner.invoke(main, ["attribute", "test-book", *args]).exit_code == 0
    chars = runner.invoke(main, ["characters", "test-book", *args])
    assert "Alicia [alice]" in chars.output


def test_edit_validation_and_undo(books_dir, fake_provider):
    runner = CliRunner()
    args = ["--books-dir", str(books_dir)]
    assert runner.invoke(main, ["attribute", "test-book", *args]).exit_code == 0

    r = runner.invoke(main, ["edit", "rename", "test-book", "ghost", "X", *args])
    assert r.exit_code != 0 and "unknown character" in r.output

    r = runner.invoke(main, ["edit", "reassign", "test-book", "ch001_b0002", "0", *args])
    assert r.exit_code != 0 and "exactly one" in r.output

    r = runner.invoke(
        main, ["edit", "reassign", "test-book", "ch001_b0002", "0", "--narration", *args]
    )
    assert r.exit_code == 0, r.output

    listed = runner.invoke(main, ["edit", "list", "test-book", *args])
    assert '"reassign"' in listed.output

    undo = runner.invoke(main, ["edit", "undo", "test-book", *args])
    assert undo.exit_code == 0 and "reassign" in undo.output
    assert "no manual edits" in runner.invoke(main, ["edit", "list", "test-book", *args]).output


def test_voice_delete_cli_guard_and_confirm(tmp_path):
    from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceMeta

    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(voice_id="v1", name="V", kind=VoiceKind.PRESET,
                  engine="kokoro", preset_id="af_heart")
    )  # fmt: skip
    runner = CliRunner()
    common = ["--voices-dir", str(lib.voices_dir), "--output-dir", str(tmp_path / "output")]

    aborted = runner.invoke(main, ["voice", "delete", "v1", *common], input="n\n")
    assert aborted.exit_code != 0
    assert lib.meta_path("v1").is_file()  # refused without confirmation

    ok = runner.invoke(main, ["voice", "delete", "v1", "--yes", *common])
    assert ok.exit_code == 0, ok.output
    assert not lib.dir_for("v1").exists()
