"""CLI: voice library (add-preset/blend/list/audition), assign, and the multi-voice
render + convert paths. The attribution provider and TTS engine are faked (no live LLM/GPU);
file I/O, the pipeline, and assembly (ffmpeg) are real.
"""

import pytest
from click.testing import CliRunner

from factories import make_book
from fake_engine import FakeEngine
from fake_provider import FakeProvider
from seiyuu.attribute.models import (
    CharacterMention,
    ChunkAttribution,
    Segment,
    SegmentType,
)
from seiyuu.cli import main


def _invoke(*args):
    return CliRunner().invoke(main, list(args))


def _ok(*args):
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    return result


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


@pytest.fixture
def fake_render_engine(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda engine_id, **kw: engine)
    return engine


# --- voice library commands ------------------------------------------------


def test_voice_clone_indextts2_records_engine(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF-fake-reference")  # clone copies + hashes bytes; no audio parse
    voices = tmp_path / "voices"
    _ok(
        "voice", "clone", "Mr Darcy", str(ref),
        "--engine", "indextts2", "--consent", "--consent-by", "ann",
        "--voice-id", "darcy_x", "--voices-dir", str(voices),
    )  # fmt: skip
    from seiyuu.voices import VoiceLibrary

    meta = VoiceLibrary(voices).load("darcy_x")
    assert meta.engine == "indextts2" and meta.kind.value == "cloned"


def test_voice_clone_rejects_unknown_engine(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF-fake-reference")
    result = _invoke(
        "voice", "clone", "X", str(ref), "--engine", "chaterbox", "--consent",
        "--voices-dir", str(tmp_path / "voices"),
    )  # fmt: skip
    assert result.exit_code != 0  # click.Choice rejects the typo before any file is written
    assert "chaterbox" in result.output


def test_voice_add_preset_then_list(tmp_path):
    vdir = str(tmp_path / "voices")
    _ok("voice", "add-preset", "Narrator", "af_heart", "--voice-id", "narr", "--voices-dir", vdir)
    listing = _ok("voice", "list", "--voices-dir", vdir)
    assert "narr" in listing.output
    assert "af_heart" in listing.output
    assert "[preset/kokoro]" in listing.output


def test_voice_list_empty(tmp_path):
    result = _ok("voice", "list", "--voices-dir", str(tmp_path / "voices"))
    assert "no voices yet" in result.output


def test_voice_blend_manual_is_canonicalized(tmp_path):
    vdir = str(tmp_path / "voices")
    result = _ok(
        "voice", "blend", "Lizzy", "--component", "af_sky:1", "--component", "af_bella:3",
        "--voice-id", "lizzy", "--voices-dir", vdir,
    )  # fmt: skip
    # normalized to sum 1 and sorted by preset id
    assert "af_bella:0.75, af_sky:0.25" in result.output


def test_voice_blend_auto_picks_family(tmp_path):
    result = _ok(
        "voice", "blend", "Darcy", "--gender", "male",
        "--voice-id", "darcy", "--voices-dir", str(tmp_path / "voices"),
    )  # fmt: skip
    assert "am_" in result.output  # american male presets


def test_voice_blend_bad_component_is_loud(tmp_path):
    result = _invoke(
        "voice", "blend", "X", "--component", "af_bella:lots", "--voices-dir", str(tmp_path / "v")
    )
    assert result.exit_code != 0
    assert "preset_id:weight" in result.output


def test_voice_audition_writes_wav(tmp_path, monkeypatch):
    import seiyuu.engines

    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: FakeEngine())
    vdir = tmp_path / "voices"
    _ok("voice", "add-preset", "N", "af_heart", "--voice-id", "n", "--voices-dir", str(vdir))
    _ok("voice", "audition", "n", "--voices-dir", str(vdir))
    assert (vdir / "n" / "audition.wav").is_file()


# --- assign + multi-voice render ------------------------------------------


def test_assign_creates_draft_voices(tmp_path, books_dir, fake_provider):
    vdir, odir = tmp_path / "voices", tmp_path / "out"
    _ok("attribute", "test-book", "--books-dir", str(books_dir))
    result = _ok(
        "assign", "test-book", "--books-dir", str(books_dir),
        "--output-dir", str(odir), "--voices-dir", str(vdir),
    )  # fmt: skip
    assert "Alice [alice] -> alice_auto" in result.output
    book_out = next(odir.iterdir())
    assert (book_out / "assignments.json").is_file()
    assert (vdir / "alice_auto" / "meta.json").is_file()  # deterministic auto-blend voice
    assert (vdir / "narrator_af_heart" / "meta.json").is_file()  # auto narrator preset


def test_render_multivoice_end_to_end(tmp_path, books_dir, fake_provider, fake_render_engine):
    vdir, odir = tmp_path / "voices", tmp_path / "out"
    common = ["--books-dir", str(books_dir)]
    _ok("attribute", "test-book", *common)
    _ok("assign", "test-book", *common, "--output-dir", str(odir), "--voices-dir", str(vdir))
    result = _ok(
        "render", "test-book", "--multivoice", *common,
        "--output-dir", str(odir), "--voices-dir", str(vdir),
    )  # fmt: skip
    assert "voices" in result.output

    from seiyuu.render import RenderManifest

    manifest_path = next(odir.iterdir()) / "manifest.json"
    manifest = RenderManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert "narrator_af_heart" in manifest.voices_used
    assert "alice_auto" in manifest.voices_used
    assert manifest.assignment is not None
    assert len(fake_render_engine.calls) > 0


def test_render_multivoice_without_assignment_is_loud(tmp_path, books_dir, fake_provider):
    _ok("attribute", "test-book", "--books-dir", str(books_dir))
    result = _invoke(
        "render", "test-book", "--multivoice", "--books-dir", str(books_dir),
        "--output-dir", str(tmp_path / "out"), "--voices-dir", str(tmp_path / "voices"),
    )  # fmt: skip
    assert result.exit_code != 0
    assert "seiyuu assign" in result.output


def test_convert_multivoice_end_to_end(synthetic_epub, tmp_path, fake_provider, fake_render_engine):
    result = _ok(
        "convert", str(synthetic_epub), "--multivoice",
        "--books-dir", str(tmp_path / "books"), "--output-dir", str(tmp_path / "out"),
        "--voices-dir", str(tmp_path / "voices"),
    )  # fmt: skip
    stages = ("== ingest ==", "== attribute ==", "== assign ==",
              "== render (multi-voice) ==", "== assemble ==")  # fmt: skip
    for stage in stages:
        assert stage in result.output
    out_book = next(d for d in (tmp_path / "out").iterdir() if d.is_dir())
    assert (out_book / "manifest.json").is_file()
    assert (out_book / "assignments.json").is_file()
    assert sorted((out_book / "chapters").glob("*.mp3"))
    assert len(fake_render_engine.calls) > 0
