"""CLI: cloud voice creation, assign --map/--stage, estimate-cost, cost-confirmed render.

The ElevenLabs engine is faked (no API, no key) via render.pipeline.get_engine.
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.attribute import ATTRIBUTION_NAME
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    Segment,
    SegmentType,
)
from seiyuu.cli import main
from seiyuu.engines import AudioFile
from seiyuu.ingest import write_normalized
from seiyuu.voices import ASSIGNMENT_NAME, VoiceKind, VoiceLibrary, VoiceMeta
from test_render_cost_gate import FakeElevenEngine

BOOK_ID = "test-book-00000000"


def _invoke(*args):
    return CliRunner().invoke(main, list(args))


def _ok(*args):
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    return result


# --- voice creation ---------------------------------------------------------


def test_voice_add_cloud_and_list(tmp_path):
    vdir = str(tmp_path / "voices")
    _ok("voice", "add-cloud", "Rachel", "EXAVITQu", "--voice-id", "rachel", "--voices-dir", vdir)
    out = _ok("voice", "list", "--voices-dir", vdir).output
    assert "rachel" in out and "elevenlabs" in out and "EXAVITQu" in out


def test_voice_clone_consent_gate(tmp_path):
    ref = tmp_path / "ref.wav"
    AudioFile(samples=np.zeros(2400, dtype=np.float32)).save(ref)
    vdir = str(tmp_path / "voices")
    refused = _invoke("voice", "clone", "Elena", str(ref), "--voice-id", "e", "--voices-dir", vdir)
    assert refused.exit_code != 0 and "consent" in refused.output

    args = ["voice", "clone", "Elena", str(ref), "--engine", "elevenlabs", "--consent"]
    _ok(*args, "--voice-id", "elena", "--voices-dir", vdir)
    lib = VoiceLibrary(tmp_path / "voices")
    meta = lib.load("elena")
    assert meta.kind is VoiceKind.CLONED and meta.engine == "elevenlabs" and meta.consent_attested
    assert lib.reference_path("elena").is_file()


# --- assign --map / --stage -------------------------------------------------


def _setup_book(tmp_path):
    write_normalized(make_book(), tmp_path / "books")
    report = AttributionReport(
        book_id=BOOK_ID,
        provider_id="local",
        model_id="m",
        prompt_version="v3",
        registry=CharacterRegistry(characters=[Character(id="alice", canonical_name="Alice")]),
        chapters=[
            AttributedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    Segment(block_id="ch001_b0001", type=SegmentType.NARRATION, text="Chapter 1"),
                    Segment(
                        block_id="ch001_b0002",
                        type=SegmentType.DIALOGUE,
                        text="Hello world.",
                        speaker="alice",
                    ),
                    Segment(
                        block_id="ch001_b0004", type=SegmentType.NARRATION, text="After the break."
                    ),
                ],
            ),
            AttributedChapter(
                index=2,
                title="Chapter 2",
                segments=[
                    Segment(block_id="ch002_b0001", type=SegmentType.NARRATION, text="Chapter 2"),
                    Segment(
                        block_id="ch002_b0002",
                        type=SegmentType.DIALOGUE,
                        text="Second chapter.",
                        speaker="alice",
                    ),
                ],
            ),
        ],
    )
    (tmp_path / "books" / BOOK_ID / ATTRIBUTION_NAME).write_text(
        report.model_dump_json(), encoding="utf-8"
    )
    return tmp_path / "books"


def _add_cloud_voice(tmp_path, voice_id="elena"):
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(
            voice_id=voice_id,
            name="Elena",
            kind=VoiceKind.CLONED,
            engine="elevenlabs",
            reference_audio="reference.wav",
            consent_attested=True,
        )
    )
    AudioFile(samples=np.zeros(2400, dtype=np.float32)).save(lib.reference_path(voice_id))


def _assign_args(tmp_path, books, *extra):
    return [
        "assign", "test-book",
        "--books-dir", str(books),
        "--output-dir", str(tmp_path / "out"),
        "--voices-dir", str(tmp_path / "voices"),
        *extra,
    ]  # fmt: skip


def test_assign_map_and_stage_final(tmp_path):
    books = _setup_book(tmp_path)
    _add_cloud_voice(tmp_path)
    out = _ok(*_assign_args(tmp_path, books, "--stage", "final", "--map", "alice=elena")).output
    assert "stage: final" in out and "alice] -> elena" in out
    data = json.loads((tmp_path / "out" / BOOK_ID / ASSIGNMENT_NAME).read_text(encoding="utf-8"))
    assert data["stage"] == "final" and data["assignments"]["alice"] == "elena"


def test_assign_map_unknown_character_errors(tmp_path):
    books = _setup_book(tmp_path)
    _add_cloud_voice(tmp_path)
    result = _invoke(*_assign_args(tmp_path, books, "--map", "bob=elena"))
    assert result.exit_code != 0 and "unknown character" in result.output


# --- estimate-cost + cost-confirmed render ----------------------------------


@pytest.fixture
def fake_eleven(monkeypatch):
    eng = FakeElevenEngine()

    def fake_get(engine_id, **kw):
        return eng if engine_id == "elevenlabs" else FakeEngine()

    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", fake_get)
    return eng


def _assign_final(tmp_path, books):
    _add_cloud_voice(tmp_path)
    _ok(*_assign_args(tmp_path, books, "--stage", "final", "--map", "alice=elena"))


def _cloud_args(tmp_path, books, *head):
    return [
        *head,
        "--books-dir", str(books),
        "--output-dir", str(tmp_path / "out"),
        "--voices-dir", str(tmp_path / "voices"),
    ]  # fmt: skip


def test_estimate_cost_command(tmp_path, fake_eleven):
    books = _setup_book(tmp_path)
    _assign_final(tmp_path, books)
    out = _ok(*_cloud_args(tmp_path, books, "estimate-cost", "test-book")).output
    assert "estimated cost" in out and "2 paid segment" in out


def test_render_multivoice_confirm_cost(tmp_path, fake_eleven):
    from seiyuu.render import RenderManifest

    books = _setup_book(tmp_path)
    _assign_final(tmp_path, books)
    head = ["render", "test-book", "--multivoice", "--confirm-cost"]
    out = _ok(*_cloud_args(tmp_path, books, *head)).output
    assert "cost estimate" in out
    manifest_path = next((tmp_path / "out").glob("*/manifest.json"))
    manifest = RenderManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert "elena" in manifest.voices_used


def test_render_multivoice_paid_declined_aborts(tmp_path, fake_eleven):
    books = _setup_book(tmp_path)
    _assign_final(tmp_path, books)
    # no --confirm-cost: the interactive prompt is declined -> abort, no render
    args = _cloud_args(tmp_path, books, "render", "test-book", "--multivoice")
    result = CliRunner().invoke(main, args, input="n\n")
    assert result.exit_code != 0
    assert "cost estimate" in result.output
    assert not list((tmp_path / "out").glob("*/manifest.json"))
