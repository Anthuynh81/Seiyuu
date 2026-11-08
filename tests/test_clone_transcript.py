"""A Qwen3-TTS clone binds its clip's transcript (ICL mode) when it safely can.

Whisper is always faked here: the default suite never loads a speech model.
"""

import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from seiyuu.cli import main
from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
from seiyuu.services.clone_transcript import (
    ICL_MAX_SECONDS,
    bind_transcript,
    choose_transcript,
)
from seiyuu.validate import ValidationError, Validator
from seiyuu.voices import VoiceLibrary
from seiyuu.voices.library import sha256_file
from test_api_m6b1 import make_settings


def _heard(text: str, seconds: float = 8.0):
    return lambda _path: (text, seconds)


def _deaf(_path):
    raise ValidationError("whisper transcription failed for clip.wav: bad header")


# -- choosing the transcript -------------------------------------------------------------------


def test_whisper_transcript_is_bound_for_a_short_clip(tmp_path):
    chosen = choose_transcript(tmp_path, provided=None, transcribe=_heard("  Hello   there. "))
    assert chosen.text == "Hello there."
    assert "transcript" in chosen.note


def test_a_typed_transcript_wins_over_whisper(tmp_path):
    chosen = choose_transcript(tmp_path, provided="Hello, there!", transcribe=_heard("hello dare"))
    assert chosen.text == "Hello, there!"
    assert "your transcript" in chosen.note


def test_a_long_clip_clones_from_the_sound_alone_even_with_a_typed_transcript(tmp_path):
    long = _heard("words", ICL_MAX_SECONDS + 1)
    for typed in (None, "words"):
        chosen = choose_transcript(tmp_path, provided=typed, transcribe=long)
        assert chosen.text is None
        assert "31 s" in chosen.note and "sound alone" in chosen.note


def test_silence_clones_from_the_sound_alone(tmp_path):
    chosen = choose_transcript(tmp_path, provided="  ", transcribe=_heard(""))
    assert chosen.text is None and "no speech" in chosen.note


def test_a_whisper_failure_falls_back_unless_the_user_typed_one(tmp_path):
    assert choose_transcript(tmp_path, provided=None, transcribe=_deaf).text is None
    assert choose_transcript(tmp_path, provided="Hi.", transcribe=_deaf).text == "Hi."


# -- binding ------------------------------------------------------------------------------------


def test_bind_writes_the_transcript_and_the_digest_the_engine_checks(tmp_path):
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"RIFF-clip")
    sha = sha256_file(ref)
    settings = bind_transcript(ref, sha, "Hello there.")
    assert (tmp_path / "reference.txt").read_text(encoding="utf-8") == "Hello there.\n"
    assert settings == {
        "qwen3tts": {REF_TEXT_DIGEST_KEY: reference_text_digest(sha, "Hello there.")}
    }


def test_bind_none_removes_a_stale_transcript(tmp_path):
    ref = tmp_path / "reference.wav"
    (tmp_path / "reference.txt").write_text("old words\n", encoding="utf-8")
    assert bind_transcript(ref, "0" * 64, None) == {}
    assert not (tmp_path / "reference.txt").exists()


# -- CLI ----------------------------------------------------------------------------------------


@pytest.fixture
def fake_whisper(monkeypatch):
    calls = []

    def transcribe_clip(self, path):
        calls.append((self.device, path))
        return "It was a dark and stormy night.", 6.0

    monkeypatch.setattr(Validator, "transcribe_clip", transcribe_clip)
    return calls


def _cli_clone(tmp_path, *extra):
    ref = tmp_path / "clip.wav"
    if not ref.exists():
        ref.write_bytes(b"RIFF-fake-reference")
    return CliRunner().invoke(
        main,
        ["voice", "clone", "Elena", str(ref), "--consent", "--consent-by", "Ann",
         "--voice-id", "elena_x", "--voices-dir", str(tmp_path / "voices"), *extra],
    )  # fmt: skip


def test_cli_clone_defaults_to_qwen_and_binds_the_transcript(tmp_path, fake_whisper):
    result = _cli_clone(tmp_path)
    assert result.exit_code == 0, result.output
    assert "cloned from the clip and its transcript" in result.output
    lib = VoiceLibrary(tmp_path / "voices")
    meta = lib.load("elena_x")
    assert meta.engine == "qwen3tts"
    ref = lib.reference_path("elena_x")
    text = ref.with_name("reference.txt").read_text(encoding="utf-8").strip()
    assert text == "It was a dark and stormy night."
    expected = reference_text_digest(meta.consent.reference_sha256, text)
    assert meta.settings["qwen3tts"][REF_TEXT_DIGEST_KEY] == expected
    lib.verify_consent(meta)
    assert fake_whisper[0][0] == "cpu"  # never a second model on the GPU


def test_cli_typed_transcript_is_bound(tmp_path, fake_whisper):
    result = _cli_clone(tmp_path, "--transcript", "Call me Elena.")
    assert result.exit_code == 0, result.output
    ref = VoiceLibrary(tmp_path / "voices").reference_path("elena_x")
    assert ref.with_name("reference.txt").read_text(encoding="utf-8") == "Call me Elena.\n"


def test_cli_reclone_on_another_engine_drops_the_transcript(tmp_path, fake_whisper):
    assert _cli_clone(tmp_path).exit_code == 0
    result = _cli_clone(tmp_path, "--engine", "chatterbox")
    assert result.exit_code == 0, result.output
    lib = VoiceLibrary(tmp_path / "voices")
    meta = lib.load("elena_x")
    assert meta.engine == "chatterbox" and meta.settings == {}
    assert not lib.reference_path("elena_x").with_name("reference.txt").exists()
    assert len(fake_whisper) == 1  # chatterbox never transcribes


# -- API ----------------------------------------------------------------------------------------


class _FakeAligner:
    def __init__(self, text="It was a dark and stormy night.", seconds=6.0):
        self.text, self.seconds, self.calls = text, seconds, 0

    def transcribe_clip(self, _path):
        self.calls += 1
        return self.text, self.seconds


@pytest.fixture
def client(tmp_path):
    app = create_app(settings=make_settings(tmp_path))
    with TestClient(app) as c:
        app.state.aligner = _FakeAligner()
        c.app = app
        yield c


def _post_clone(client, **form):
    return client.post(
        "/api/voices/clone",
        files={"file": ("ref.wav", b"RIFF-fake-reference", "audio/wav")},
        data={"name": "Elena", "consent": "true", "attested_by": "Ann", "voice_id": "elena_x",
              **form},
    )  # fmt: skip


def test_api_clone_defaults_to_qwen_and_binds_the_transcript(client):
    resp = _post_clone(client)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["engine"] == "qwen3tts"
    assert body["clone_note"] == "cloned from the clip and its transcript"
    digest = reference_text_digest(
        body["consent"]["reference_sha256"], "It was a dark and stormy night."
    )
    assert body["settings"]["qwen3tts"][REF_TEXT_DIGEST_KEY] == digest
    voices = client.app.state.settings.voices_dir
    meta = json.loads((voices / "elena_x" / "meta.json").read_text(encoding="utf-8"))
    assert meta["settings"]["qwen3tts"][REF_TEXT_DIGEST_KEY] == digest
    assert (voices / "elena_x" / "reference.txt").is_file()


def test_api_long_clip_clones_from_the_sound_alone(client):
    client.app.state.aligner = _FakeAligner(seconds=45.0)
    body = _post_clone(client).json()
    assert body["settings"] == {}
    assert "45 s" in body["clone_note"]
    assert not (client.app.state.settings.voices_dir / "elena_x" / "reference.txt").exists()


def test_api_typed_transcript_is_used(client):
    body = _post_clone(client, transcript="Call me Elena.").json()
    voices = client.app.state.settings.voices_dir
    assert (voices / "elena_x" / "reference.txt").read_text(encoding="utf-8") == "Call me Elena.\n"
    assert body["clone_note"] == "cloned from the clip and your transcript"


def test_api_other_engines_skip_whisper(client):
    body = _post_clone(client, engine="chatterbox").json()
    assert body["engine"] == "chatterbox" and body["clone_note"] is None
    assert client.app.state.aligner.calls == 0
