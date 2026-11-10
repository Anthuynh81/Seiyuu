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


# -- Breeze: the same transcript step, but no transcript-free fallback ---------------------------


@pytest.fixture
def breeze_set_up(monkeypatch):
    """This machine has the Breeze environment: it becomes the default cloning engine."""
    monkeypatch.setattr("seiyuu.services.clone_transcript.breeze_ready", lambda _cfg: True)


def test_the_default_engine_is_breeze_only_where_it_is_set_up(monkeypatch):
    from seiyuu.services.clone_transcript import default_clone_engine

    assert default_clone_engine(object()) == "qwen3tts"  # the suite's baseline: not set up
    monkeypatch.setattr("seiyuu.services.clone_transcript.breeze_ready", lambda _cfg: True)
    assert default_clone_engine(object()) == "breeze"


def test_breeze_ready_needs_the_python_the_checkout_and_the_weights(tmp_path):
    from types import SimpleNamespace

    from seiyuu.engines.breeze_engine import breeze_ready

    python = tmp_path / "python.exe"
    repo, ckpt = tmp_path / "repo", tmp_path / "ckpt"
    cfg = SimpleNamespace(
        breeze_worker_python=python, breeze_repo_dir=repo, breeze_checkpoint_dir=ckpt
    )
    assert not breeze_ready(SimpleNamespace())  # nothing configured
    assert not breeze_ready(cfg)
    python.write_bytes(b"")
    (repo / "breeze_infer").mkdir(parents=True)
    (repo / "breeze_infer" / "templates.py").write_text("", encoding="utf-8")
    (repo / "models").mkdir()
    (repo / "models" / "fast_streaming.py").write_text("", encoding="utf-8")
    assert not breeze_ready(cfg)  # still no weights
    (ckpt / "audio_tokenizer").mkdir(parents=True)
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    (ckpt / "model.safetensors").write_bytes(b"")
    assert breeze_ready(cfg)
    python.unlink()
    assert not breeze_ready(cfg)  # the weights alone are not a working engine


def test_breeze_binds_the_transcript_under_its_own_key(tmp_path):
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"clip")
    settings = bind_transcript(ref, "ab" * 32, "Hello there.", engine="breeze")
    assert settings == {
        "breeze": {REF_TEXT_DIGEST_KEY: reference_text_digest("ab" * 32, "Hello there.")}
    }
    # the digest has no engine in it: the same clip and words bind the same value on Qwen
    assert settings["breeze"] == bind_transcript(ref, "ab" * 32, "Hello there.")["qwen3tts"]


@pytest.mark.parametrize(
    ("heard", "typed", "why"),
    [
        (("some words", 31.0), None, "the clip is 31 s, over 30 s"),
        (("some words", 45.0), "typed words", "the clip is 45 s"),  # too long even when typed
        (("   ", 8.0), None, "no speech was heard"),
    ],
)
def test_breeze_refuses_a_clip_it_has_no_transcript_for(tmp_path, heard, typed, why):
    from seiyuu.services.clone_transcript import CloneRefused

    with pytest.raises(CloneRefused, match=why) as refused:
        choose_transcript(
            tmp_path / "c.wav", provided=typed, transcribe=lambda _p: heard, engine="breeze"
        )
    assert "qwen3tts" in str(refused.value)  # says what to do instead
    # the same clip on Qwen is not refused: it clones from the sound alone
    assert (
        choose_transcript(tmp_path / "c.wav", provided=typed, transcribe=lambda _p: heard).text
        is None
    )


def test_breeze_takes_a_typed_transcript_when_whisper_fails(tmp_path):
    from seiyuu.services.clone_transcript import CloneRefused
    from seiyuu.validate import ValidationError

    def broken(_path):
        raise ValidationError("no model")

    chosen = choose_transcript(
        tmp_path / "c.wav", provided=" My  words. ", transcribe=broken, engine="breeze"
    )
    assert chosen.text == "My words."
    with pytest.raises(CloneRefused, match="couldn't transcribe"):
        choose_transcript(tmp_path / "c.wav", provided=None, transcribe=broken, engine="breeze")


def test_cli_clone_defaults_to_breeze_where_it_is_set_up(tmp_path, fake_whisper, breeze_set_up):
    result = _cli_clone(tmp_path)
    assert result.exit_code == 0, result.output
    assert (
        "(breeze)" in result.output and "cloned from the clip and its transcript" in result.output
    )
    meta = VoiceLibrary(tmp_path / "voices").load("elena_x")
    assert meta.engine == "breeze"
    text = "It was a dark and stormy night."
    assert meta.settings == {
        "breeze": {REF_TEXT_DIGEST_KEY: reference_text_digest(meta.consent.reference_sha256, text)}
    }


def test_cli_breeze_clone_of_a_long_clip_is_refused_and_saves_nothing(tmp_path, monkeypatch):
    class Long:
        def __init__(self, **kw):
            pass

        def transcribe_clip(self, _path):
            return "Plenty of words here.", 42.0

    monkeypatch.setattr("seiyuu.validate.Validator", Long)
    result = _cli_clone(tmp_path, "--engine", "breeze")
    assert result.exit_code != 0
    assert "the clip is 42 s" in result.output and "qwen3tts" in result.output
    assert not (tmp_path / "voices" / "elena_x").exists()  # no half-made voice


def test_api_clone_on_breeze_binds_under_breeze_and_refuses_a_long_clip(client):
    made = _post_clone(client, engine="breeze")
    assert made.status_code == 201, made.text
    body = made.json()
    assert body["engine"] == "breeze" and body["clone_note"].endswith("its transcript")
    assert list(body["settings"]) == ["breeze"]
    voices = client.app.state.settings.voices_dir
    assert (voices / "elena_x" / "reference.txt").is_file()

    client.app.state.aligner = _FakeAligner(text="Plenty of words here.", seconds=42.0)
    refused = _post_clone(client, engine="breeze", voice_id="too_long")
    assert refused.status_code == 422, refused.text
    assert "the clip is 42 s" in refused.json()["error"]["message"]
    assert not (voices / "too_long").exists()  # nothing saved for a voice that could not speak


def test_api_clone_default_follows_the_machine(client, breeze_set_up):
    assert client.get("/api/system").json()["default_clone_engine"] == "breeze"
    made = _post_clone(client)  # no engine named
    assert made.status_code == 201 and made.json()["engine"] == "breeze"


def test_api_system_reports_qwen_where_breeze_is_not_set_up(client):
    assert client.get("/api/system").json()["default_clone_engine"] == "qwen3tts"
