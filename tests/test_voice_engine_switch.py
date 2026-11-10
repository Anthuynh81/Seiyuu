"""Moving a cloned voice to another local cloning engine, in place (decision D2).

No model and no whisper: the transcriber is a function, and "is the engine installed" is a seam.
"""

import hashlib
import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from seiyuu.cli import main
from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
from seiyuu.render.cache import SegmentKey
from seiyuu.repository import JobKind
from seiyuu.services import ServiceError, save_assignment
from seiyuu.services.voice_engine import (
    EngineNotInstalled,
    engine_installed,
    switch_engine,
    switchable_engines,
)
from seiyuu.voices import VoiceAssignment, VoiceKind, VoiceLibrary, VoiceMeta
from seiyuu.voices.blends import render_voice_args
from seiyuu.voices.models import ConsentAttestation
from test_api_m6b1 import make_settings

TEXT = "It was a dark and stormy night."
CLIP = b"RIFF-fake-reference-clip"
SHA = hashlib.sha256(CLIP).hexdigest()
DIGEST = reference_text_digest(SHA, TEXT)


def _clone(lib: VoiceLibrary, voice_id="kim_v", engine="qwen3tts", *, bound=True, extra=None):
    """A cloned voice as the clone flow leaves it: clip, consent, and (bound) its transcript."""
    ref = lib.reference_path(voice_id)
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(CLIP)
    settings: dict = {}
    if bound:
        ref.with_name("reference.txt").write_text(TEXT + "\n", encoding="utf-8")
        settings = {engine: {REF_TEXT_DIGEST_KEY: DIGEST, **(extra or {})}}
    lib.save(
        VoiceMeta(
            voice_id=voice_id,
            name="Kim",
            kind=VoiceKind.CLONED,
            engine=engine,
            reference_audio="reference.wav",
            consent_attested=True,
            consent=ConsentAttestation(attested_by="Ann", reference_sha256=SHA),
            seed=7,
            tags=["designed"],
            settings=settings,
        )
    )
    return lib.load(voice_id)


def _switch(lib, tmp_path, voice_id, target, **kw):
    kw.setdefault("installed", lambda _cfg, _engine: True)
    return switch_engine(lib, voice_id, target, cfg=object(), output_dir=tmp_path / "output", **kw)


def _heard(text=TEXT, seconds=8.0):
    return lambda _path: (text, seconds)


def _key(meta: VoiceMeta) -> SegmentKey:
    _voice, settings = render_voice_args(meta)
    return SegmentKey.build(
        engine=meta.engine,
        engine_model_version="m1",
        voice_id=meta.voice_id,
        settings=settings,
        seed=meta.seed,
        normalized_text="Hello.",
    )


def test_local_cloning_engines_are_the_switchable_ones():
    assert switchable_engines() == ["breeze", "chatterbox", "indextts2", "qwen3tts"]


def test_qwen_to_breeze_copies_the_binding_and_changes_nothing_else(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    before = _clone(lib, extra={"temperature": 0.7})
    (lib.dir_for("kim_v") / "audition.wav").write_bytes(b"old engine sample")
    book = VoiceAssignment(book_id="ghost", narrator_voice_id="kim_v", assignments={"kim": "kim_v"})
    save_assignment(book, tmp_path / "output")

    done = _switch(lib, tmp_path, "kim_v", "breeze")
    after = lib.load("kim_v")
    assert done.previous_engine == "qwen3tts" and after.engine == "breeze"
    # copied, not moved, and only the binding: Qwen's sampler tunables stay Qwen's
    assert after.settings == {
        "qwen3tts": {REF_TEXT_DIGEST_KEY: DIGEST, "temperature": 0.7},
        "breeze": {REF_TEXT_DIGEST_KEY: DIGEST},
    }
    assert after.engine_settings() == {REF_TEXT_DIGEST_KEY: DIGEST}  # what Breeze reads
    same = ("voice_id", "name", "kind", "seed", "consent", "tags", "reference_audio", "source")
    assert all(getattr(after, f) == getattr(before, f) for f in same)
    assert lib.reference_path("kim_v").read_bytes() == CLIP
    assert lib.reference_path("kim_v").with_name("reference.txt").read_text("utf-8") == TEXT + "\n"
    lib.verify_consent(after)  # the consent record still binds the same bytes
    assert not (lib.dir_for("kim_v") / "audition.wav").exists()  # the old engine's sample
    assert done.books == ["ghost"] and "ghost" in done.note and "breeze" in done.note


def test_switching_back_builds_the_old_cache_keys_again(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    original = _key(_clone(lib))
    _switch(lib, tmp_path, "kim_v", "breeze")
    on_breeze = _key(lib.load("kim_v"))
    assert on_breeze.key_hash != original.key_hash  # the engine is in the key: no stale reuse
    _switch(lib, tmp_path, "kim_v", "qwen3tts")
    assert _key(lib.load("kim_v")).key_hash == original.key_hash  # old audio hits again


def test_a_breeze_voice_moving_to_qwen_takes_its_transcript_along(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib, engine="breeze")
    _switch(lib, tmp_path, "kim_v", "qwen3tts")
    after = lib.load("kim_v")
    assert after.engine == "qwen3tts"
    assert after.settings["qwen3tts"] == {REF_TEXT_DIGEST_KEY: DIGEST}  # Qwen's closer mode


def test_a_voice_with_no_transcript_is_transcribed_for_breeze(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib, engine="chatterbox", bound=False)
    done = _switch(lib, tmp_path, "kim_v", "breeze", transcribe=_heard("Call me Kim."))
    after = lib.load("kim_v")
    path = lib.reference_path("kim_v").with_name("reference.txt")
    assert path.read_text(encoding="utf-8") == "Call me Kim.\n"
    digest = reference_text_digest(SHA, "Call me Kim.")
    # bound for Breeze only; the engine it left keeps an (empty) block marking it was there
    assert after.settings == {"chatterbox": {}, "breeze": {REF_TEXT_DIGEST_KEY: digest}}
    assert "transcribed" in done.note and done.books == []


def test_a_sound_alone_qwen_voice_comes_back_to_exactly_that(tmp_path):
    # A Qwen clone that bound no transcript (x-vector mode) visits Breeze, which needs one. On
    # the way back Qwen must not pick the new transcript up: that would change its cache keys
    # and strand every line already recorded on Qwen.
    lib = VoiceLibrary(tmp_path / "voices")
    original = _key(_clone(lib, bound=False))
    assert lib.load("kim_v").engine_settings() == {}
    _switch(lib, tmp_path, "kim_v", "breeze", transcribe=_heard("Call me Kim."))
    assert REF_TEXT_DIGEST_KEY in lib.load("kim_v").engine_settings()  # Breeze has its binding
    _switch(lib, tmp_path, "kim_v", "qwen3tts")
    back = lib.load("kim_v")
    assert back.engine_settings() == {} and _key(back).key_hash == original.key_hash
    # and Breeze again reuses the transcript it was given: no second whisper pass
    _switch(lib, tmp_path, "kim_v", "breeze")
    assert REF_TEXT_DIGEST_KEY in lib.load("kim_v").engine_settings()


def test_a_voice_new_to_qwen_gets_the_closer_transcript_mode(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib, engine="chatterbox", bound=False)
    _switch(lib, tmp_path, "kim_v", "breeze", transcribe=_heard("Call me Kim."))
    _switch(lib, tmp_path, "kim_v", "qwen3tts")  # never on Qwen before: nothing to preserve
    assert REF_TEXT_DIGEST_KEY in lib.load("kim_v").engine_settings()


def test_an_unreadable_casting_file_stops_the_switch_before_anything_changes(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    before = _clone(lib)
    (lib.dir_for("kim_v") / "audition.wav").write_bytes(b"sample")
    broken = tmp_path / "output" / "some-book"
    broken.mkdir(parents=True)
    (broken / "assignments.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ServiceError, match="cannot verify voice references"):
        _switch(lib, tmp_path, "kim_v", "breeze")
    assert lib.load("kim_v") == before  # not half-switched
    assert (lib.dir_for("kim_v") / "audition.wav").exists()


@pytest.mark.parametrize("heard", [("words", 45.0), ("  ", 8.0)])
def test_a_voice_breeze_cannot_clone_is_left_exactly_as_it_was(tmp_path, heard):
    lib = VoiceLibrary(tmp_path / "voices")
    before = _clone(lib, engine="chatterbox", bound=False)
    with pytest.raises(ServiceError, match="cannot move to breeze"):
        _switch(lib, tmp_path, "kim_v", "breeze", transcribe=lambda _p: heard)
    assert lib.load("kim_v") == before
    assert not lib.reference_path("kim_v").with_name("reference.txt").exists()


def test_engines_that_take_no_transcript_only_change_the_engine(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib, engine="breeze")
    _switch(lib, tmp_path, "kim_v", "chatterbox")
    after = lib.load("kim_v")
    assert after.engine == "chatterbox" and after.settings == {
        "breeze": {REF_TEXT_DIGEST_KEY: DIGEST}
    }  # nothing is written for an engine that takes no transcript
    assert after.engine_settings() == {}


def test_refusals(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib)
    lib.save(
        VoiceMeta(voice_id="p1", name="P", kind=VoiceKind.PRESET, engine="kokoro", preset_id="af")
    )
    with pytest.raises(ServiceError, match="only a cloned voice"):
        _switch(lib, tmp_path, "p1", "breeze")
    with pytest.raises(ServiceError, match="not a local cloning engine"):
        _switch(lib, tmp_path, "kim_v", "elevenlabs")
    with pytest.raises(ServiceError, match="not a local cloning engine"):
        _switch(lib, tmp_path, "kim_v", "kokoro")
    with pytest.raises(ServiceError, match="already uses qwen3tts"):
        _switch(lib, tmp_path, "kim_v", "qwen3tts")
    with pytest.raises(ServiceError, match="nope"):
        _switch(lib, tmp_path, "nope", "breeze")
    with pytest.raises(EngineNotInstalled, match="not set up on this machine"):
        _switch(lib, tmp_path, "kim_v", "breeze", installed=lambda _cfg, _engine: False)
    assert lib.load("kim_v").engine == "qwen3tts"  # none of them changed anything


def test_a_transcript_that_no_longer_matches_its_binding_stops_the_switch(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib)
    lib.reference_path("kim_v").with_name("reference.txt").write_text("Edited.\n", "utf-8")
    with pytest.raises(ServiceError, match="no longer matches"):
        _switch(lib, tmp_path, "kim_v", "breeze", transcribe=_heard())
    assert lib.load("kim_v").engine == "qwen3tts"


def test_a_replaced_clip_fails_the_consent_check_before_anything_changes(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib)
    lib.reference_path("kim_v").write_bytes(b"someone else")
    with pytest.raises(ServiceError):
        _switch(lib, tmp_path, "kim_v", "breeze")
    assert lib.load("kim_v").engine == "qwen3tts"


def test_engine_installed_reads_the_given_settings():
    from types import SimpleNamespace

    assert engine_installed(SimpleNamespace(), "breeze") is False
    assert engine_installed(SimpleNamespace(), "qwen3tts") is False
    assert engine_installed(SimpleNamespace(), "chatterbox") is None  # cannot be told cheaply


# -- API ----------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("seiyuu.services.voice_engine.engine_installed", lambda _c, _e: True)
    app = create_app(settings=make_settings(tmp_path))
    with TestClient(app) as c:
        c.app = app
        _clone(VoiceLibrary(tmp_path / "voices"))
        yield c


def test_api_switches_and_reports_the_books_to_record_again(client, tmp_path):
    book = VoiceAssignment(book_id="ghost", narrator_voice_id="kim_v", assignments={})
    save_assignment(book, tmp_path / "output")
    resp = client.post("/api/voices/kim_v/engine", json={"engine": "breeze"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["engine"], body["previous_engine"]) == ("breeze", "qwen3tts")
    assert body["books"] == ["ghost"] and "breeze" in body["note"]
    assert body["settings"]["breeze"] == {REF_TEXT_DIGEST_KEY: DIGEST}
    assert client.get("/api/voices/kim_v").json()["engine"] == "breeze"


def test_api_refusals(client, monkeypatch):
    url = "/api/voices/kim_v/engine"
    assert client.post("/api/voices/nope/engine", json={"engine": "breeze"}).status_code == 404
    assert client.post(url, json={"engine": "kokoro"}).status_code == 422
    assert client.post(url, json={"engine": "qwen3tts"}).status_code == 422  # already on it
    assert client.post(url, json={"engine": "breeze", "seed": 1}).status_code == 422
    assert client.post(url, json={}).status_code == 422

    job = client.app.state.store.create(book_id="ghost", kind=JobKind.RENDER, params={})
    busy = client.post(url, json={"engine": "breeze"})
    assert busy.status_code == 409 and busy.json()["error"]["code"] == "render_active"
    client.app.state.store.request_cancel(job.job_id)

    monkeypatch.setattr("seiyuu.services.voice_engine.engine_installed", lambda _c, _e: False)
    missing = client.post(url, json={"engine": "breeze"})
    assert missing.status_code == 503 and missing.json()["error"]["code"] == "not_ready"
    assert client.get("/api/voices/kim_v").json()["engine"] == "qwen3tts"


def test_api_transcribes_on_the_shared_cpu_whisper_when_breeze_needs_it(client, tmp_path):
    class Aligner:
        calls = 0

        def transcribe_clip(self, _path):
            Aligner.calls += 1
            return "Call me Kim.", 8.0

    client.app.state.aligner = Aligner()
    _clone(VoiceLibrary(tmp_path / "voices"), voice_id="old_v", engine="chatterbox", bound=False)
    resp = client.post("/api/voices/old_v/engine", json={"engine": "breeze"})
    assert resp.status_code == 200, resp.text
    assert Aligner.calls == 1 and "transcribed" in resp.json()["note"]
    # a voice that already binds a transcript never reaches whisper
    assert client.post("/api/voices/kim_v/engine", json={"engine": "breeze"}).status_code == 200
    assert Aligner.calls == 1


# -- CLI ----------------------------------------------------------------------------------------


def _cli(tmp_path, *args):
    return CliRunner().invoke(
        main,
        ["voice", "use-engine", *args, "--voices-dir", str(tmp_path / "voices"),
         "--output-dir", str(tmp_path / "output")],
    )  # fmt: skip


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr("seiyuu.services.voice_engine.engine_installed", lambda _c, _e: True)


def test_cli_switches_named_voices(tmp_path, installed):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib)
    _clone(lib, voice_id="alice_v")
    result = _cli(tmp_path, "breeze", "kim_v", "alice_v")
    assert result.exit_code == 0, result.output
    assert "kim_v: qwen3tts -> breeze" in result.output
    assert "2 voice(s) now clone on breeze" in result.output
    assert {lib.load(v).engine for v in ("kim_v", "alice_v")} == {"breeze"}


def test_cli_switches_every_cloned_voice_of_a_book(tmp_path, installed):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib)
    _clone(lib, voice_id="alice_v")
    _clone(lib, voice_id="elsewhere_v")  # not in this book: untouched
    lib.save(
        VoiceMeta(voice_id="p1", name="P", kind=VoiceKind.PRESET, engine="kokoro", preset_id="af")
    )
    cast = VoiceAssignment(
        book_id="ghost",
        narrator_voice_id="kim_v",
        assignments={"kim": "kim_v", "alice": "alice_v", "bob": "p1"},
    )
    save_assignment(cast, tmp_path / "output")
    result = _cli(tmp_path, "breeze", "--book", "ghost")
    assert result.exit_code == 0, result.output
    assert "2 voice(s) now clone on breeze" in result.output
    assert "record again to hear them: ghost" in result.output
    engines = {v: lib.load(v).engine for v in ("kim_v", "alice_v", "elsewhere_v", "p1")}
    assert engines == {
        "kim_v": "breeze", "alice_v": "breeze", "elsewhere_v": "qwen3tts", "p1": "kokoro"
    }  # fmt: skip
    again = _cli(tmp_path, "breeze", "--book", "ghost")
    assert again.exit_code == 0 and "already uses breeze" in again.output


def test_cli_reports_what_it_could_not_switch_and_fails(tmp_path, installed):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib)
    _clone(lib, voice_id="tampered_v")
    lib.reference_path("tampered_v").with_name("reference.txt").write_text("Edited.\n", "utf-8")
    result = _cli(tmp_path, "breeze", "kim_v", "tampered_v")
    assert result.exit_code != 0
    assert "kim_v: qwen3tts -> breeze" in result.output  # the good one still went through
    assert "NOT switched: tampered_v" in result.output
    assert lib.load("tampered_v").engine == "qwen3tts"


def test_cli_argument_errors(tmp_path, installed):
    _clone(VoiceLibrary(tmp_path / "voices"))
    assert "not a local cloning engine" in _cli(tmp_path, "kokoro", "kim_v").output
    assert "not both" in _cli(tmp_path, "breeze").output
    assert "not both" in _cli(tmp_path, "breeze", "kim_v", "--book", "ghost").output
    missing = _cli(tmp_path, "breeze", "--book", "ghost")
    assert missing.exit_code != 0 and "no saved casting" in missing.output


def test_meta_json_keeps_its_shape(tmp_path):
    # no new field: a switched voice is still a plain VoiceMeta any older build can read
    lib = VoiceLibrary(tmp_path / "voices")
    before = set(json.loads(lib.meta_path(_clone(lib).voice_id).read_text("utf-8")))
    _switch(lib, tmp_path, "kim_v", "breeze")
    assert set(json.loads(lib.meta_path("kim_v").read_text("utf-8"))) == before


def test_cli_book_switch_leaves_a_cloud_clone_alone_without_failing(tmp_path, installed):
    lib = VoiceLibrary(tmp_path / "voices")
    _clone(lib)
    _clone(lib, voice_id="cloud_v", engine="elevenlabs", bound=False)
    cast = VoiceAssignment(
        book_id="ghost", narrator_voice_id="kim_v", assignments={"kim": "kim_v", "eve": "cloud_v"}
    )
    save_assignment(cast, tmp_path / "output")
    result = _cli(tmp_path, "breeze", "--book", "ghost")
    assert result.exit_code == 0, result.output
    assert "cloud_v: left on elevenlabs" in result.output
    assert "1 voice(s) now clone on breeze" in result.output
    assert lib.load("cloud_v").engine == "elevenlabs"
    again = _cli(tmp_path, "breeze", "--book", "ghost")  # nothing left to do: still not a failure
    assert again.exit_code == 0 and "already uses breeze" in again.output


def test_an_audition_that_finishes_after_a_switch_is_not_published(client, tmp_path, monkeypatch):
    """The switch waits for renders, not auditions. A sample still being spoken on the old
    engine when the voice moves must not land on the card as the new engine's sample."""
    from fake_engine import FakeEngine

    lib = VoiceLibrary(tmp_path / "voices")

    class SwitchesMidLine(FakeEngine):
        def _synthesize_native(self, text, voice, settings):
            if lib.load("kim_v").engine == "qwen3tts":  # the user clicks "use Breeze" meanwhile
                switch_engine(lib, "kim_v", "breeze", cfg=object(), output_dir=tmp_path / "output")
            return super()._synthesize_native(text, voice, settings)

    engine = SwitchesMidLine()
    monkeypatch.setattr("seiyuu.api.registry.get_engine", lambda eid, **kw: engine)
    monkeypatch.setattr("seiyuu.api.routes.voices.get_engine_class", lambda eid: FakeEngine)
    monkeypatch.setattr("seiyuu.api.routes.voices.weights_cached", lambda eid: True)

    stale = client.post("/api/voices/kim_v/audition", json={})
    assert stale.status_code == 409, stale.text
    assert "audition it again" in stale.json()["error"]["message"]
    voice_dir = lib.dir_for("kim_v")
    assert not (voice_dir / "audition.wav").exists()
    assert not (voice_dir / "audition.part.wav").exists()  # the dropped take leaves nothing

    fresh = client.post("/api/voices/kim_v/audition", json={})  # now on Breeze: published
    assert fresh.status_code == 200, fresh.text
    assert (voice_dir / "audition.wav").exists()
