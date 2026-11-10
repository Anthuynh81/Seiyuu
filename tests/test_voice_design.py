"""Voice design: describe a voice, design scored takes (Qwen3-TTS VoiceDesign), keep one.

No GPU and no whisper: the designer engine and the speech check are fakes. The GPU manager is
the real one (it only sees a fake engine), so the acquire/free discipline is exercised.
"""

import time
from contextlib import contextmanager

import numpy as np
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from fake_engine import FakeEngine
from seiyuu.api.main import create_app
from seiyuu.cli import main
from seiyuu.engines.audio import AudioFile
from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
from seiyuu.repository import JobState
from seiyuu.services import ServiceError
from seiyuu.services.voice_design import (
    CONSENT_TEXT,
    SOURCE,
    design_dir,
    keep_take,
    load_record,
    run_design,
    start_design,
    take_path,
    take_seed,
)
from seiyuu.validate.models import ValidationResult
from seiyuu.voices import VoiceLibrary
from test_api_m6b1 import make_settings

TEXT = "one two three four five six seven eight nine ten"  # 10 words


class Designer(FakeEngine):
    """design_voice speaks at the scripted pace (wpm) per call, in order."""

    engine_id = "qwen3tts"

    def __init__(self, paces=(175.0, 175.0, 175.0)) -> None:
        super().__init__()
        self.paces = list(paces)
        self.designs: list[tuple[str, str, int]] = []

    def design_voice(self, text, description, *, seed, settings=None):
        wpm = self.paces[len(self.designs) % len(self.paces)]
        self.designs.append((text, description, seed))
        seconds = len(text.split()) / wpm * 60.0
        return AudioFile(samples=np.zeros(int(seconds * 24_000), dtype=np.float32))


class Checker:
    """validate() returns the scripted pass/fail per call, in order."""

    def __init__(self, oks=(True, True, True)) -> None:
        self.oks = list(oks)
        self.calls = 0

    def validate(self, path, text):
        ok = self.oks[self.calls % len(self.oks)]
        self.calls += 1
        return ValidationResult(
            ok=ok,
            score=0.98 if ok else 0.4,
            transcript=text if ok else "one two",
            expected=text,
            reason=None if ok else "dropped words",
        )


class Gpu:
    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.freed = 0

    @contextmanager
    def acquire(self, engine, label):
        self.acquired.append(label)
        yield engine

    def free_all(self):
        self.freed += 1


def _start(tmp_path, takes=3):
    record = start_design(
        tmp_path, name="Janitor", description="gruff old janitor", text=TEXT, takes=takes
    )
    return record, design_dir(tmp_path, record.design_id)


# -- the service --------------------------------------------------------------------------------


def test_run_design_scores_every_take_and_recommends_the_best(tmp_path):
    record, directory = _start(tmp_path)
    engine, checker, gpu = (
        Designer(paces=(260.0, 175.0, 150.0)),
        Checker((True, True, False)),
        Gpu(),
    )
    messages: list[str] = []
    done = run_design(
        directory, engine=engine, validator=checker, gpu=gpu, progress=messages.append
    )

    assert [t.take for t in done.takes] == [1, 2, 3]
    assert [s for _t, _d, s in engine.designs] == [
        take_seed(record.design_id, k) for k in (1, 2, 3)
    ]
    assert all(d == "gruff old janitor" and t == TEXT for t, d, _s in engine.designs)
    assert [t.in_pace for t in done.takes] == [False, True, True]
    assert done.recommended == 2  # passes the speech check AND in pace beats a too-fast pass
    assert all(take_path(directory, k).is_file() for k in (1, 2, 3))
    assert load_record(directory).recommended == 2  # persisted
    assert gpu.freed == 1  # VoiceDesign is unloaded once, at the end
    assert "designing take 1 of 3" in messages and "checking take 3 of 3" in messages


def test_a_canceled_design_keeps_its_takes_and_a_rerun_makes_only_the_rest(tmp_path):
    _record, directory = _start(tmp_path)
    engine, gpu = Designer(), Gpu()
    calls = {"n": 0}

    def cancel_on_third():
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("canceled")

    with pytest.raises(RuntimeError, match="canceled"):
        run_design(
            directory, engine=engine, validator=Checker(), gpu=gpu, check_cancel=cancel_on_third
        )
    assert [t.take for t in load_record(directory).takes] == [1, 2]
    assert gpu.freed == 1  # freed even on the way out

    done = run_design(directory, engine=engine, validator=Checker(), gpu=Gpu())
    assert [t.take for t in done.takes] == [1, 2, 3]
    assert len(engine.designs) == 3  # takes 1-2 were not designed again


def test_keep_a_passing_take_makes_a_consented_qwen_voice_with_its_transcript(tmp_path):
    record, directory = _start(tmp_path)
    run_design(directory, engine=Designer(), validator=Checker(), gpu=Gpu())
    lib = VoiceLibrary(tmp_path / "voices")

    meta = keep_take(directory, 2, library=lib, name="Old Janitor")

    assert meta.kind.value == "cloned" and meta.engine == "qwen3tts" and meta.source == SOURCE
    assert meta.name == "Old Janitor" and meta.seed == take_seed(record.design_id, 2)
    assert meta.consent.statement == CONSENT_TEXT
    lib.verify_consent(lib.load(meta.voice_id))  # the attestation binds the published copy
    ref = lib.reference_path(meta.voice_id)
    assert ref.read_bytes() == take_path(directory, 2).read_bytes()
    assert ref.with_name("reference.txt").read_text(encoding="utf-8") == TEXT + "\n"
    digest = reference_text_digest(meta.consent.reference_sha256, TEXT)
    assert meta.settings == {"qwen3tts": {REF_TEXT_DIGEST_KEY: digest}}
    assert load_record(directory).kept == {2: meta.voice_id}


def test_keep_a_flagged_take_clones_from_the_sound_alone(tmp_path):
    _record, directory = _start(tmp_path, takes=1)
    run_design(directory, engine=Designer(), validator=Checker((False,)), gpu=Gpu())
    lib = VoiceLibrary(tmp_path / "voices")
    meta = keep_take(directory, 1, library=lib)
    assert meta.settings == {} and meta.name == "Janitor"
    assert not lib.reference_path(meta.voice_id).with_name("reference.txt").exists()


def test_a_kept_take_can_clone_on_breeze_while_qwen_designed_it(tmp_path):
    # The engine a kept voice clones on is independent of the engine that designed the take.
    _record, directory = _start(tmp_path)
    run_design(directory, engine=Designer(), validator=Checker(), gpu=Gpu())
    lib = VoiceLibrary(tmp_path / "voices")
    meta = keep_take(directory, 1, library=lib, engine="breeze")
    assert meta.engine == "breeze" and meta.source == SOURCE  # still a Qwen-designed voice
    digest = reference_text_digest(meta.consent.reference_sha256, TEXT)
    assert meta.settings == {"breeze": {REF_TEXT_DIGEST_KEY: digest}}  # where Breeze reads it
    ref = lib.reference_path(meta.voice_id)
    assert ref.with_name("reference.txt").read_text(encoding="utf-8") == TEXT + "\n"


def test_a_flagged_take_is_never_kept_as_a_breeze_voice(tmp_path):
    # Breeze cannot clone without a transcript, and a flagged take's text is not known to be
    # what it says: it stays a Qwen voice cloned from the sound alone, as before.
    _record, directory = _start(tmp_path, takes=1)
    run_design(directory, engine=Designer(), validator=Checker((False,)), gpu=Gpu())
    lib = VoiceLibrary(tmp_path / "voices")
    meta = keep_take(directory, 1, library=lib, engine="breeze")
    assert meta.engine == "qwen3tts" and meta.settings == {}
    assert not lib.reference_path(meta.voice_id).with_name("reference.txt").exists()


def test_keep_refuses_a_missing_take_and_an_existing_voice(tmp_path):
    _record, directory = _start(tmp_path, takes=1)
    run_design(directory, engine=Designer(), validator=Checker(), gpu=Gpu())
    lib = VoiceLibrary(tmp_path / "voices")
    with pytest.raises(ServiceError, match="no take 2"):
        keep_take(directory, 2, library=lib)
    keep_take(directory, 1, library=lib, voice_id="janitor")
    with pytest.raises(ServiceError, match="already exists"):
        keep_take(directory, 1, library=lib, voice_id="janitor")


def test_design_ids_are_validated(tmp_path):
    with pytest.raises(ServiceError, match="invalid design id"):
        design_dir(tmp_path, "../voices")


def test_a_design_needs_a_description_and_a_line(tmp_path):
    with pytest.raises(ServiceError, match="description"):
        start_design(tmp_path, name="x", description=" ", text=TEXT, takes=1)
    with pytest.raises(ServiceError, match="sample line"):
        start_design(tmp_path, name="x", description="calm", text="", takes=1)


# -- API ----------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    designer = Designer()
    monkeypatch.setattr("seiyuu.api.registry.get_engine", lambda engine_id, **kw: designer)
    monkeypatch.setattr("seiyuu.validate.Validator", lambda **kw: Checker())
    app = create_app(settings=make_settings(tmp_path))
    with TestClient(app) as c:
        c.app = app
        c.designer = designer
        yield c


def _wait_terminal(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.app.state.store.get(job_id)
        if job.is_terminal:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} not terminal: {client.app.state.store.get(job_id)}")


def test_api_design_flow_start_listen_keep(client):
    resp = client.post(
        "/api/designs", json={"name": "Janitor", "description": "gruff old janitor", "takes": 2}
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    design_id = body["design"]["design_id"]
    assert resp.headers["location"] == f"/api/designs/{design_id}"
    assert body["job"]["kind"] == "design" and body["job"]["book_id"] == f"design:{design_id}"
    assert body["design"]["takes"] == []  # the job makes them
    assert body["design"]["text"].startswith("I didn't expect")  # the default line

    job = _wait_terminal(client, body["job"]["job_id"])
    assert job.state is JobState.SUCCEEDED, job.error

    design = client.get(f"/api/designs/{design_id}").json()
    assert [t["take"] for t in design["takes"]] == [1, 2]
    assert design["recommended"] in (1, 2)
    url = design["takes"][0]["audio_url"]
    assert url == f"/api/designs/{design_id}/takes/1.wav"
    wav = client.get(url)
    assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"

    kept = client.post(f"/api/designs/{design_id}/keep", json={"take": 1, "name": "Mr. Kwon"})
    assert kept.status_code == 201, kept.text
    voice = kept.json()
    assert voice["engine"] == "qwen3tts" and voice["name"] == "Mr. Kwon"
    assert kept.headers["location"] == f"/api/voices/{voice['voice_id']}"
    listed = {v["voice_id"] for v in client.get("/api/voices").json()["voices"]}
    assert voice["voice_id"] in listed
    again = client.get(f"/api/designs/{design_id}").json()
    assert again["takes"][0]["kept_voice_id"] == voice["voice_id"]


def test_api_design_uses_the_typed_line_normalized_for_qwen(client):
    resp = client.post(
        "/api/designs",
        json={
            "name": "N",
            "description": "calm narrator",
            "sample_text": "It cost $5.",
            "takes": 1,
        },
    )
    assert resp.json()["design"]["text"] == "It cost five dollars."
    _wait_terminal(client, resp.json()["job"]["job_id"])
    assert client.designer.designs[0][0] == "It cost five dollars."


def test_api_design_validation_and_missing(client):
    assert client.post("/api/designs", json={"name": "x", "description": "ok"}).status_code == 422
    too_many = {"name": "x", "description": "calm", "takes": 9}
    assert client.post("/api/designs", json=too_many).status_code == 422
    assert client.get("/api/designs/000000000000").status_code == 404
    assert client.get("/api/designs/not-an-id").status_code == 404
    assert client.get("/api/designs/000000000000/takes/1.wav").status_code == 404
    keep = client.post("/api/designs/000000000000/keep", json={"take": 1})
    assert keep.status_code == 404


# -- CLI ----------------------------------------------------------------------------------------


def test_cli_design_keeps_the_recommended_take(tmp_path, monkeypatch):
    designer = Designer(paces=(260.0, 175.0, 175.0))
    monkeypatch.setattr("seiyuu.engines.get_engine", lambda engine_id, **kw: designer)
    monkeypatch.setattr("seiyuu.validate.Validator", lambda **kw: Checker())
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from seiyuu.settings import get_settings

    get_settings.cache_clear()
    voices = tmp_path / "voices"
    result = CliRunner().invoke(
        main,
        ["voice", "design", "Janitor", "--description", "gruff old janitor",
         "--voices-dir", str(voices)],
    )  # fmt: skip
    get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert "take 2: " in result.output and "(recommended)" in result.output
    assert "kept take 2 as voice" in result.output
    (meta,) = VoiceLibrary(voices).list_voices()
    assert meta.engine == "qwen3tts" and meta.name == "Janitor"


def test_cli_design_keep_none_only_lists(tmp_path, monkeypatch):
    monkeypatch.setattr("seiyuu.engines.get_engine", lambda engine_id, **kw: Designer())
    monkeypatch.setattr("seiyuu.validate.Validator", lambda **kw: Checker())
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from seiyuu.settings import get_settings

    get_settings.cache_clear()
    voices = tmp_path / "voices"
    result = CliRunner().invoke(
        main,
        ["voice", "design", "J", "--description", "calm", "--takes", "1", "--keep", "none",
         "--voices-dir", str(voices)],
    )  # fmt: skip
    get_settings.cache_clear()
    assert result.exit_code == 0, result.output
    assert not voices.exists() or VoiceLibrary(voices).list_voices() == []
