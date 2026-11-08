"""The Qwen3-TTS A/B scripts: voice design (take scoring, picking, staging) and the A/B arms.

No GPU, no SDK: a fake engine returns canned audio, a fake validator returns canned verdicts and
a fake GPU manager records acquires. The scripts are loaded from scripts/ by path.
"""

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from seiyuu.engines.audio import AudioFile
from seiyuu.engines.qwen3tts_engine import (
    REF_TEXT_DIGEST_KEY,
    Qwen3TTSEngine,
    reference_text_digest,
)
from seiyuu.settings import get_settings
from seiyuu.validate.models import ValidationResult
from seiyuu.voices import VoiceLibrary
from seiyuu.voices.blends import render_voice_args
from seiyuu.voices.library import sha256_file
from seiyuu.voices.models import ConsentAttestation, VoiceKind, VoiceMeta

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


design = _load("design_voices")


# -- design: pure pieces ----------------------------------------------------------------------


def _take(k, *, ok=True, score=0.95, wpm=175.0):
    return design.TakeScore(
        take=k, seed=k, path=f"t{k}.wav", text="x", seconds=1.0, words=3, wpm=wpm,
        ok=ok, score=score, transcript="x",
    )  # fmt: skip


def test_pick_prefers_passing_then_in_range_pace_then_match_then_nearest_pace():
    assert design.pick_best([_take(1, ok=False, score=1.0), _take(2, score=0.9)]).take == 2
    assert design.pick_best([_take(1, wpm=240.0, score=1.0), _take(2, wpm=150.0)]).take == 2
    assert design.pick_best([_take(1, score=0.91), _take(2, score=0.99)]).take == 2
    assert design.pick_best([_take(1, wpm=150.0), _take(2, wpm=180.0)]).take == 2
    assert design.pick_best([_take(1), _take(2)]).take == 1  # a full tie keeps the first take
    with pytest.raises(ValueError):
        design.pick_best([])


def test_take_seeds_are_stable_and_distinct():
    seeds = {design.take_seed("narrator", k) for k in (1, 2, 3)}
    assert len(seeds) == 3
    assert design.take_seed("narrator", 1) == design.take_seed("narrator", 1)
    assert design.take_seed("narrator", 1) != design.take_seed("staff_auto", 1)


def test_parse_picks():
    assert design.parse_picks(["narrator=2", "staff_auto=3"]) == {"narrator": 2, "staff_auto": 3}
    for bad in ("narrator", "narrator=0", "narrator=x"):
        with pytest.raises(ValueError):
            design.parse_picks([bad])


def test_load_specs_accepts_a_list_or_a_voices_object(tmp_path):
    row = {"voice_id": "narrator", "description": "calm", "sample_text": "Hello there."}
    path = tmp_path / "cast.json"
    path.write_text(json.dumps({"voices": [row]}), encoding="utf-8")
    assert design.load_specs(path)[0].name == "narrator"
    path.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        design.load_specs(path)
    path.write_text(json.dumps([{**row, "description": " "}]), encoding="utf-8")
    with pytest.raises(ValueError, match="description"):
        design.load_specs(path)


# -- design: the run loop ---------------------------------------------------------------------


class _FakeEngine:
    """Take k of every voice lasts ``durations[k]`` seconds (so pace is controllable)."""

    def __init__(self, durations):
        self.durations = durations
        self.calls = []

    def design_voice(self, text, description, *, seed):
        self.calls.append((text, description, seed))
        k = len([c for c in self.calls if c[1] == description])
        n = int(self.durations[k] * 24000)
        return AudioFile(samples=np.full(n, 0.1 * k, dtype=np.float32))


class _FakeValidator:
    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = 0

    def validate(self, path, text):
        self.calls += 1
        ok = Path(path).name not in self.failing
        return ValidationResult(
            ok=ok, score=0.97 if ok else 0.4, transcript=text, expected=text, reason=None
        )


class _FakeGpu:
    def __init__(self):
        self.acquired = []
        self.freed = 0

    @contextmanager
    def acquire(self, consumer, name):
        self.acquired.append(name)
        yield

    def free_all(self):
        self.freed += 1


SAMPLE = "one two three four five six seven eight nine ten"  # 10 words


def _run(tmp_path, engine, validator, gpu, **kw):
    spec = design.DesignSpec("narrator", "calm narrator", SAMPLE, "Narrator")
    return design.run(
        [spec],
        out=tmp_path,
        takes=3,
        engine=engine,
        validator=validator,
        gpu=gpu,
        normalize=lambda t: t,
        **kw,
    )


def test_run_designs_scores_picks_and_stages_a_consented_clone(tmp_path):
    # 10 words: take1 2 s = 300 wpm, take2 3.5 s ~ 171 wpm, take3 5 s = 120 wpm
    engine = _FakeEngine({1: 2.0, 2: 3.5, 3: 5.0})
    gpu = _FakeGpu()
    summary = _run(tmp_path, engine, _FakeValidator(), gpu)
    assert summary["narrator"]["chosen"] == 2 and summary["narrator"]["how"] == "auto"
    assert [c[2] for c in engine.calls] == [design.take_seed("narrator", k) for k in (1, 2, 3)]
    assert gpu.acquired == ["qwen3tts"] * 3 and gpu.freed == 1
    for k in (1, 2, 3):
        assert (tmp_path / "takes" / "narrator" / f"take{k}.wav").is_file()

    lib = VoiceLibrary(tmp_path / "voices")
    meta = lib.load("narrator")
    assert meta.kind is VoiceKind.CLONED and meta.engine == "qwen3tts"
    assert meta.source == "qwen_design" and meta.seed == design.take_seed("narrator", 2)
    assert meta.consent.statement == design.CONSENT_TEXT
    lib.verify_consent(meta)  # the attestation binds the staged clip's bytes
    ref = lib.reference_path("narrator")
    assert sha256_file(ref) == sha256_file(tmp_path / "takes" / "narrator" / "take2.wav")
    txt = ref.with_name("reference.txt")
    assert txt.read_text(encoding="utf-8").strip() == SAMPLE
    # the transcript is bound in the meta (and so in every segment's cache key)
    assert meta.settings["qwen3tts"][REF_TEXT_DIGEST_KEY] == reference_text_digest(
        sha256_file(ref), SAMPLE
    )
    scores = json.loads((tmp_path / "takes" / "narrator" / "scores.json").read_text("utf-8"))
    assert [t["take"] for t in scores["takes"]] == [1, 2, 3]
    assert "| narrator | 2 |" in (tmp_path / "design.md").read_text("utf-8")


def test_a_take_that_fails_the_speech_check_is_not_picked(tmp_path):
    engine = _FakeEngine({1: 3.5, 2: 3.4, 3: 5.0})
    summary = _run(tmp_path, engine, _FakeValidator(failing={"take1.wav"}), _FakeGpu())
    assert summary["narrator"]["chosen"] == 2


class _CloneTransport:
    """Stands in for the Qwen worker: records clone requests, writes a short wav."""

    def __init__(self):
        self.requests = []

    def request(self, message, *, timeout):
        self.requests.append(dict(message))
        if message["cmd"] == "synthesize":
            AudioFile(samples=np.full(2400, 0.1, dtype=np.float32)).save(Path(message["out_path"]))
        return {"ok": True}

    def close(self):
        pass


def _clone_ref_text(tmp_path, voice_id="narrator"):
    """What the real adapter sends as ref_text for a staged voice, rendered as the pipeline does
    (settings from render_voice_args(meta))."""
    meta = VoiceLibrary(tmp_path / "voices").load(voice_id)
    engine_voice, settings = render_voice_args(meta)
    transport = _CloneTransport()
    engine = Qwen3TTSEngine(
        voices_dir=tmp_path / "voices", models_dir=tmp_path, worker_python=tmp_path / "py",
        backend="fast", language="English", max_seq_len=2048, load_timeout=5.0,
        request_timeout=5.0, max_restarts=0, transport_factory=lambda: transport,
    )  # fmt: skip
    engine.synthesize("Hi.", engine_voice, {**settings, "seed": meta.seed})
    return transport.requests[-1]["ref_text"]


def test_staged_voices_clone_in_icl_only_with_a_clean_take(tmp_path):
    engine = _FakeEngine({1: 3.5, 2: 3.4, 3: 5.0})
    _run(tmp_path, engine, _FakeValidator(), _FakeGpu())
    assert _clone_ref_text(tmp_path) == SAMPLE  # the bound transcript reaches the worker

    # every take flagged: the best is still staged, but without a transcript (x-vector)
    flagged = _FakeValidator(failing={"take1.wav", "take2.wav", "take3.wav"})
    summary = _run(tmp_path / "f", _FakeEngine({1: 3.5, 2: 3.4, 3: 5.0}), flagged, _FakeGpu())
    assert summary["narrator"]["takes"][summary["narrator"]["chosen"] - 1]["ok"] is False
    ref = VoiceLibrary(tmp_path / "f" / "voices").reference_path("narrator")
    assert not ref.with_name("reference.txt").exists()
    assert VoiceLibrary(tmp_path / "f" / "voices").load("narrator").settings == {}
    assert _clone_ref_text(tmp_path / "f") is None

    # restaging a clean voice with a flagged --pick removes its old transcript too
    _run(tmp_path, _FakeEngine({}), _FakeValidator(), _FakeGpu())  # scores reused: all ok
    scores = tmp_path / "takes" / "narrator" / "scores.json"
    data = json.loads(scores.read_text("utf-8"))
    data["takes"][2]["ok"] = False
    scores.write_text(json.dumps(data), encoding="utf-8")
    _run(tmp_path, _FakeEngine({}), _FakeValidator(), _FakeGpu(), picks={"narrator": 3})
    assert _clone_ref_text(tmp_path) is None


def _run_with(tmp_path, engine, *, description="calm narrator", sample=SAMPLE, **kw):
    spec = design.DesignSpec("narrator", description, sample, "Narrator")
    return design.run(
        [spec], out=tmp_path, takes=3, engine=engine, validator=_FakeValidator(),
        gpu=_FakeGpu(), normalize=lambda t: t, **kw,
    )  # fmt: skip


def test_an_edited_sample_text_designs_new_takes(tmp_path):
    _run_with(tmp_path, _FakeEngine({1: 2.0, 2: 3.5, 3: 5.0}))
    take1 = tmp_path / "takes" / "narrator" / "take1.wav"
    before = take1.read_bytes()
    engine = _FakeEngine({1: 2.5, 2: 3.0, 3: 4.0})
    new_text = "a whole different sample sentence spoken by the narrator today"
    _run_with(tmp_path, engine, sample=new_text)
    assert [c[0] for c in engine.calls] == [new_text] * 3  # every take redesigned
    assert take1.read_bytes() != before
    ref = VoiceLibrary(tmp_path / "voices").reference_path("narrator")
    assert ref.with_name("reference.txt").read_text("utf-8").strip() == new_text
    assert _clone_ref_text(tmp_path) == new_text  # the clip speaks its transcript


def test_an_edited_description_designs_new_takes(tmp_path):
    _run_with(tmp_path, _FakeEngine({1: 2.0, 2: 3.5, 3: 5.0}), description="calm old man")
    engine = _FakeEngine({1: 2.0, 2: 3.5, 3: 5.0})
    _run_with(tmp_path, engine, description="bright young girl")
    assert [c[1] for c in engine.calls] == ["bright young girl"] * 3
    scores = json.loads((tmp_path / "takes" / "narrator" / "scores.json").read_text("utf-8"))
    assert scores["description"] == "bright young girl"
    # unchanged entry: nothing is redesigned
    again = _FakeEngine({})
    _run_with(tmp_path, again, description="bright young girl")
    assert again.calls == []


def test_pick_restages_from_existing_takes_without_the_gpu(tmp_path):
    _run(tmp_path, _FakeEngine({1: 2.0, 2: 3.5, 3: 5.0}), _FakeValidator(), _FakeGpu())
    engine, validator, gpu = _FakeEngine({}), _FakeValidator(), _FakeGpu()
    summary = _run(tmp_path, engine, validator, gpu, picks={"narrator": 3})
    assert engine.calls == [] and gpu.acquired == [] and gpu.freed == 0
    assert validator.calls == 0  # scores reused from scores.json
    assert (summary["narrator"]["chosen"], summary["narrator"]["how"]) == (3, "picked")
    lib = VoiceLibrary(tmp_path / "voices")
    meta = lib.load("narrator")
    lib.verify_consent(meta)  # fresh consent for the new clip
    assert meta.seed == design.take_seed("narrator", 3)
    with pytest.raises(ValueError, match="only 3 takes"):
        _run(tmp_path, engine, validator, gpu, picks={"narrator": 4})
    with pytest.raises(ValueError, match="not in the cast"):
        _run(tmp_path, engine, validator, gpu, picks={"ghost": 1})


def test_parse_shares():
    assert design.parse_shares(["new_employee=kim"]) == {"new_employee": "kim"}
    for bad in (["kim"], ["kim=kim"], ["a=b", "b=c"], ["=kim"]):
        with pytest.raises(ValueError):
            design.parse_shares(bad)


def test_share_stages_the_source_clip_under_a_second_id_without_designing_it(tmp_path):
    engine, gpu = _FakeEngine({1: 2.0, 2: 3.5, 3: 5.0}), _FakeGpu()
    specs = [
        design.DesignSpec("kim", "polite young man", SAMPLE, "Kim"),
        design.DesignSpec("new_employee", "polite young man", SAMPLE, "New Employee"),
    ]
    kw = dict(out=tmp_path, takes=3, validator=_FakeValidator(), gpu=gpu, normalize=lambda t: t)
    summary = design.run(specs, engine=engine, shares={"new_employee": "kim"}, **kw)
    assert len(engine.calls) == 3  # only kim was designed
    assert summary["new_employee"]["how"] == "same as kim"
    assert not (tmp_path / "takes" / "new_employee").exists()
    lib = VoiceLibrary(tmp_path / "voices")
    kim, new = lib.load("kim"), lib.load("new_employee")
    lib.verify_consent(new)
    assert new.name == "New Employee" and new.seed == kim.seed
    assert sha256_file(lib.reference_path("new_employee")) == sha256_file(lib.reference_path("kim"))
    assert "same as kim" in (tmp_path / "design.md").read_text("utf-8")

    # re-picking the source restages the share too, still without the GPU
    engine2 = _FakeEngine({})
    design.run(specs, engine=engine2, picks={"kim": 3}, shares={"new_employee": "kim"}, **kw)
    assert engine2.calls == []
    assert lib.load("new_employee").seed == design.take_seed("kim", 3)
    with pytest.raises(ValueError, match="has not been designed"):
        design.run(specs[1:], engine=engine2, shares={"new_employee": "ghost"}, **kw)


def test_a_share_follows_its_source_without_repeating_share(tmp_path):
    kim = design.DesignSpec("kim", "polite young man", SAMPLE, "Kim")
    kw = dict(out=tmp_path, takes=3, validator=_FakeValidator(), gpu=_FakeGpu(),
              normalize=lambda t: t)  # fmt: skip
    design.run([kim], engine=_FakeEngine({1: 2.0, 2: 3.5, 3: 5.0}), shares={"newb": "kim"}, **kw)
    lib = VoiceLibrary(tmp_path / "voices")
    assert lib.load("newb").seed == lib.load("kim").seed

    design.run([kim], engine=_FakeEngine({}), picks={"kim": 1}, **kw)  # no --share this time
    kim_meta, newb = lib.load("kim"), lib.load("newb")
    assert kim_meta.seed == newb.seed == design.take_seed("kim", 1)
    assert sha256_file(lib.reference_path("newb")) == sha256_file(lib.reference_path("kim"))
    lib.verify_consent(newb)
    summary = json.loads((tmp_path / "design.json").read_text("utf-8"))
    assert summary["newb"]["chosen"] == 1 and summary["newb"]["shared_from"] == "kim"
    assert "same as kim, take 1" in (tmp_path / "design.md").read_text("utf-8")

    # --unshare drops it; listed in the cast, it is then designed on its own
    newb_spec = design.DesignSpec("newb", "nervous new hire", SAMPLE, "Newb")
    engine = _FakeEngine({1: 2.0, 2: 3.5, 3: 5.0})
    design.run([kim, newb_spec], engine=engine, unshare={"newb"}, **kw)
    assert [c[1] for c in engine.calls] == ["nervous new hire"] * 3
    summary = json.loads((tmp_path / "design.json").read_text("utf-8"))
    assert "shared_from" not in summary["newb"]
    assert lib.load("newb").seed != lib.load("kim").seed


def test_staging_never_overwrites_a_voice_it_did_not_stage(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    ref = lib.reference_path("narrator")
    ref.parent.mkdir(parents=True)
    AudioFile(samples=np.full(2400, 0.1, dtype=np.float32)).save(ref)
    lib.save(
        VoiceMeta(
            voice_id="narrator",
            name="Real",
            kind=VoiceKind.CLONED,
            engine="chatterbox",
            reference_audio="reference.wav",
            consent_attested=True,
            consent=ConsentAttestation(attested_by="t", reference_sha256=sha256_file(ref)),
            source="user_upload",
        )  # fmt: skip
    )
    before = ref.read_bytes()
    engine, gpu = _FakeEngine({1: 2.0, 2: 3.5, 3: 5.0}), _FakeGpu()
    with pytest.raises(ValueError, match="not a designed voice"):
        _run(tmp_path, engine, _FakeValidator(), gpu)
    assert engine.calls == [] and gpu.acquired == []  # refused before any GPU work
    assert ref.read_bytes() == before and lib.load("narrator").source == "user_upload"


def _cast(tmp_path):
    path = tmp_path / "cast.json"
    rows = [
        {"voice_id": "narrator", "description": "calm", "sample_text": SAMPLE},
        {"voice_id": "kim", "description": "polite", "sample_text": SAMPLE},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_design_main_refuses_an_out_over_the_voice_library(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VOICES_DIR", str(tmp_path / "lib" / "voices"))
    get_settings.cache_clear()
    with pytest.raises(SystemExit) as exc:
        design.main([str(_cast(tmp_path)), "--out", str(tmp_path / "lib")])
    assert exc.value.code == 2 and "voice library" in capsys.readouterr().err


def test_design_main_checks_only_and_its_picks(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VOICES_DIR", str(tmp_path / "lib"))
    get_settings.cache_clear()
    cast, out = str(_cast(tmp_path)), str(tmp_path / "out")
    with pytest.raises(SystemExit) as exc:
        design.main([cast, "--out", out, "--only", "narator"])
    assert exc.value.code == 2 and "not in the cast file: ['narator']" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        design.main([cast, "--out", out, "--only", "narrator", "--pick", "kim=2"])
    assert exc.value.code == 2 and "--only leaves out: ['kim']" in capsys.readouterr().err
