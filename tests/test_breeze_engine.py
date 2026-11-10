"""Breeze TTS 2 adapter tests — CPU only, worker faked (no subprocess, no Breeze repo, no GPU).

A fake transport FACTORY stands in for the worker subprocess, as in the Qwen3-TTS and IndexTTS-2
tests: it records requests and writes the handoff WAV, so the readback path runs end to end.
"""

import hashlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from seiyuu.api.registry import ENGINE_FACTS, EngineRegistry, weights_cached
from seiyuu.engines import get_engine_class, list_engine_ids, voices_dir_kwargs
from seiyuu.engines.base import SynthesisError
from seiyuu.engines.breeze_engine import (
    BREEZE_FAST_CONFIGS,
    BREEZE_SAMPLE_RATE,
    BreezeEngine,
    breeze_config,
    checkpoint_fingerprint,
    checkpoint_present,
    repo_present,
    worker_env,
)
from seiyuu.engines.breeze_worker import (
    GEN_RECIPE,
    PIECE_GAP_SECONDS,
    PROTOCOL_VERSION,
    SPLIT_OVER_CHARS,
)
from seiyuu.engines.indextts2_engine import SubprocessTransport, WorkerError
from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
from seiyuu.gpu import GpuResourceManager
from seiyuu.normalize import normalize_text, profile_for
from seiyuu.render.cache import SegmentKey
from seiyuu.settings import Settings, get_settings

TRANSCRIPT = "I keep a small notebook."


def _write_wav(path: str, *, seconds: float = 0.2, sr: int = BREEZE_SAMPLE_RATE) -> None:
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="PCM_16")


class FakeTransport:
    def __init__(self, behavior):
        self.requests: list[dict] = []
        self.timeouts: list[float] = []
        self.closed = False
        self._behavior = behavior

    def request(self, message, *, timeout):
        self.requests.append(dict(message))
        self.timeouts.append(timeout)
        return self._behavior(message)

    def close(self):
        self.closed = True


def _happy(message):
    if message["cmd"] == "load":
        return {"ok": True, "fast": "decode", "warmup_seconds": 0.0}
    if message["cmd"] == "synthesize":
        _write_wav(message["out_path"])
        return {"ok": True, "path": message["out_path"], "sample_rate": 24000, "hit_cap": False}
    return {"ok": False, "error": "unexpected"}


def _checkpoint(tmp_path):
    d = tmp_path / "ckpt"
    (d / "audio_tokenizer").mkdir(parents=True)
    (d / "config.json").write_text('{"model_type": "breeze"}', encoding="utf-8")
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 32)
    (d / "audio_tokenizer" / "model.safetensors").write_bytes(b"\0" * 16)
    return d


def _repo(tmp_path):
    d = tmp_path / "repo"
    (d / "breeze_infer").mkdir(parents=True)
    (d / "models").mkdir()
    (d / "breeze_infer" / "templates.py").write_text("", encoding="utf-8")
    (d / "models" / "fast_streaming.py").write_text("", encoding="utf-8")
    return d


def _voice(tmp_path, voice_id="hero_1a2b", *, transcript=TRANSCRIPT):
    d = tmp_path / "voices" / voice_id
    d.mkdir(parents=True)
    _write_wav(str(d / "reference.wav"), seconds=0.5)
    if transcript is not None:
        (d / "reference.txt").write_text(transcript + "\n", encoding="utf-8")
    return tmp_path / "voices"


def _bound(tmp_path, text=TRANSCRIPT, voice_id="hero_1a2b", **extra):
    """The settings a voice whose meta binds ``text`` as its reference transcript renders with."""
    ref = tmp_path / "voices" / voice_id / "reference.wav"
    sha = hashlib.sha256(ref.read_bytes()).hexdigest()
    return {"seed": 1, REF_TEXT_DIGEST_KEY: reference_text_digest(sha, text), **extra}


def _engine(tmp_path, factory=None, **kw):
    params = dict(
        voices_dir=tmp_path / "voices",
        checkpoint_dir=tmp_path / "ckpt",
        repo_dir=tmp_path / "repo",
        worker_python=tmp_path / "python.exe",
        fast="decode",
        load_timeout=50.0,
        request_timeout=5.0,
        max_restarts=1,
        transport_factory=factory,
    )
    params.update(kw)
    return BreezeEngine(**params)


# -- catalog / registration -------------------------------------------------------------------


def test_registered_as_a_local_cloning_engine_that_needs_validation(tmp_path):
    assert "breeze" in list_engine_ids()
    cls = get_engine_class("breeze")
    assert cls is BreezeEngine
    assert cls.requires_validation and cls.clones_from_library and cls.uses_gpu
    assert voices_dir_kwargs("breeze", tmp_path) == {"voices_dir": tmp_path}
    assert ENGINE_FACTS["breeze"] == {"paid": False, "supports_cloning": True}
    engine = _engine(tmp_path)
    assert engine.native_sample_rate == 24000 and engine.list_voices() == []
    assert engine.cost_estimate("free and local") == 0.0


def test_text_profile_is_explicit_and_folds_dashes_like_the_default():
    assert profile_for("breeze") == "breeze"
    line = "Wait — I… no."
    assert normalize_text(line, profile="breeze") == normalize_text(line, profile="default")
    # a parenthesized sound is something Breeze performs: the profile leaves it in the text
    assert "(sigh)" in normalize_text("(sigh) Fine.", profile="breeze")


def test_text_profile_drops_square_brackets_and_keeps_their_words():
    # Square brackets are Breeze's tag syntax: it does not read what is inside them. A real
    # render spoke "[Cover]" as "Four" and dropped "[Roulette Event]" from the line it opens.
    def spoken(text: str) -> str:
        return normalize_text(text, profile="breeze")

    assert spoken("[Cover]") == "Cover"
    assert spoken("[Roulette Event] : Create your own character.") == (
        "Roulette Event : Create your own character."
    )
    assert spoken("He said [quietly] that (sigh) it was over.") == (
        "He said quietly that (sigh) it was over."
    )
    assert spoken("[Welcome, explorer.][Stage 2]") == "Welcome, explorer. Stage two"
    # the words are the same as every other engine's, so whisper scores against the same text
    line = "[1st Place] : A <Real> Merch Box"
    words = lambda t: t.replace("[", " ").replace("]", " ").split()  # noqa: E731
    assert spoken(line).split() == words(normalize_text(line, profile="qwen3tts"))
    # only Breeze: the other engines read bracketed text and keep their cache keys
    for profile in ("default", "kokoro", "chatterbox", "indextts2", "qwen3tts"):
        assert "[Cover]" in normalize_text("[Cover] page", profile=profile)


def test_config_defaults_and_settings():
    assert breeze_config(SimpleNamespace()) == {
        "worker_python": None,
        "repo_dir": None,
        "checkpoint_dir": None,
        "fast": "decode",
    }
    cfg = Settings(_env_file=None)
    assert cfg.breeze_fast == "decode" and cfg.breeze_worker_max_restarts == 1
    assert cfg.breeze_worker_python is None and cfg.breeze_checkpoint_dir is None
    assert BREEZE_FAST_CONFIGS == ("decode", "off")
    with pytest.raises(ValueError):
        Settings(_env_file=None, breeze_fast="turbo")


def test_registry_constructs_from_injected_settings(tmp_path):
    settings = Settings(
        _env_file=None,
        voices_dir=tmp_path / "voices",
        data_dir=tmp_path / "data",
        breeze_checkpoint_dir=tmp_path / "ckpt",
        breeze_repo_dir=tmp_path / "repo",
        breeze_worker_python=tmp_path / "py.exe",
        breeze_fast="off",
        breeze_worker_request_timeout=12.0,
    )
    engine = EngineRegistry(settings, gpu_manager=GpuResourceManager()).get("breeze")
    assert engine._voices_dir == settings.voices_dir  # consent invariant: same voices root
    assert engine._checkpoint_dir == tmp_path / "ckpt" and engine._repo_dir == tmp_path / "repo"
    assert engine._fast == "off" and engine._request_timeout == 12.0


def test_adapter_falls_back_to_global_settings(monkeypatch, tmp_path):
    # The cost estimate builds the adapter with voices_dir only: everything else must resolve
    # from the same settings the registry injects, or estimate and render keys would diverge.
    ckpt = _checkpoint(tmp_path)
    monkeypatch.setenv("BREEZE_CHECKPOINT_DIR", str(ckpt))
    monkeypatch.setenv("BREEZE_FAST", "off")
    get_settings.cache_clear()
    try:
        estimate = BreezeEngine(voices_dir=tmp_path / "voices")
        render = EngineRegistry(get_settings(), gpu_manager=GpuResourceManager()).get("breeze")
        assert estimate.model_version == render.model_version
        assert estimate.model_version.endswith(f"-bf16-off-gen{GEN_RECIPE}")
    finally:
        get_settings.cache_clear()


def test_weights_cached_needs_the_repo_and_the_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("BREEZE_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("BREEZE_REPO_DIR", str(tmp_path / "repo"))
    get_settings.cache_clear()
    try:
        assert weights_cached("breeze") is False
        _checkpoint(tmp_path)
        assert weights_cached("breeze") is False  # weights alone can't run: no code to load them
        _repo(tmp_path)
        assert weights_cached("breeze") is True
    finally:
        get_settings.cache_clear()


# -- model_version ----------------------------------------------------------------------------


def test_model_version_fingerprints_the_checkpoint_and_names_the_fast_config(tmp_path):
    ckpt = _checkpoint(tmp_path)
    fp = checkpoint_fingerprint(ckpt)
    assert fp.startswith("breeze-") and len(fp) == len("breeze-") + 12
    assert _engine(tmp_path).model_version == f"{fp}-bf16-decode-gen{GEN_RECIPE}"
    # eager execution is not bit-exact with the compiled path: never share cache entries
    assert _engine(tmp_path, fast="off").model_version == f"{fp}-bf16-off-gen{GEN_RECIPE}"
    assert checkpoint_present(ckpt) and not checkpoint_present(tmp_path / "nope")
    assert not checkpoint_present(None) and not repo_present(None)


def test_fingerprint_ignores_hub_bookkeeping_but_not_configs(tmp_path):
    ckpt = _checkpoint(tmp_path)
    before = checkpoint_fingerprint(ckpt)
    cache = ckpt / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "model-00001-of-00002.safetensors.metadata").write_text("etag 2026", encoding="utf-8")
    assert checkpoint_fingerprint(ckpt) == before  # re-running a download changes nothing
    (ckpt / "audio_tokenizer" / "config.json").write_text('{"x": 1}', encoding="utf-8")
    changed = checkpoint_fingerprint(ckpt)
    assert changed != before
    (ckpt / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 33)  # size is in the digest
    assert checkpoint_fingerprint(ckpt) != changed


def test_model_version_fails_loudly_without_weights(tmp_path):
    with pytest.raises(SynthesisError, match="not found"):
        _ = _engine(tmp_path).model_version
    with pytest.raises(SynthesisError, match="BREEZE_CHECKPOINT_DIR"):
        checkpoint_fingerprint(None)
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(SynthesisError, match="empty"):
        _ = _engine(tmp_path).model_version


def test_unknown_fast_config_is_refused(tmp_path):
    with pytest.raises(SynthesisError, match="unknown fast config"):
        _engine(tmp_path, fast="all")


# -- worker argv / env ------------------------------------------------------------------------


def test_worker_argv_and_env(monkeypatch, tmp_path):
    captured = {}

    class _Transport:
        def __init__(self, argv, **kw):
            captured.update(argv=argv, **kw)

    monkeypatch.setattr("seiyuu.engines.breeze_engine.SubprocessTransport", _Transport)
    _checkpoint(tmp_path)
    _repo(tmp_path)
    engine = _engine(tmp_path, fast="off")
    (tmp_path / "python.exe").write_bytes(b"")
    engine._default_factory()
    argv = captured["argv"]
    assert argv[0] == str(tmp_path / "python.exe")
    assert argv[1].endswith("breeze_worker.py")
    assert argv[argv.index("--repo-dir") + 1] == str(tmp_path / "repo")
    assert argv[argv.index("--checkpoint-dir") + 1] == str(tmp_path / "ckpt")
    assert argv[argv.index("--fast") + 1] == "off"
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["name"] == "breeze"
    assert captured["boot_timeout"] == 50.0
    assert captured["protocol"] == PROTOCOL_VERSION  # the boot handshake checks it


def test_worker_env_switches_off_the_static_cuda_launcher_on_windows_only():
    base = {"PATH": "x", "QWEN_TTS_STREAM_BACKEND": "custom"}
    win = worker_env(base, windows=True)
    assert win["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] == "0"
    assert win["TRANSFORMERS_OFFLINE"] == "1" and win["PATH"] == "x"
    assert "QWEN_TTS_STREAM_BACKEND" not in win  # the stream runtime refuses a non-official one
    assert "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER" not in worker_env(base, windows=False)
    assert base == {"PATH": "x", "QWEN_TTS_STREAM_BACKEND": "custom"}  # never mutated
    here = worker_env({})
    assert ("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER" in here) == (os.name == "nt")


def _stub_worker(tmp_path, protocol):
    script = tmp_path / f"stub_worker_{protocol}.py"
    script.write_text(
        "import json, sys\n"
        f'print(json.dumps({{"ok": True, "event": "ready", "protocol": {protocol}}}), flush=True)\n'
        "for line in sys.stdin:\n"
        '    print(json.dumps({"ok": True}), flush=True)\n',
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_boot_refuses_a_worker_on_another_protocol(tmp_path):
    with pytest.raises(WorkerError, match=f"protocol 999, expected {PROTOCOL_VERSION}"):
        SubprocessTransport(
            _stub_worker(tmp_path, 999),
            env={**os.environ},
            cwd=None,
            boot_timeout=30.0,
            name="breeze",
            protocol=PROTOCOL_VERSION,
        )


def test_default_factory_explains_missing_setup(tmp_path):
    with pytest.raises(SynthesisError, match="BREEZE_WORKER_PYTHON"):
        _engine(tmp_path)._default_factory()
    (tmp_path / "python.exe").write_bytes(b"")
    with pytest.raises(SynthesisError, match="BREEZE_REPO_DIR"):
        _engine(tmp_path)._default_factory()
    _repo(tmp_path)
    with pytest.raises(SynthesisError, match="BREEZE_CHECKPOINT_DIR"):
        _engine(tmp_path)._default_factory()


# -- synthesis --------------------------------------------------------------------------------


def test_synthesize_loads_then_sends_one_clone_request(tmp_path):
    _voice(tmp_path)
    transports = []

    def factory():
        transports.append(FakeTransport(_happy))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    audio = engine.synthesize('"Hello there."', "hero_1a2b", _bound(tmp_path, seed=7))
    assert audio.sample_rate == 24000 and audio.samples.size > 0
    load, synth = transports[0].requests
    assert load == {"cmd": "load"}
    assert transports[0].timeouts == [50.0, 5.0]  # the load gets the load timeout
    ref = tmp_path / "voices" / "hero_1a2b" / "reference.wav"
    assert synth.pop("out_path").endswith(".breeze.wav")
    assert synth == {
        "cmd": "synthesize",
        "text": '"Hello there."',  # sent as written: Breeze was chosen on quoted lines
        "reference_wav": str(ref),
        "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
        "ref_text": TRANSCRIPT,
        "seed": 7,
    }  # a plain clone: no instruction, no guidance scale, and never the digest itself
    engine.synthesize("Again.", "hero_1a2b", _bound(tmp_path))
    assert [r["cmd"] for r in transports[0].requests] == ["load", "synthesize", "synthesize"]


def test_an_instruction_selects_voice_direction(tmp_path):
    _voice(tmp_path)
    transport = FakeTransport(_happy)
    engine = _engine(tmp_path, lambda: transport)
    engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path, instruction="  calm and dry  "))
    sent = transport.requests[-1]
    assert sent["instruction"] == "calm and dry"
    assert "cfg_scale" not in sent  # the worker applies its default guidance
    engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path, instruction="calm", cfg_scale=2))
    assert transport.requests[-1]["cfg_scale"] == 2.0
    # guidance needs an instruction: a scale alone (or a blank instruction) stays a plain clone
    engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path, instruction="  ", cfg_scale=3.0))
    assert "instruction" not in transport.requests[-1]
    assert "cfg_scale" not in transport.requests[-1]


def test_direction_rides_the_segment_key(tmp_path):
    _voice(tmp_path)

    def key(settings):
        return SegmentKey.build(
            engine="breeze", engine_model_version="v", voice_id="hero_1a2b",
            settings={k: v for k, v in settings.items() if k != "seed"}, seed=1,
            normalized_text="Hello there.",
        ).key_hash  # fmt: skip

    plain = key(_bound(tmp_path))
    directed = key(_bound(tmp_path, instruction="calm and dry"))
    assert len({plain, directed, key(_bound(tmp_path, instruction="angry"))}) == 3
    assert key(_bound(tmp_path, instruction="calm and dry", cfg_scale=2.0)) != directed
    assert key(_bound(tmp_path, text="Another transcript.")) != plain  # so does the transcript


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"instruction": 3.0}, "instruction must be text"),
        ({"instruction": "calm", "cfg_scale": "4"}, "cfg_scale must be a number"),
        ({"instruction": "calm", "cfg_scale": True}, "cfg_scale must be a number"),
        ({"instruction": "calm", "cfg_scale": 0}, "cfg_scale must be above 0"),
        ({"cfg_scale": float("nan")}, "cfg_scale must be above 0"),
    ],
)
def test_malformed_direction_fails_before_the_worker(tmp_path, extra, match):
    _voice(tmp_path)
    engine = _engine(tmp_path, lambda: pytest.fail("worker must not start"))
    with pytest.raises(SynthesisError, match=match):
        engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path, **extra))


def test_a_voice_that_binds_no_transcript_is_refused(tmp_path):
    # Breeze has no transcript-free mode, and a reference.txt the key never saw must not be
    # used: without the binding in the voice's settings the line fails before the worker starts.
    _voice(tmp_path)  # reference.txt is there, but nothing binds it
    engine = _engine(tmp_path, lambda: pytest.fail("worker must not start"))
    with pytest.raises(SynthesisError, match="binds no reference transcript"):
        engine.synthesize("Hi.", "hero_1a2b", {"seed": 1})


def test_a_transcript_that_no_longer_matches_its_binding_fails_loudly(tmp_path):
    _voice(tmp_path)
    engine = _engine(tmp_path, lambda: pytest.fail("worker must not start"))
    settings = _bound(tmp_path)
    txt = tmp_path / "voices" / "hero_1a2b" / "reference.txt"
    txt.write_text("I keep a large notebook.\n", encoding="utf-8")  # edited by hand
    with pytest.raises(SynthesisError, match="does not match"):
        engine.synthesize("Hi.", "hero_1a2b", settings)
    txt.write_text(TRANSCRIPT + "\n", encoding="utf-8")
    _write_wav(str(txt.with_name("reference.wav")), seconds=0.7)  # clip replaced, text kept
    with pytest.raises(SynthesisError, match="does not match"):
        engine.synthesize("Hi.", "hero_1a2b", settings)
    txt.unlink()
    with pytest.raises(SynthesisError, match="missing"):
        engine.synthesize("Hi.", "hero_1a2b", settings)


def test_a_qwen_designed_voice_carries_the_binding_breeze_checks(tmp_path):
    # The binding is engine-independent (clip bytes + text), and a single-voice render hands the
    # engine the voice's OWN settings (render_voice_args): so a voice cloned or designed for
    # Qwen, with its transcript bound the way the clone and design flows bind it, renders on
    # Breeze as it is. (`seiyuu render --engine breeze --voice <designed voice>` relies on this.)
    from seiyuu.services.clone_transcript import bind_transcript
    from seiyuu.voices import VoiceKind, VoiceMeta, render_voice_args

    _voice(tmp_path, transcript=None)
    ref = tmp_path / "voices" / "hero_1a2b" / "reference.wav"
    sha = hashlib.sha256(ref.read_bytes()).hexdigest()
    meta = VoiceMeta(
        voice_id="hero_1a2b", name="Hero", kind=VoiceKind.CLONED, engine="qwen3tts",
        reference_audio="reference.wav", consent_attested=True, seed=3,
        settings=bind_transcript(ref, sha, TRANSCRIPT),
    )  # fmt: skip
    engine_voice, settings = render_voice_args(meta)
    assert settings == {"ref_text_digest": reference_text_digest(sha, TRANSCRIPT)}
    transport = FakeTransport(_happy)
    engine = _engine(tmp_path, lambda: transport)
    engine.synthesize("Hi.", engine_voice, {**settings, "seed": meta.seed})
    assert transport.requests[-1]["ref_text"] == TRANSCRIPT
    assert transport.requests[-1]["seed"] == 3


def test_a_request_the_worker_refuses_fails_the_line_without_a_restart(tmp_path):
    # e.g. a paragraph too long for the model's window: a fresh worker would refuse it the same
    # way, so the healthy one is kept (no 8 GB reload) and the failure is not counted as a crash
    _voice(tmp_path)
    transports = []

    def refusing(message):
        if message["cmd"] == "synthesize" and len(message["text"]) > 10:
            return {
                "ok": False,
                "error": "RequestError: prompt too long: 2100 tokens",
                "oom": False,
                "request_error": True,
            }
        return _happy(message)

    def factory():
        transports.append(FakeTransport(refusing))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    with pytest.raises(SynthesisError, match="prompt too long: 2100 tokens") as failure:
        engine.synthesize("A very long paragraph.", "hero_1a2b", _bound(tmp_path))
    assert "hero_1a2b" in str(failure.value)  # names the voice and the line
    assert len(transports) == 1 and not transports[0].closed
    assert engine.worker_restarts == 0 and engine.worker_ooms == 0
    engine.synthesize("Short.", "hero_1a2b", _bound(tmp_path))  # same worker, still serving
    assert [r["cmd"] for r in transports[0].requests] == ["load", "synthesize", "synthesize"]


def test_missing_reference_fails_before_the_worker(tmp_path):
    engine = _engine(tmp_path, lambda: pytest.fail("worker must not start"))
    with pytest.raises(SynthesisError, match="no reference.wav"):
        engine.synthesize("Hi.", "ghost", {"seed": 1})
    with pytest.raises(SynthesisError, match="no reference.wav"):
        engine.prepare_voice("ghost")


def test_wordless_text_never_reaches_the_worker(tmp_path):
    _voice(tmp_path)
    engine = _engine(tmp_path, lambda: pytest.fail("worker must not start"))
    assert engine.synthesize("...", "hero_1a2b", _bound(tmp_path)).samples.size > 0  # a beat
    with pytest.raises(SynthesisError, match="empty text"):
        engine.synthesize("  ", "hero_1a2b", _bound(tmp_path))


def test_oom_kills_restarts_and_retries(tmp_path):
    _voice(tmp_path)
    transports = []
    replies = iter([{"ok": False, "error": "CUDA out of memory", "oom": True}])

    def behavior(message):
        if message["cmd"] == "synthesize":
            failure = next(replies, None)
            if failure is not None:
                return failure
        return _happy(message)

    def factory():
        transports.append(FakeTransport(behavior))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path))
    assert len(transports) == 2 and transports[0].closed
    assert engine.worker_restarts == 1 and engine.worker_ooms == 1
    assert transports[1].requests[0] == {"cmd": "load"}  # the fresh worker reloads
    assert transports[1].requests[1]["text"] == "Hi."  # and the same line is retried


def test_a_dead_or_hung_worker_is_replaced(tmp_path):
    _voice(tmp_path)
    transports = []

    class _Dying(FakeTransport):
        def request(self, message, *, timeout):
            if message["cmd"] == "synthesize" and len(transports) == 1:
                raise WorkerError("breeze worker timed out after 5s")
            return super().request(message, timeout=timeout)

    def factory():
        transports.append(_Dying(_happy))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    audio = engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path))
    assert audio.samples.size > 0
    assert transports[0].closed and len(transports) == 2  # killed, not left holding the GPU
    assert engine.worker_restarts == 1 and engine.worker_ooms == 0


def test_a_failed_load_is_retried_on_a_fresh_worker(tmp_path):
    _voice(tmp_path)
    transports = []

    def behavior(message):
        if message["cmd"] == "load" and len(transports) == 1:
            return {"ok": False, "error": "CUDA out of memory", "oom": True}
        return _happy(message)

    def factory():
        transports.append(FakeTransport(behavior))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path))
    assert len(transports) == 2 and transports[0].closed and engine.worker_ooms == 1
    with pytest.raises(WorkerError, match="failed to load"):
        _engine(tmp_path, lambda: FakeTransport(lambda m: {"ok": False, "error": "no CUDA"})).warm()


def test_failure_after_retries_is_loud_and_leaves_no_scratch_file(tmp_path, monkeypatch):
    _voice(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(scratch))
    engine = _engine(
        tmp_path,
        lambda: FakeTransport(lambda m: {"ok": m["cmd"] == "load", "error": "boom"}),
        max_restarts=0,
    )
    with pytest.raises(SynthesisError, match="failed after 1 attempt"):
        engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path))
    assert list(scratch.iterdir()) == []


def test_empty_audio_is_a_loud_failure_not_a_retry(tmp_path):
    _voice(tmp_path)
    transports = []

    def silent(message):
        if message["cmd"] == "synthesize":
            sf.write(message["out_path"], np.zeros(0, dtype=np.int16), 24000, subtype="PCM_16")
            return {"ok": True, "path": message["out_path"], "hit_cap": False}
        return _happy(message)

    def factory():
        transports.append(FakeTransport(silent))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    with pytest.raises(SynthesisError, match="empty audio"):
        engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path))
    # The worker is healthy, so it is not restarted. The render fails loudly on this block (as
    # for the other worker engines): the seed retry only follows a whisper verdict.
    assert len(transports) == 1 and not transports[0].closed


def test_capped_lines_are_counted(tmp_path):
    _voice(tmp_path)

    def capped(message):
        reply = _happy(message)
        if message["cmd"] == "synthesize":
            reply["hit_cap"] = True
        return reply

    engine = _engine(tmp_path, lambda: FakeTransport(capped))
    engine.synthesize("Hi.", "hero_1a2b", _bound(tmp_path))
    assert engine.capped_lines == 1


def test_unload_terminates_the_worker_and_the_next_line_reloads(tmp_path):
    _voice(tmp_path)
    transports = []

    def factory():
        transports.append(FakeTransport(_happy))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    engine.warm()
    engine.warm()  # already loaded: no second load request
    assert transports[0].requests == [{"cmd": "load"}]
    engine.unload()
    assert transports[0].closed and engine._transport is None
    engine.unload()  # idempotent
    engine.prepare_voice("hero_1a2b")
    assert len(transports) == 2 and transports[1].requests == [{"cmd": "load"}]


# -- long lines ----------------------------------------------------------------------------------

SENTENCE = "The rain kept falling on the old slate roof all through the night."
LONG_LINE = " ".join([SENTENCE] * 30)  # 2,039 characters: three pieces


def _by_length(message):
    """A worker that speaks 0.01 s per character, so a join can be checked to the sample."""
    if message["cmd"] == "synthesize":
        _write_wav(message["out_path"], seconds=0.01 * len(message["text"]))
        return {"ok": True, "path": message["out_path"], "sample_rate": 24000, "hit_cap": False}
    return _happy(message)


def test_a_long_line_is_spoken_in_pieces_and_joined(tmp_path):
    _voice(tmp_path)
    transport = FakeTransport(_by_length)
    engine = _engine(tmp_path, lambda: transport)
    settings = _bound(tmp_path, instruction="Calm and dry.", cfg_scale=3.0)
    audio = engine.synthesize(LONG_LINE, "hero_1a2b", settings)

    spoken = [r for r in transport.requests if r["cmd"] == "synthesize"]
    assert len(spoken) == 3
    assert " ".join(r["text"] for r in spoken).split() == LONG_LINE.split()  # every word, in order
    assert all(len(r["text"]) <= SPLIT_OVER_CHARS for r in spoken)
    # one performance: the same reference, transcript, seed and direction for every piece
    for field in ("reference_wav", "reference_sha256", "ref_text", "seed", "instruction"):
        assert len({r[field] for r in spoken}) == 1, field
    assert {r["cfg_scale"] for r in spoken} == {3.0}
    assert len({r["out_path"] for r in spoken}) == 3

    sr = BREEZE_SAMPLE_RATE
    expected = sum(int(0.01 * len(r["text"]) * sr) for r in spoken) + 2 * round(
        PIECE_GAP_SECONDS * sr
    )
    assert audio.samples.size == expected and audio.sample_rate == sr
    first = int(0.01 * len(spoken[0]["text"]) * sr)
    gap = audio.samples[first : first + round(PIECE_GAP_SECONDS * sr)]
    assert not gap.any()  # a silent pause between the pieces
    assert engine.split_lines == 1 and engine.capped_lines == 0


def test_a_line_at_the_limit_is_still_one_request_with_its_text_untouched(tmp_path):
    _voice(tmp_path)
    transport = FakeTransport(_happy)
    engine = _engine(tmp_path, lambda: transport)
    line = ("word " * 160).strip()  # 799 characters
    engine.synthesize(line, "hero_1a2b", _bound(tmp_path))
    spoken = [r for r in transport.requests if r["cmd"] == "synthesize"]
    assert [r["text"] for r in spoken] == [line] and engine.split_lines == 0


def test_a_capped_piece_counts_the_line_once(tmp_path):
    _voice(tmp_path)

    def capped(message):
        reply = _by_length(message)
        if message["cmd"] == "synthesize":
            reply["hit_cap"] = True
        return reply

    engine = _engine(tmp_path, lambda: FakeTransport(capped))
    engine.synthesize(LONG_LINE, "hero_1a2b", _bound(tmp_path))
    assert engine.capped_lines == 1  # three capped pieces, one line


def test_a_refused_piece_fails_the_whole_line_and_names_the_piece(tmp_path):
    _voice(tmp_path)
    seen = {"n": 0}

    def refuse_second(message):
        if message["cmd"] == "synthesize":
            seen["n"] += 1
            if seen["n"] == 2:
                return {"ok": False, "error": "prompt too long", "request_error": True}
        return _by_length(message)

    transport = FakeTransport(refuse_second)
    engine = _engine(tmp_path, lambda: transport)
    with pytest.raises(SynthesisError, match="piece 2 of 3"):
        engine.synthesize(LONG_LINE, "hero_1a2b", _bound(tmp_path))
    assert engine.worker_restarts == 0 and not transport.closed  # no partial audio is returned


def test_an_empty_piece_is_a_loud_failure(tmp_path):
    _voice(tmp_path)
    seen = {"n": 0}

    def silent_third(message):
        if message["cmd"] == "synthesize":
            seen["n"] += 1
            if seen["n"] == 3:
                sf.write(message["out_path"], np.zeros(0, dtype=np.float32), 24000)
                return {"ok": True, "path": message["out_path"], "sample_rate": 24000}
        return _by_length(message)

    engine = _engine(tmp_path, lambda: FakeTransport(silent_third))
    with pytest.raises(SynthesisError, match="empty audio.*piece 3 of 3"):
        engine.synthesize(LONG_LINE, "hero_1a2b", _bound(tmp_path))


# -- voice design ---------------------------------------------------------------------------------


def _designs(message):
    if message["cmd"] == "design":
        _write_wav(message["out_path"], seconds=0.5)
        return {"ok": True, "path": message["out_path"], "sample_rate": 24000, "hit_cap": False}
    return _happy(message)


def test_design_voice_sends_a_reference_free_request(tmp_path):
    transport = FakeTransport(_designs)
    engine = _engine(tmp_path, lambda: transport)
    audio = engine.design_voice("Hello there.", "  A warm, low voice. ", seed=77)
    assert audio.sample_rate == 24_000 and audio.samples.size == 12_000
    load, design = transport.requests
    assert load["cmd"] == "load"
    assert {k: v for k, v in design.items() if k != "out_path"} == {
        "cmd": "design",
        "text": "Hello there.",
        "instruction": "A warm, low voice.",
        "seed": 77,
    }  # no voice, no clip, no transcript
    engine.design_voice("Hi.", "Bright.", seed=1, settings={"cfg_scale": 2})
    assert transport.requests[-1]["cfg_scale"] == 2.0


def test_design_voice_needs_text_and_a_description(tmp_path):
    engine = _engine(tmp_path, lambda: FakeTransport(_designs))
    for text, description in (("", "Warm."), ("Hello.", "  ")):
        with pytest.raises(SynthesisError, match="needs both text and a description"):
            engine.design_voice(text, description, seed=1)


def test_a_design_survives_a_worker_restart_like_a_line(tmp_path):
    transports = []

    def flaky(message):
        if message["cmd"] == "design" and len(transports) == 1:
            return {"ok": False, "error": "CUDA out of memory", "oom": True}
        return _designs(message)

    def factory():
        transports.append(FakeTransport(flaky))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    assert engine.design_voice("Hello there.", "Warm.", seed=3).samples.size > 0
    assert engine.worker_restarts == 1 and engine.worker_ooms == 1 and transports[0].closed
