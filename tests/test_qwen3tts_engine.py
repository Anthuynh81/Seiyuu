"""Qwen3-TTS adapter tests — CPU only, worker faked (no subprocess, no SDK, no GPU).

A fake transport FACTORY stands in for the worker subprocess, as in the IndexTTS-2 tests: it
records requests and writes the handoff WAV, so the readback path runs end to end.
"""

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from seiyuu.api.registry import ENGINE_FACTS, EngineRegistry
from seiyuu.engines import get_engine_class, list_engine_ids, voices_dir_kwargs
from seiyuu.engines.base import SynthesisError
from seiyuu.engines.qwen3tts_engine import (
    BASE_CHECKPOINT,
    QWEN3TTS_SAMPLE_RATE,
    Qwen3TTSEngine,
    models_present,
    qwen3tts_config,
    weights_fingerprint,
)
from seiyuu.gpu import GpuResourceManager
from seiyuu.normalize import normalize_text, profile_for
from seiyuu.settings import Settings


def _write_wav(path: str, *, seconds: float = 0.2, sr: int = QWEN3TTS_SAMPLE_RATE) -> None:
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="PCM_16")


class FakeTransport:
    def __init__(self, behavior):
        self.requests: list[dict] = []
        self.closed = False
        self._behavior = behavior

    def request(self, message, *, timeout):
        self.requests.append(dict(message))
        return self._behavior(message)

    def close(self):
        self.closed = True


def _happy(message):
    if message["cmd"] == "load":
        return {"ok": True, "model": message["model"]}
    if message["cmd"] in ("synthesize", "design"):
        _write_wav(message["out_path"])
        return {"ok": True, "path": message["out_path"], "sample_rate": 24000, "hit_cap": False}
    return {"ok": False, "error": "unexpected"}


def _models(tmp_path):
    d = tmp_path / "models" / BASE_CHECKPOINT
    (d / "speech_tokenizer").mkdir(parents=True)
    (d / "config.json").write_text('{"tts_model_type": "base"}', encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\0" * 32)
    (d / "speech_tokenizer" / "model.safetensors").write_bytes(b"\0" * 16)
    return tmp_path / "models"


def _voice(tmp_path, voice_id="hero_1a2b", *, transcript=None):
    d = tmp_path / "voices" / voice_id
    d.mkdir(parents=True)
    _write_wav(str(d / "reference.wav"), seconds=0.5)
    if transcript is not None:
        (d / "reference.txt").write_text(transcript + "\n", encoding="utf-8")
    return tmp_path / "voices"


def _engine(tmp_path, factory=None, **kw):
    params = dict(
        voices_dir=tmp_path / "voices",
        models_dir=kw.pop("models_dir", tmp_path / "models"),
        worker_python=tmp_path / "python.exe",
        backend="fast",
        language="English",
        max_seq_len=2048,
        load_timeout=5.0,
        request_timeout=5.0,
        max_restarts=1,
        transport_factory=factory,
    )
    params.update(kw)
    return Qwen3TTSEngine(**params)


# -- catalog / registration -------------------------------------------------------------------


def test_registered_as_a_local_cloning_engine_that_needs_validation(tmp_path):
    assert "qwen3tts" in list_engine_ids()
    cls = get_engine_class("qwen3tts")
    assert cls is Qwen3TTSEngine
    assert cls.requires_validation and cls.clones_from_library and cls.uses_gpu
    assert voices_dir_kwargs("qwen3tts", tmp_path) == {"voices_dir": tmp_path}
    assert ENGINE_FACTS["qwen3tts"] == {"paid": False, "supports_cloning": True}


def test_text_profile_is_explicit_and_folds_dashes_like_the_default():
    assert profile_for("qwen3tts") == "qwen3tts"
    line = "Wait — I… no."
    assert normalize_text(line, profile="qwen3tts") == normalize_text(line, profile="default")


def test_config_defaults_and_settings():
    assert qwen3tts_config(SimpleNamespace()) == {
        "worker_python": None,
        "models_dir": None,
        "backend": "fast",
        "language": "English",
        "max_seq_len": 2048,
    }
    cfg = Settings(_env_file=None)
    assert cfg.qwen3tts_backend == "fast" and cfg.qwen3tts_language == "English"
    assert cfg.qwen3tts_worker_max_restarts == 1


def test_registry_constructs_from_injected_settings(tmp_path):
    settings = Settings(
        _env_file=None,
        voices_dir=tmp_path / "voices",
        data_dir=tmp_path / "data",
        qwen3tts_models_dir=tmp_path / "models",
        qwen3tts_worker_python=tmp_path / "py.exe",
        qwen3tts_backend="stock",
        qwen3tts_worker_request_timeout=12.0,
    )
    engine = EngineRegistry(settings, gpu_manager=GpuResourceManager()).get("qwen3tts")
    assert engine._voices_dir == settings.voices_dir  # consent invariant: same voices root
    assert engine._models_dir == tmp_path / "models"
    assert engine._backend == "stock" and engine._request_timeout == 12.0


# -- model_version ----------------------------------------------------------------------------


def test_model_version_fingerprints_the_base_checkpoint(tmp_path):
    models = _models(tmp_path)
    fp = weights_fingerprint(models / BASE_CHECKPOINT)
    assert fp.startswith("qwen3tts-") and len(fp) == len("qwen3tts-") + 12
    assert _engine(tmp_path).model_version == f"{fp}-bf16"
    assert _engine(tmp_path, backend="stock").model_version == f"{fp}-bf16-stock"
    assert _engine(tmp_path, language="Auto").model_version == f"{fp}-bf16-auto"
    assert models_present(models) and not models_present(tmp_path / "nope")


def test_fingerprint_ignores_hub_bookkeeping_but_not_configs(tmp_path):
    models = _models(tmp_path)
    base = models / BASE_CHECKPOINT
    before = weights_fingerprint(base)
    cache = base / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "model.safetensors.metadata").write_text("etag 2026-11-07", encoding="utf-8")
    assert weights_fingerprint(base) == before  # re-running a download changes nothing
    (base / "config.json").write_text('{"tts_model_type": "base", "x": 1}', encoding="utf-8")
    assert weights_fingerprint(base) != before


def test_model_version_fails_loudly_without_weights(tmp_path):
    with pytest.raises(SynthesisError, match="not found"):
        _ = _engine(tmp_path).model_version
    with pytest.raises(SynthesisError, match="QWEN3TTS_MODELS_DIR"):
        weights_fingerprint(None)
    (tmp_path / "models" / BASE_CHECKPOINT).mkdir(parents=True)
    with pytest.raises(SynthesisError, match="empty"):
        _ = _engine(tmp_path).model_version


def test_unknown_backend_is_refused(tmp_path):
    with pytest.raises(SynthesisError, match="unknown backend"):
        _engine(tmp_path, backend="ggml")


# -- worker argv ------------------------------------------------------------------------------


def test_worker_argv_and_env(monkeypatch, tmp_path):
    captured = {}

    class _Transport:
        def __init__(self, argv, **kw):
            captured.update(argv=argv, **kw)

    monkeypatch.setattr("seiyuu.engines.qwen3tts_engine.SubprocessTransport", _Transport)
    _models(tmp_path)
    engine = _engine(tmp_path, backend="stock", max_seq_len=3072)
    (tmp_path / "python.exe").write_bytes(b"")
    engine._default_factory()
    argv = captured["argv"]
    assert argv[0] == str(tmp_path / "python.exe")
    assert argv[1].endswith("qwen3tts_worker.py")
    assert argv[argv.index("--models-dir") + 1] == str(tmp_path / "models")
    assert argv[argv.index("--backend") + 1] == "stock"
    assert argv[argv.index("--language") + 1] == "English"
    assert argv[argv.index("--max-seq-len") + 1] == "3072"
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["name"] == "qwen3tts"


def test_default_factory_explains_missing_setup(tmp_path):
    with pytest.raises(SynthesisError, match="QWEN3TTS_WORKER_PYTHON"):
        _engine(tmp_path)._default_factory()
    (tmp_path / "python.exe").write_bytes(b"")
    with pytest.raises(SynthesisError, match="QWEN3TTS_MODELS_DIR"):
        _engine(tmp_path)._default_factory()


# -- synthesis --------------------------------------------------------------------------------


def test_synthesize_loads_base_then_sends_one_clone_request(tmp_path):
    _voice(tmp_path)
    transports = []

    def factory():
        transports.append(FakeTransport(_happy))
        return transports[-1]

    engine = _engine(tmp_path, factory)
    audio = engine.synthesize('"Hello there."', "hero_1a2b", {"seed": 7})
    assert audio.sample_rate == 24000 and audio.samples.size > 0
    load, synth = transports[0].requests
    assert load == {"cmd": "load", "model": "base"}
    ref = tmp_path / "voices" / "hero_1a2b" / "reference.wav"
    assert synth["cmd"] == "synthesize"
    assert synth["text"] == '"Hello there."'  # the worker strips the quotes, not the adapter
    assert synth["reference_wav"] == str(ref)
    assert synth["reference_sha256"] == hashlib.sha256(ref.read_bytes()).hexdigest()
    assert synth["ref_text"] is None  # no reference.txt -> x-vector clone
    assert synth["seed"] == 7
    assert "gen" not in synth  # no tunables -> the key is omitted
    engine.synthesize("Again.", "hero_1a2b", {"seed": 7})
    assert [r["cmd"] for r in transports[0].requests] == ["load", "synthesize", "synthesize"]


def test_reference_transcript_selects_icl_unless_older_than_the_clip(tmp_path):
    _voice(tmp_path, transcript="I keep a small notebook.")
    transport = FakeTransport(_happy)
    engine = _engine(tmp_path, lambda: transport)
    engine.synthesize("Hi.", "hero_1a2b", {"seed": 1})
    assert transport.requests[-1]["ref_text"] == "I keep a small notebook."

    # a re-clone replaces reference.wav but not the transcript: the stale text is ignored
    ref = tmp_path / "voices" / "hero_1a2b" / "reference.wav"
    txt = ref.with_name("reference.txt")
    st = txt.stat()
    os.utime(txt, ns=(st.st_atime_ns, st.st_mtime_ns - 10_000_000_000))
    engine.synthesize("Hi.", "hero_1a2b", {"seed": 1})
    assert transport.requests[-1]["ref_text"] is None


def test_whitelisted_tunables_only(tmp_path):
    _voice(tmp_path)
    transport = FakeTransport(_happy)
    engine = _engine(tmp_path, lambda: transport)
    engine.synthesize(
        "Hi.", "hero_1a2b", {"seed": 1, "temperature": 0.7, "top_k": 30.0, "do_sample": 0.0}
    )
    assert transport.requests[-1]["gen"] == {"temperature": 0.7, "top_k": 30.0}


def test_missing_reference_fails_before_the_worker(tmp_path):
    engine = _engine(tmp_path, lambda: pytest.fail("worker must not start"))
    with pytest.raises(SynthesisError, match="no reference.wav"):
        engine.synthesize("Hi.", "ghost", {"seed": 1})


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
    engine.synthesize("Hi.", "hero_1a2b", {"seed": 1})
    assert len(transports) == 2 and transports[0].closed
    assert engine.worker_restarts == 1 and engine.worker_ooms == 1
    assert transports[1].requests[0] == {"cmd": "load", "model": "base"}  # fresh worker reloads


def test_failure_after_retries_is_loud(tmp_path):
    _voice(tmp_path)
    engine = _engine(
        tmp_path,
        lambda: FakeTransport(lambda m: {"ok": m["cmd"] == "load", "error": "boom"}),
        max_restarts=0,
    )
    with pytest.raises(SynthesisError, match="failed after 1 attempt"):
        engine.synthesize("Hi.", "hero_1a2b", {"seed": 1})


def test_capped_lines_are_counted(tmp_path):
    _voice(tmp_path)

    def capped(message):
        reply = _happy(message)
        if message["cmd"] == "synthesize":
            reply["hit_cap"] = True
        return reply

    engine = _engine(tmp_path, lambda: FakeTransport(capped))
    engine.synthesize("Hi.", "hero_1a2b", {"seed": 1})
    assert engine.capped_lines == 1


def test_unload_terminates_the_worker(tmp_path):
    _voice(tmp_path)
    transport = FakeTransport(_happy)
    engine = _engine(tmp_path, lambda: transport)
    engine.warm()
    engine.unload()
    assert transport.closed and engine._transport is None


# -- voice design -----------------------------------------------------------------------------


def test_design_swaps_to_the_design_checkpoint_and_back(tmp_path):
    _voice(tmp_path)
    transport = FakeTransport(_happy)
    engine = _engine(tmp_path, lambda: transport)
    audio = engine.design_voice("I keep a notebook.", "gruff old janitor", seed=11)
    assert audio.sample_rate == 24000
    engine.synthesize("Hi.", "hero_1a2b", {"seed": 1})
    cmds = [(r["cmd"], r.get("model")) for r in transport.requests]
    assert cmds == [("load", "design"), ("design", None), ("load", "base"), ("synthesize", None)]
    design = transport.requests[1]
    assert design["instruct"] == "gruff old janitor" and design["seed"] == 11
    assert design["text"] == "I keep a notebook."


def test_design_needs_text_and_description(tmp_path):
    engine = _engine(tmp_path, lambda: pytest.fail("worker must not start"))
    with pytest.raises(SynthesisError, match="description"):
        engine.design_voice("Hello.", "  ", seed=1)
