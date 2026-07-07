"""IndexTTS-2 adapter tests — CPU only, worker faked (no subprocess, no SDK, no GPU).

A fake transport FACTORY stands in for the real subprocess: it records requests and writes a
handoff WAV so the readback + 22050->24000 resample path is exercised end to end. The real worker
subprocess is only driven by a gated GPU smoke test in the separate env, never here.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from seiyuu.engines import get_engine
from seiyuu.engines import indextts2_worker as worker_module
from seiyuu.engines.base import SynthesisError
from seiyuu.engines.indextts2_engine import (
    INDEXTTS2_SAMPLE_RATE,
    IndexTTS2Engine,
    SubprocessTransport,
    WorkerError,
    checkpoints_present,
    weights_fingerprint,
)


def _write_wav(path: str, *, seconds: float = 0.1, sr: int = INDEXTTS2_SAMPLE_RATE) -> None:
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="PCM_16")


class FakeTransport:
    """Records every request; drives its reply from ``behavior(message) -> reply``. On a
    successful synthesize the behavior is responsible for writing the handoff WAV to out_path."""

    def __init__(self, behavior):
        self.requests: list[dict] = []
        self.closed = False
        self._behavior = behavior

    def request(self, message, *, timeout):
        self.requests.append(message)
        return self._behavior(message)

    def close(self):
        self.closed = True


def _happy_behavior(message):
    if message["cmd"] == "load":
        return {"ok": True}
    if message["cmd"] == "synthesize":
        _write_wav(message["out_path"])
        return {"ok": True, "sample_rate": INDEXTTS2_SAMPLE_RATE, "path": message["out_path"]}
    return {"ok": False, "error": "unexpected"}


def _voice(tmp_path, voice_id="hero_1a2b", *, with_reference=True) -> Path:
    d = tmp_path / voice_id
    d.mkdir(parents=True)
    if with_reference:
        (d / "reference.wav").write_bytes(b"RIFFfake")
    return tmp_path


def _engine(
    tmp_path, factory, *, checkpoints_dir=None, max_restarts=1, use_fp16=True
) -> IndexTTS2Engine:
    return IndexTTS2Engine(
        voices_dir=tmp_path,
        checkpoints_dir=checkpoints_dir if checkpoints_dir is not None else tmp_path / "ckpt",
        worker_python=tmp_path / "python.exe",  # unused when a factory is injected
        use_fp16=use_fp16,
        load_timeout=5.0,
        request_timeout=5.0,
        max_restarts=max_restarts,
        transport_factory=factory,
    )


# -- catalog facts ---------------------------------------------------------------------------


def test_catalog_facts(tmp_path):
    eng = _engine(tmp_path, lambda: FakeTransport(_happy_behavior))
    assert eng.engine_id == "indextts2"
    assert eng.native_sample_rate == 22050
    assert eng.requires_validation is True and eng.uses_gpu is True
    assert eng.clones_from_library is True
    assert eng.list_voices() == [] and eng.cost_estimate("anything") == 0.0


def test_get_engine_resolves_indextts2(tmp_path):
    eng = get_engine(
        "indextts2",
        voices_dir=tmp_path,
        checkpoints_dir=tmp_path,
        worker_python=tmp_path / "py",
        use_fp16=False,
        load_timeout=1.0,
        request_timeout=1.0,
        max_restarts=0,
        transport_factory=lambda: FakeTransport(_happy_behavior),
    )
    assert isinstance(eng, IndexTTS2Engine)


# -- weights fingerprint (offline model_version) ---------------------------------------------


def _make_checkpoints(dir_path: Path, *, config="a: 1", weight_bytes=b"\x00" * 2048) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.yaml").write_text(config, encoding="utf-8")
    (dir_path / "gpt.pth").write_bytes(weight_bytes)
    return dir_path


def test_weights_fingerprint_stable_and_content_sensitive(tmp_path):
    a = _make_checkpoints(tmp_path / "a")
    b = _make_checkpoints(tmp_path / "b")
    assert weights_fingerprint(a) == weights_fingerprint(b)  # same (name,size,config) -> same id
    assert weights_fingerprint(a).startswith("indextts2-")

    changed_config = _make_checkpoints(tmp_path / "c", config="a: 2")
    assert weights_fingerprint(changed_config) != weights_fingerprint(a)  # config content matters

    changed_weight = _make_checkpoints(tmp_path / "d", weight_bytes=b"\x00" * 4096)
    assert weights_fingerprint(changed_weight) != weights_fingerprint(a)  # weight SIZE matters


def test_weights_fingerprint_missing_or_empty_raises(tmp_path):
    with pytest.raises(SynthesisError, match="no checkpoints dir configured"):
        weights_fingerprint(None)
    with pytest.raises(SynthesisError, match="not found"):
        weights_fingerprint(tmp_path / "does_not_exist")
    (tmp_path / "empty").mkdir()
    with pytest.raises(SynthesisError, match="empty"):
        weights_fingerprint(tmp_path / "empty")


def test_checkpoints_present(tmp_path):
    assert checkpoints_present(None) is False
    assert checkpoints_present(tmp_path / "nope") is False
    (tmp_path / "empty").mkdir()
    assert checkpoints_present(tmp_path / "empty") is False
    ckpt = _make_checkpoints(tmp_path / "ckpt")
    assert checkpoints_present(ckpt) is True


def test_model_version_folds_checkpoints_and_precision(tmp_path):
    ckpt = _make_checkpoints(tmp_path / "ckpt")
    eng16 = _engine(tmp_path, lambda: FakeTransport(_happy_behavior), checkpoints_dir=ckpt)
    assert eng16.model_version == f"{weights_fingerprint(ckpt)}-fp16"
    assert eng16.model_version == eng16.model_version  # memoized, stable

    # fp16 vs fp32 changes the sampled audio, so it must change the key (base.py contract) —
    # else flipping indextts2_use_fp16 and re-rendering would serve stale cached audio.
    eng32 = _engine(
        tmp_path, lambda: FakeTransport(_happy_behavior), checkpoints_dir=ckpt, use_fp16=False
    )
    assert eng32.model_version == f"{weights_fingerprint(ckpt)}-fp32"
    assert eng16.model_version != eng32.model_version


# -- synthesis + settings forwarding ---------------------------------------------------------


def test_synthesize_returns_canonical_24k_from_22050(tmp_path):
    _voice(tmp_path)
    eng = _engine(tmp_path, lambda: FakeTransport(_happy_behavior))
    audio = eng.synthesize("Hello there.", "hero_1a2b", {"seed": 3})
    assert audio.sample_rate == 24_000  # to_canonical resampled 22050 -> 24000
    assert audio.samples.dtype == np.float32 and audio.samples.ndim == 1


def test_seed_and_emotion_are_forwarded_to_the_worker(tmp_path):
    _voice(tmp_path)
    transports: list[FakeTransport] = []

    def factory():
        t = FakeTransport(_happy_behavior)
        transports.append(t)
        return t

    eng = _engine(tmp_path, factory)
    emo = [0.0, 0.64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # ANGRY@2 from voices/emotion.py
    eng.synthesize("Rage.", "hero_1a2b", {"seed": 9, "emo_vector": emo, "emo_alpha": 0.8})

    synth = next(r for r in transports[0].requests if r["cmd"] == "synthesize")
    assert synth["seed"] == 9 and synth["emo_alpha"] == 0.8
    assert synth["emo_vector"] == emo
    assert synth["reference_wav"].endswith("reference.wav")
    assert synth["out_path"].endswith(".wav")


def test_neutral_settings_send_none_emotion(tmp_path):
    _voice(tmp_path)
    transports: list[FakeTransport] = []
    eng = _engine(tmp_path, lambda: transports.append(t := FakeTransport(_happy_behavior)) or t)
    eng.synthesize("Calm.", "hero_1a2b", {"seed": 1})
    synth = next(r for r in transports[0].requests if r["cmd"] == "synthesize")
    assert synth["emo_vector"] is None and synth["emo_alpha"] is None


def test_load_is_sent_once_per_worker(tmp_path):
    _voice(tmp_path)
    transports: list[FakeTransport] = []
    eng = _engine(tmp_path, lambda: transports.append(t := FakeTransport(_happy_behavior)) or t)
    eng.synthesize("One.", "hero_1a2b", {"seed": 1})
    eng.synthesize("Two.", "hero_1a2b", {"seed": 2})
    cmds = [r["cmd"] for r in transports[0].requests]
    assert cmds.count("load") == 1 and cmds.count("synthesize") == 2  # one worker, loaded once


def test_missing_reference_raises(tmp_path):
    _voice(tmp_path, with_reference=False)
    eng = _engine(tmp_path, lambda: FakeTransport(_happy_behavior))
    with pytest.raises(SynthesisError, match="no reference.wav"):
        eng.synthesize("Hi.", "hero_1a2b", {"seed": 1})


def test_load_failure_raises_worker_error(tmp_path):
    _voice(tmp_path)

    def behavior(message):
        if message["cmd"] == "load":
            return {"ok": False, "error": "checkpoint corrupt"}
        return _happy_behavior(message)

    eng = _engine(tmp_path, lambda: FakeTransport(behavior), max_restarts=0)
    with pytest.raises(SynthesisError, match="failed to load model"):
        eng.synthesize("Hi.", "hero_1a2b", {"seed": 1})


# -- OOM restart + unload --------------------------------------------------------------------


def test_oom_kills_worker_restarts_and_retries(tmp_path):
    _voice(tmp_path)
    transports: list[FakeTransport] = []

    def make(behavior):
        def factory():
            t = FakeTransport(behavior)
            transports.append(t)
            return t

        return factory

    # first worker OOMs on synthesize; the SECOND (fresh) worker succeeds.
    def first_oom(message):
        if message["cmd"] == "synthesize" and len(transports) == 1:
            return {"ok": False, "error": "CUDA OOM", "oom": True}
        return _happy_behavior(message)

    eng = _engine(tmp_path, make(first_oom), max_restarts=1)
    audio = eng.synthesize("Big block.", "hero_1a2b", {"seed": 1})
    assert audio.sample_rate == 24_000
    assert len(transports) == 2  # killed + restarted once
    assert transports[0].closed is True  # the OOMed worker was terminated


def test_oom_exhausts_restarts_then_raises(tmp_path):
    _voice(tmp_path)
    transports: list[FakeTransport] = []

    def always_oom(message):
        if message["cmd"] == "synthesize":
            return {"ok": False, "error": "CUDA OOM", "oom": True}
        return {"ok": True}

    def factory():
        t = FakeTransport(always_oom)
        transports.append(t)
        return t

    eng = _engine(tmp_path, factory, max_restarts=1)
    with pytest.raises(SynthesisError, match="failed after 2 attempt"):
        eng.synthesize("Too big.", "hero_1a2b", {"seed": 1})
    assert len(transports) == 2 and all(t.closed for t in transports)


def test_empty_audio_is_a_loud_failure(tmp_path):
    _voice(tmp_path)

    def empties(message):
        if message["cmd"] == "synthesize":
            _write_wav(message["out_path"], seconds=0.0)  # zero-length wav
            return {"ok": True, "sample_rate": INDEXTTS2_SAMPLE_RATE, "path": message["out_path"]}
        return {"ok": True}

    eng = _engine(tmp_path, lambda: FakeTransport(empties), max_restarts=0)
    with pytest.raises(SynthesisError, match="empty audio"):
        eng.synthesize("Hi.", "hero_1a2b", {"seed": 1})


def test_unload_terminates_worker(tmp_path):
    _voice(tmp_path)
    transports: list[FakeTransport] = []
    eng = _engine(tmp_path, lambda: transports.append(t := FakeTransport(_happy_behavior)) or t)
    eng.warm()  # boots + loads a worker
    assert transports and transports[0].closed is False
    eng.unload()
    assert transports[0].closed is True  # synchronous terminate = VRAM reclaim

    # a synthesize after unload boots a FRESH worker (loaded again)
    eng.synthesize("After.", "hero_1a2b", {"seed": 1})
    assert len(transports) == 2 and "load" in [r["cmd"] for r in transports[1].requests]


def test_handoff_wav_is_cleaned_up(tmp_path):
    _voice(tmp_path)
    seen_paths: list[str] = []

    def behavior(message):
        if message["cmd"] == "synthesize":
            seen_paths.append(message["out_path"])
        return _happy_behavior(message)

    eng = _engine(tmp_path, lambda: FakeTransport(behavior))
    eng.synthesize("Hi.", "hero_1a2b", {"seed": 1})
    assert seen_paths and not Path(seen_paths[0]).exists()  # temp handoff file removed


def test_worker_error_carries_oom_flag():
    err = WorkerError("boom", oom=True)
    assert isinstance(err, SynthesisError) and err.oom is True


# -- real SubprocessTransport (CPU only: drives the actual worker's ping, no torch/GPU) -------


def test_real_worker_ping_over_subprocess_transport(tmp_path):
    """End-to-end of the stdio protocol layer: the real worker booted as a subprocess, a real
    ping round-trip, and a synchronous close(). `ping` never loads weights, so no torch/GPU."""
    worker = Path(worker_module.__file__)
    transport = SubprocessTransport(
        [sys.executable, str(worker), "--checkpoints", str(tmp_path)],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=None,
        boot_timeout=30.0,
    )
    try:
        reply = transport.request({"cmd": "ping"}, timeout=30.0)
        assert reply == {"ok": True, "protocol": worker_module.PROTOCOL_VERSION}
    finally:
        transport.close()


def _spy_popen(monkeypatch):
    """Record every real subprocess spawned so a test can assert it was terminated, not orphaned."""
    created: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", spy)
    return created


def test_boot_garbage_line_raises_and_terminates_worker(tmp_path, monkeypatch):
    """A worker whose first stdout line isn't the ready handshake (e.g. an env banner) must not be
    orphaned: __init__ raises WorkerError AND the spawned process is terminated."""
    script = tmp_path / "bad_worker.py"
    script.write_text(
        'import time\nprint(\'{"not": "ready"}\', flush=True)\ntime.sleep(60)\n',
        encoding="utf-8",
    )
    created = _spy_popen(monkeypatch)
    with pytest.raises(WorkerError, match="did not announce readiness"):
        SubprocessTransport(
            [sys.executable, str(script)], env={**os.environ}, cwd=None, boot_timeout=10.0
        )
    assert created and created[0].poll() is not None  # terminated, not left running


def test_boot_timeout_raises_and_terminates_worker(tmp_path, monkeypatch):
    """A worker that never announces readiness within boot_timeout is killed, not orphaned."""
    script = tmp_path / "silent_worker.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    created = _spy_popen(monkeypatch)
    with pytest.raises(WorkerError, match="timed out"):
        SubprocessTransport(
            [sys.executable, str(script)], env={**os.environ}, cwd=None, boot_timeout=1.5
        )
    assert created and created[0].poll() is not None
