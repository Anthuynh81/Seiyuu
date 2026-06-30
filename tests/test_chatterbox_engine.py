"""Chatterbox adapter — CPU tests with the SDK mocked (no weights, no GPU).

Covers the load-vs-prepare conds cache, settings->generate mapping, the seed-before-generate
requirement, and unload. The real model is exercised only by the gated test_chatterbox_gpu.py.
"""

from importlib.metadata import version as pkg_version

import pytest
import torch

from seiyuu.engines import get_engine
from seiyuu.engines.base import SynthesisError
from seiyuu.engines.chatterbox_engine import ChatterboxEngine


class _FakeConds:
    def save(self, path):
        from pathlib import Path

        Path(path).write_text("conds", encoding="utf-8")  # create file so the cache exists


class _FakeModel:
    def __init__(self):
        self.conds = None
        self.prepared = None
        self.gen_calls = []

    def prepare_conditionals(self, reference, exaggeration=0.5):
        self.prepared = (reference, exaggeration)
        self.conds = _FakeConds()

    def generate(self, text, **kwargs):
        # Capture the active torch seed to prove we seeded BEFORE generating.
        self.gen_calls.append({"text": text, "kwargs": kwargs, "seed": torch.initial_seed()})
        return torch.zeros(1, 2400)  # (1, n) float tensor, "24 kHz"


def _voice(tmp_path, voice_id="elena_9f3a", *, with_reference=True):
    d = tmp_path / voice_id
    d.mkdir(parents=True)
    if with_reference:
        (d / "reference.wav").write_bytes(b"RIFFfake")
    return tmp_path


def test_model_version_native_rate_and_listing(tmp_path):
    eng = ChatterboxEngine(device="cpu", voices_dir=tmp_path, model=_FakeModel())
    assert eng.model_version == f"chatterbox-{pkg_version('chatterbox-tts')}"
    assert eng.native_sample_rate == 24_000
    assert eng.list_voices() == [] and eng.cost_estimate("x") == 0.0


def test_conds_path_embeds_model_version(tmp_path):
    eng = ChatterboxEngine(device="cpu", voices_dir=tmp_path, model=_FakeModel())
    assert eng.conds_path("v").name == f"conds_{eng.model_version}.pt"


def test_prepares_and_caches_conds_then_generates_with_seed(tmp_path):
    voices = _voice(tmp_path)
    model = _FakeModel()
    eng = ChatterboxEngine(device="cpu", voices_dir=voices, model=model)
    settings = {"seed": 123, "exaggeration": 0.7, "cfg_weight": 0.4, "temperature": 0.9, "speed": 9}

    audio = eng.synthesize("Hello there.", "elena_9f3a", settings)

    assert audio.sample_rate == 24_000
    assert model.prepared[1] == 0.7  # exaggeration passed to prepare_conditionals
    assert eng.conds_path("elena_9f3a").is_file()  # conds were cached
    call = model.gen_calls[0]
    # only the generate tunables forwarded — no seed/speed leak into generate()
    assert call["kwargs"] == {"exaggeration": 0.7, "cfg_weight": 0.4, "temperature": 0.9}
    assert call["seed"] == 123  # seeded BEFORE generate (frozen cache key correctness)


def test_cached_conds_are_loaded_not_reprepared(tmp_path, monkeypatch):
    voices = _voice(tmp_path)
    model = _FakeModel()
    eng = ChatterboxEngine(device="cpu", voices_dir=voices, model=model)
    eng.conds_path("elena_9f3a").write_text("cached", encoding="utf-8")  # pretend cache exists
    monkeypatch.setattr(eng, "_load_conds", lambda path: "LOADED")

    eng.synthesize("Hi.", "elena_9f3a", {"seed": 1})

    assert model.conds == "LOADED" and model.prepared is None  # loaded, not re-prepared


def test_missing_reference_raises(tmp_path):
    voices = _voice(tmp_path, with_reference=False)
    eng = ChatterboxEngine(device="cpu", voices_dir=voices, model=_FakeModel())
    with pytest.raises(SynthesisError, match="no reference.wav"):
        eng.synthesize("Hi.", "elena_9f3a", {"seed": 1})


def test_unload_drops_model(tmp_path):
    eng = ChatterboxEngine(device="cpu", voices_dir=tmp_path, model=_FakeModel())
    eng.unload()
    assert eng._model is None


def test_get_engine_resolves_chatterbox(tmp_path):
    eng = get_engine("chatterbox", voices_dir=tmp_path, model=_FakeModel())
    assert isinstance(eng, ChatterboxEngine) and eng.engine_id == "chatterbox"
