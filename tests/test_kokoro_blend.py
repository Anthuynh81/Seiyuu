"""Kokoro blends: pure recipe canonicalization + the engine's weighted-voicepack path."""

import torch

from seiyuu.engines.kokoro_engine import KokoroEngine
from seiyuu.voices.blends import auto_blend_recipe, canonical_recipe
from seiyuu.voices.models import BlendComponent

# --- pure recipe canonicalization (no torch needed, but torch is already imported) ---


def test_canonical_recipe_normalizes_rounds_and_sorts():
    r = canonical_recipe([("af_jessica", 3.0), ("af_bella", 1.0)])
    assert r == [("af_bella", 0.25), ("af_jessica", 0.75)]  # sorted by id, normalized to 1


def test_canonical_recipe_accepts_blend_components():
    r = canonical_recipe(
        [
            BlendComponent(preset_id="af_bella", weight=1),
            BlendComponent(preset_id="af_sky", weight=1),
        ]
    )
    assert r == [("af_bella", 0.5), ("af_sky", 0.5)]


def test_auto_blend_is_deterministic_same_family_and_two_components():
    a = auto_blend_recipe("Elizabeth", "female")
    b = auto_blend_recipe("Elizabeth", "female")
    assert a == b  # deterministic -> stable cache + reproducible `assign`
    assert len(a) == 2
    assert {p[:2] for p, _ in a} == {"af"}  # American female family
    assert abs(sum(w for _, w in a) - 1.0) < 0.01


def test_auto_blend_gender_picks_family():
    assert all(p.startswith("am") for p, _ in auto_blend_recipe("Darcy", "male"))
    assert all(p.startswith("af") for p, _ in auto_blend_recipe("Jane", "female"))


# --- engine path with a mocked KPipeline ---


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def load_single_voice(self, preset):
        return torch.ones(4) * (1.0 if preset == "af_bella" else 3.0)

    def __call__(self, text, voice=None, speed=1.0):
        self.calls.append({"voice": voice, "speed": speed})

        class _R:
            audio = torch.zeros(2400)

        yield _R()


def test_blend_builds_weighted_voicepack_and_skips_preset_check():
    eng = KokoroEngine(device="cpu")
    fake = _FakePipeline()
    eng._pipelines["a"] = fake  # pre-seed so no SDK import
    settings = {"blend": [["af_bella", 0.75], ["af_jessica", 0.25]], "seed": 1}

    # voice_id "lizzy_ab12" is NOT a preset, but a blend recipe is present -> allowed.
    audio = eng.synthesize("Hello.", "lizzy_ab12", settings)

    assert audio.sample_rate == 24_000
    sent = fake.calls[0]["voice"]
    assert torch.is_tensor(sent)
    # 0.75 * 1.0 + 0.25 * 3.0 = 1.5
    assert torch.allclose(sent, torch.full((4,), 1.5))


def test_blend_voicepack_is_memoized():
    eng = KokoroEngine(device="cpu")
    fake = _FakePipeline()
    eng._pipelines["a"] = fake
    recipe = [("af_bella", 0.5), ("af_jessica", 0.5)]
    first = eng._blend_voicepack("a", recipe)
    second = eng._blend_voicepack("a", recipe)
    assert first is second  # cached, not rebuilt


def test_non_preset_voice_without_blend_still_errors():
    from seiyuu.engines.base import SynthesisError

    eng = KokoroEngine(device="cpu")
    eng._pipelines["l"] = _FakePipeline()
    try:
        eng.synthesize("Hi.", "lizzy_ab12", {"seed": 1})
    except SynthesisError as e:
        assert "unknown voice" in str(e)
    else:
        raise AssertionError("expected SynthesisError for a non-preset voice with no blend")
