"""ElevenLabs adapter: pcm_24000 → canonical audio, cost, voices, settings — SDK mocked."""

from types import SimpleNamespace

import numpy as np
import pytest

from seiyuu.engines import get_engine
from seiyuu.engines.base import SynthesisError
from seiyuu.engines.elevenlabs_engine import ElevenLabsEngine


def _pcm_chunks(n_samples=2400):
    t = np.arange(n_samples) / 24_000
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32_767).astype("<i2").tobytes()
    half = len(pcm) // 2
    yield pcm[:half]  # ElevenLabs streams the audio in chunks
    yield pcm[half:]


class FakeTTS:
    def __init__(self):
        self.calls = []

    def convert(self, voice_id, **kwargs):
        self.calls.append({"voice_id": voice_id, **kwargs})
        return _pcm_chunks()


class FakeVoices:
    def __init__(self):
        self.deleted = []

    def get_all(self, **kwargs):
        return SimpleNamespace(
            voices=[
                SimpleNamespace(voice_id="EXAVITQu", name="Rachel"),
                SimpleNamespace(voice_id="21m00", name="Adam"),
            ]
        )

    def delete(self, voice_id, **kwargs):
        self.deleted.append(voice_id)


class FakeClient:
    def __init__(self):
        self.text_to_speech = FakeTTS()
        self.voices = FakeVoices()


def _engine(client=None):
    return ElevenLabsEngine(
        api_key="k",
        model_id="eleven_multilingual_v2",
        price_per_1k_chars=0.30,
        client=client or FakeClient(),
    )


def test_synthesize_is_canonical_and_forwards_request():
    eng = _engine()
    audio = eng.synthesize("Hello there.", "EXAVITQu", {"seed": 7})
    assert audio.sample_rate == 24_000
    assert audio.samples.ndim == 1 and len(audio.samples) == 2400
    call = eng._client.text_to_speech.calls[0]
    assert call["voice_id"] == "EXAVITQu"
    assert call["seed"] == 7
    assert call["output_format"] == "pcm_24000"
    assert call["model_id"] == "eleven_multilingual_v2"


def test_voice_settings_built_from_settings():
    eng = _engine()
    eng.synthesize("Hi.", "v", {"stability": 0.4, "similarity_boost": 0.8})
    vs = eng._client.text_to_speech.calls[0]["voice_settings"]
    assert vs is not None
    assert vs.stability == 0.4 and vs.similarity_boost == 0.8


def test_no_voice_settings_when_none_given():
    eng = _engine()
    eng.synthesize("Hi.", "v", {"seed": 1})
    assert eng._client.text_to_speech.calls[0]["voice_settings"] is None


def test_cost_estimate_per_character():
    eng = _engine()
    assert eng.cost_estimate("a" * 1000) == pytest.approx(0.30)
    assert eng.cost_estimate("") == 0.0


def test_list_voices_maps_account_voices():
    eng = _engine()
    voices = eng.list_voices()
    assert [v.id for v in voices] == ["EXAVITQu", "21m00"]
    assert voices[0].name == "Rachel"


def test_model_version_and_validation_flag():
    eng = _engine()
    assert eng.model_version == "elevenlabs-eleven_multilingual_v2"
    assert eng.requires_validation is False  # paid: no auto-retry


def test_missing_api_key_is_loud():
    eng = ElevenLabsEngine(api_key=None, model_id="m", price_per_1k_chars=0.3, client=None)
    with pytest.raises(SynthesisError, match="ELEVENLABS_API_KEY"):
        eng.synthesize("hi", "v", {})


def test_empty_audio_is_loud():
    client = FakeClient()
    client.text_to_speech.convert = lambda voice_id, **kw: iter([])
    with pytest.raises(SynthesisError, match="no audio"):
        _engine(client=client).synthesize("hi", "v", {})


def test_factory_builds_engine():
    eng = get_engine("elevenlabs", api_key="k", client=FakeClient())
    assert eng.engine_id == "elevenlabs"
