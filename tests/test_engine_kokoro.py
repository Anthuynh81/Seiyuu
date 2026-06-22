"""Kokoro adapter tests. CPU tests never load model weights; real synthesis
is behind the gpu marker (run with `uv run pytest -m gpu`)."""

import numpy as np
import pytest
import soundfile as sf

from seiyuu.engines import CANONICAL_SAMPLE_RATE, SynthesisError, get_engine
from seiyuu.engines.kokoro_engine import KokoroEngine


def test_factory_returns_kokoro() -> None:
    engine = get_engine("kokoro")
    assert isinstance(engine, KokoroEngine)
    assert engine.engine_id == "kokoro"


def test_factory_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unknown TTS engine"):
        get_engine("does-not-exist")


def test_voice_listing_metadata() -> None:
    voices = {v.id: v for v in get_engine("kokoro").list_voices()}
    assert "af_heart" in voices
    assert voices["af_heart"].language == "en-US"
    assert voices["af_heart"].gender == "female"
    assert voices["bm_george"].language == "en-GB"
    assert voices["bm_george"].gender == "male"


def test_engine_metadata() -> None:
    engine = get_engine("kokoro")
    assert engine.native_sample_rate == 24_000
    assert engine.cost_estimate("any text") == 0.0
    assert engine.model_version.startswith("kokoro-")


def test_empty_text_rejected_without_model_load() -> None:
    with pytest.raises(SynthesisError, match="empty text"):
        get_engine("kokoro").synthesize("   ", "af_heart")


def test_unknown_voice_rejected_without_model_load() -> None:
    with pytest.raises(SynthesisError, match="unknown voice"):
        get_engine("kokoro").synthesize("Hello there.", "xx_nobody")


@pytest.mark.gpu
def test_kokoro_smoke_synthesis(tmp_path) -> None:
    engine = get_engine("kokoro")
    audio = engine.synthesize(
        "The quick brown fox jumps over the lazy dog.",
        "af_heart",
        settings={"seed": 41172},
    )
    assert audio.sample_rate == CANONICAL_SAMPLE_RATE
    assert 1.0 < audio.duration_seconds < 8.0
    rms = float(np.sqrt(np.mean(audio.samples**2)))
    assert rms > 0.01, "synthesized audio is near-silent"

    path = audio.save(tmp_path / "smoke.wav")
    info = sf.info(str(path))
    assert info.samplerate == CANONICAL_SAMPLE_RATE
    assert info.channels == 1
    assert info.subtype == "PCM_16"
