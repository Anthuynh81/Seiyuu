import numpy as np
import pytest
import soundfile as sf
import torch

from seiyuu.engines import CANONICAL_SAMPLE_RATE, to_canonical


def sine(seconds: float, sample_rate: int, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_stereo_48k_to_canonical() -> None:
    mono = sine(0.5, 48_000)
    stereo = np.stack([mono, mono])  # (2, n)
    audio = to_canonical(stereo, 48_000)
    assert audio.sample_rate == CANONICAL_SAMPLE_RATE
    assert audio.samples.ndim == 1
    assert audio.samples.dtype == np.float32
    assert audio.duration_seconds == pytest.approx(0.5, abs=0.01)


def test_samples_by_channels_orientation() -> None:
    mono = sine(0.5, 24_000)
    audio = to_canonical(np.stack([mono, mono], axis=1), 24_000)  # (n, 2)
    assert audio.duration_seconds == pytest.approx(0.5, abs=0.01)


def test_mono_canonical_rate_passthrough() -> None:
    mono = sine(0.25, CANONICAL_SAMPLE_RATE)
    audio = to_canonical(mono, CANONICAL_SAMPLE_RATE)
    assert len(audio.samples) == len(mono)
    assert np.allclose(audio.samples, mono, atol=1e-6)


def test_torch_tensor_input() -> None:
    mono = torch.from_numpy(sine(0.25, 48_000))
    audio = to_canonical(mono, 48_000)
    assert audio.sample_rate == CANONICAL_SAMPLE_RATE
    assert audio.duration_seconds == pytest.approx(0.25, abs=0.01)


def test_clamping() -> None:
    loud = sine(0.1, CANONICAL_SAMPLE_RATE, amp=2.0)
    audio = to_canonical(loud, CANONICAL_SAMPLE_RATE)
    assert float(np.max(np.abs(audio.samples))) <= 1.0


def test_rejects_3d_audio() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        to_canonical(np.zeros((2, 3, 4), dtype=np.float32), 24_000)


def test_save_writes_canonical_pcm16_wav(tmp_path) -> None:
    audio = to_canonical(sine(0.3, 48_000), 48_000)
    path = audio.save(tmp_path / "nested" / "out.wav")
    info = sf.info(str(path))
    assert info.samplerate == CANONICAL_SAMPLE_RATE
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    loaded, sr = sf.read(str(path), dtype="float32")
    assert sr == CANONICAL_SAMPLE_RATE
    rms_in = float(np.sqrt(np.mean(audio.samples**2)))
    rms_out = float(np.sqrt(np.mean(loaded**2)))
    assert rms_out == pytest.approx(rms_in, rel=0.01)
