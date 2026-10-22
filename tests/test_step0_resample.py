"""to_canonical resampling (soxr) keeps length, dtype, range, pitch and loudness."""

import numpy as np
import pytest
import torch

from seiyuu.engines import CANONICAL_SAMPLE_RATE, to_canonical


def sine(seconds: float, sample_rate: int, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_hz(samples: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


@pytest.mark.parametrize("in_rate", [22_050, 44_100])
def test_resample_length_is_canonical(in_rate: int) -> None:
    src = sine(1.0, in_rate)
    audio = to_canonical(src, in_rate)
    expected = len(src) * CANONICAL_SAMPLE_RATE / in_rate
    assert audio.sample_rate == CANONICAL_SAMPLE_RATE
    assert abs(len(audio.samples) - expected) <= 1


@pytest.mark.parametrize("in_rate", [22_050, 44_100])
def test_resample_preserves_dtype_shape_and_range(in_rate: int) -> None:
    audio = to_canonical(sine(0.5, in_rate, amp=0.99), in_rate)
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert float(np.max(np.abs(audio.samples))) <= 1.0


@pytest.mark.parametrize("in_rate", [22_050, 44_100])
def test_resample_keeps_frequency_and_rms(in_rate: int) -> None:
    src = sine(1.0, in_rate, freq=1_000.0)
    audio = to_canonical(src, in_rate)
    # 1 s of audio -> 1 Hz FFT bins.
    assert peak_hz(audio.samples, CANONICAL_SAMPLE_RATE) == pytest.approx(1_000.0, abs=2.0)
    assert rms(audio.samples) == pytest.approx(rms(src), rel=0.01)


def test_resample_torch_stereo_input() -> None:
    mono = sine(0.5, 44_100)
    stereo = torch.from_numpy(np.stack([mono, mono]))  # (2, n)
    audio = to_canonical(stereo, 44_100)
    assert audio.samples.dtype == np.float32
    assert abs(len(audio.samples) - 12_000) <= 1


def test_identity_when_rates_match() -> None:
    src = sine(0.5, CANONICAL_SAMPLE_RATE)
    audio = to_canonical(src, CANONICAL_SAMPLE_RATE)
    assert len(audio.samples) == len(src)
    np.testing.assert_array_equal(audio.samples, src)


def test_resample_clamps_overshoot() -> None:
    audio = to_canonical(sine(0.25, 22_050, amp=1.5), 22_050)
    assert float(np.max(np.abs(audio.samples))) <= 1.0
