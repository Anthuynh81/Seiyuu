"""Canonical audio format utilities (SPEC audio policy).

Canonical intermediate: mono, 24,000 Hz, 16-bit PCM WAV. Every engine adapter
output passes through to_canonical(); mixed sample rates must never reach
assembly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

CANONICAL_SAMPLE_RATE = 24_000
CANONICAL_SUBTYPE = "PCM_16"


@dataclass(frozen=True)
class AudioFile:
    """Mono float32 audio at the canonical sample rate."""

    samples: np.ndarray  # shape (n,), float32, in [-1, 1]
    sample_rate: int = CANONICAL_SAMPLE_RATE

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    def save(self, path: Path) -> Path:
        """Write as canonical 16-bit PCM WAV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), self.samples, self.sample_rate, subtype=CANONICAL_SUBTYPE)
        return path


def to_canonical(samples: np.ndarray | torch.Tensor, sample_rate: int) -> AudioFile:
    """Any engine output → mono, 24 kHz, float32, clamped to [-1, 1].

    Accepts 1-D mono or 2-D audio in either (channels, n) or (n, channels)
    orientation, as numpy array or torch tensor.
    """
    if isinstance(samples, torch.Tensor):
        tensor = samples.detach().cpu().float()
    else:
        tensor = torch.from_numpy(np.asarray(samples, dtype=np.float32))

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        if tensor.shape[0] > tensor.shape[1]:  # (n, channels) → (channels, n)
            tensor = tensor.T
    else:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {tuple(tensor.shape)}")

    mono = tensor.mean(dim=0)
    if sample_rate != CANONICAL_SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sample_rate, CANONICAL_SAMPLE_RATE)
    mono = mono.clamp(-1.0, 1.0)
    return AudioFile(samples=mono.numpy().astype(np.float32), sample_rate=CANONICAL_SAMPLE_RATE)
