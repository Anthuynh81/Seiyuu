"""TTSEngine interface (SPEC provider lineup).

Engine SDKs live ONLY inside seiyuu/engines behind this interface; pipeline
code never imports an engine SDK directly. The public synthesize() is a
template method that forces every adapter's output through to_canonical(), so
non-canonical audio cannot structurally reach later stages.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

from seiyuu.engines.audio import AudioFile, to_canonical


class EngineVoice(BaseModel):
    id: str
    name: str
    language: str | None = None
    gender: str | None = None


class SynthesisError(Exception):
    """Loud synthesis failure; the render stage adds book/chapter/block context."""


class TTSEngine(ABC):
    engine_id: str

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Part of the TTS segment cache key; must change when output would."""

    @property
    @abstractmethod
    def native_sample_rate(self) -> int: ...

    @abstractmethod
    def list_voices(self) -> list[EngineVoice]: ...

    @abstractmethod
    def cost_estimate(self, text: str) -> float:
        """Estimated cost in USD; 0.0 for local engines."""

    def prepare_voice(self, reference_wav: Path) -> None:
        """Optional: build engine-side voice assets from a reference clip (M3+)."""
        raise NotImplementedError(f"{self.engine_id} does not support voice cloning")

    def synthesize(
        self, text: str, voice: str, settings: dict[str, Any] | None = None
    ) -> AudioFile:
        """Synthesize text with a voice; always returns canonical audio."""
        if not text or not text.strip():
            raise SynthesisError(f"{self.engine_id}: refusing to synthesize empty text")
        samples, sample_rate = self._synthesize_native(text, voice, settings or {})
        return to_canonical(samples, sample_rate)

    @abstractmethod
    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[np.ndarray | torch.Tensor, int]:
        """Engine-specific synthesis; returns (samples, native_sample_rate)."""
