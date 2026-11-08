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

from seiyuu.engines.audio import AudioFile, silence, to_canonical

# A line with nothing to pronounce ("……", "───", "♪♪", an emoji) renders as this much silence
# instead of reaching an engine: Kokoro returns no audio for it and LLM-style engines may invent
# speech. The assembler's gaps on both sides and the neighbours' own clip edges supply most of
# the beat: at 0.6 s a Kokoro beat totalled ~2.6 s (as long as a scene break, the longest
# silences in a chapter); at 0.2 s it is ~1.7-2.2 s on Kokoro (~0.8 s of clip edges) and
# ~1.1 s on IndexTTS (~0.2 s), still clearly longer than a plain join.
SILENT_BEAT_SECONDS = 0.2


def nothing_to_say(text: str) -> bool:
    """True when ``text`` has no letter or digit in any script, so there is nothing to speak."""
    return not any(ch.isalnum() for ch in text)


class EngineVoice(BaseModel):
    id: str
    name: str
    language: str | None = None
    gender: str | None = None
    description: str | None = None  # short editorial character note for pickers/mixers


class SynthesisError(Exception):
    """Loud synthesis failure; the render stage adds book/chapter/block context."""


class TTSEngine(ABC):
    engine_id: str
    # LLM-style engines (autoregressive: Chatterbox, Fish) can hallucinate/drop/repeat words,
    # so their output must pass whisper validation before assembly. Deterministic engines
    # (Kokoro) set this False and the render loop skips validation for them.
    requires_validation: bool = False
    # Local engines hold a model on the single GPU and synthesize through the GPU manager;
    # cloud engines (ElevenLabs) set this False so they don't needlessly unload a resident
    # local model while waiting on the network.
    uses_gpu: bool = True
    # Cloning engines resolve a voice from voices/{voice_id}/reference.wav, so they must be
    # constructed with the voice-library dir (and the consent gate binds to those exact bytes —
    # see api/registry). The pipeline/CLI pass voices_dir iff this is True; preset/cloud engines
    # (Kokoro, ElevenLabs) leave it False. Chatterbox, IndexTTS-2 and Qwen3-TTS set it True.
    clones_from_library: bool = False

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

    def unload(self) -> None:  # noqa: B027 — intentional no-op default; cloud engines override nothing
        """Free GPU memory (GpuConsumer protocol). No-op default; cloud engines need nothing."""

    def warm(self) -> None:  # noqa: B027 — intentional no-op default; cloud engines have no weights
        """Pre-load model weights so the first synthesis is fast (M6b warmup job).
        No-op default; local engines override to trigger their lazy load."""

    def synthesize(
        self, text: str, voice: str, settings: dict[str, Any] | None = None
    ) -> AudioFile:
        """Synthesize text with a voice; always returns canonical audio."""
        if not text or not text.strip():
            raise SynthesisError(f"{self.engine_id}: refusing to synthesize empty text")
        if nothing_to_say(text):
            return silence(SILENT_BEAT_SECONDS)
        samples, sample_rate = self._synthesize_native(text, voice, settings or {})
        return to_canonical(samples, sample_rate)

    @abstractmethod
    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[np.ndarray | torch.Tensor, int]:
        """Engine-specific synthesis; returns (samples, native_sample_rate)."""
