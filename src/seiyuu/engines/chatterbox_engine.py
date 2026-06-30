"""Chatterbox-tts adapter: zero-shot voice cloning from a reference clip.

The SDK import is deferred into _get_model() so listing/cost/validation never load weights.
Cloning works by precomputing "conditionals" from voices/{voice_id}/reference.wav and caching
them to conds_{model_version}.pt; a voice switch reloads that small object instead of the
model, which is what makes multi-voice render cheap. Native output is already canonical
24 kHz. Chatterbox's generate() has NO seed parameter and samples (do_sample=True), so we MUST
seed torch right before every generate() or identical cache keys would map to different audio.
"""

import gc
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import torch

from seiyuu.engines.audio import CANONICAL_SAMPLE_RATE
from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine

# Per-segment tunables we forward to generate() (all part of the frozen settings_hash).
_GEN_KEYS = ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p")


class ChatterboxEngine(TTSEngine):
    engine_id = "chatterbox"

    def __init__(
        self,
        *,
        device: str | None = None,
        voices_dir: Path | None = None,
        model: Any | None = None,  # injectable for tests; lazy-loaded otherwise
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if voices_dir is None:
            from seiyuu.settings import get_settings

            voices_dir = get_settings().voices_dir
        self._voices_dir = Path(voices_dir)
        self._model = model

    @property
    def model_version(self) -> str:
        # This SAME string keys conds_{model_version}.pt AND SegmentKey.engine_model_version,
        # so a package bump invalidates both caches in lockstep.
        return f"chatterbox-{pkg_version('chatterbox-tts')}"

    @property
    def native_sample_rate(self) -> int:
        return CANONICAL_SAMPLE_RATE  # Chatterbox S3Gen is 24 kHz == canonical

    def list_voices(self) -> list[EngineVoice]:
        return []  # cloned voices live in the voice library, not in the engine

    def cost_estimate(self, text: str) -> float:
        return 0.0

    def _get_model(self) -> Any:
        if self._model is None:
            from chatterbox.tts import ChatterboxTTS  # SDK import stays inside the adapter

            self._model = ChatterboxTTS.from_pretrained(self._device)
        return self._model

    def conds_path(self, voice_id: str) -> Path:
        return self._voices_dir / voice_id / f"conds_{self.model_version}.pt"

    def _load_conds(self, path: Path) -> Any:  # seam: patched in tests, real SDK otherwise
        from chatterbox.tts import Conditionals

        return Conditionals.load(str(path), map_location=self._device)

    def _ensure_conds(self, model: Any, voice_id: str, exaggeration: float) -> None:
        path = self.conds_path(voice_id)
        if path.is_file():
            model.conds = self._load_conds(path)  # cheap voice switch
            return
        reference = self._voices_dir / voice_id / "reference.wav"
        if not reference.is_file():
            raise SynthesisError(
                f"chatterbox: no reference.wav for voice {voice_id!r} at {reference}"
            )
        model.prepare_conditionals(str(reference), exaggeration=exaggeration)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.conds.save(str(path))

    def prepare_voice(self, voice_id: str) -> None:  # type: ignore[override]
        """Precompute + cache conds for a voice (curation/audition warm-up)."""
        self._ensure_conds(self._get_model(), voice_id, 0.5)

    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[torch.Tensor, int]:
        model = self._get_model()
        self._ensure_conds(model, voice, float(settings.get("exaggeration", 0.5)))

        seed = settings.get("seed")
        if seed is not None:  # REQUIRED: generate() has no seed arg and samples
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        gen_kwargs = {k: settings[k] for k in _GEN_KEYS if k in settings}
        audio = model.generate(text, **gen_kwargs)  # (1, n) float tensor, 24 kHz
        return audio, CANONICAL_SAMPLE_RATE

    def unload(self) -> None:
        self._model = None  # drops t3/s3gen/ve/tokenizer/watermarker refs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
