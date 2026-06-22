"""Kokoro-82M adapter: local preset voices. Blends arrive in M3.

The kokoro SDK import is deferred into _pipeline() so that listing voices,
cost estimates, and argument validation never load model weights.
"""

from importlib.metadata import version as pkg_version
from typing import Any

import torch

from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine

KOKORO_SAMPLE_RATE = 24_000
KOKORO_REPO = "hexgrad/Kokoro-82M"

# Official Kokoro v1.0 English presets: [a)merican|b)ritish][f)emale|m)ale]_name.
_PRESETS = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]  # fmt: skip


def _voice_meta(preset: str) -> EngineVoice:
    return EngineVoice(
        id=preset,
        name=preset.split("_", 1)[1].title(),
        language={"a": "en-US", "b": "en-GB"}[preset[0]],
        gender={"f": "female", "m": "male"}[preset[1]],
    )


class KokoroEngine(TTSEngine):
    engine_id = "kokoro"

    def __init__(self, device: str | None = None) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipelines: dict[str, Any] = {}  # lang_code -> KPipeline

    @property
    def model_version(self) -> str:
        return f"kokoro-{pkg_version('kokoro')}"

    @property
    def native_sample_rate(self) -> int:
        return KOKORO_SAMPLE_RATE

    def list_voices(self) -> list[EngineVoice]:
        return [_voice_meta(p) for p in _PRESETS]

    def cost_estimate(self, text: str) -> float:
        return 0.0

    def unload(self) -> None:
        """Drop loaded KPipelines and free VRAM (GPU resource manager handoff)."""
        import gc

        self._pipelines.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _pipeline(self, lang_code: str) -> Any:
        if lang_code not in self._pipelines:
            from kokoro import KPipeline  # SDK import stays inside the adapter

            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code, repo_id=KOKORO_REPO, device=self._device
            )
        return self._pipelines[lang_code]

    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[torch.Tensor, int]:
        if voice not in _PRESETS:
            raise SynthesisError(
                f"kokoro: unknown voice {voice!r}; known presets: {', '.join(_PRESETS)}"
            )
        seed = settings.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))  # Kokoro is deterministic; seeded for safety
        speed = float(settings.get("speed", 1.0))

        pipeline = self._pipeline(voice[0])
        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for result in pipeline(text, voice=voice, speed=speed):
                if result.audio is not None:
                    chunks.append(result.audio.detach().cpu())
        if not chunks:
            raise SynthesisError(
                f"kokoro: produced no audio for voice {voice!r} (text: {text[:80]!r})"
            )
        return torch.cat(chunks), KOKORO_SAMPLE_RATE
