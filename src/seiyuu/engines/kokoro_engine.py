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

# Editorial character notes for pickers/mixers — subjective by nature, tuned by ear.
_DESCRIPTIONS = {
    "af_alloy": "even, matter-of-fact",
    "af_aoede": "light, youthful lilt",
    "af_bella": "bright and expressive — big range",
    "af_heart": "warm, rounded — the default narrator",
    "af_jessica": "soft and gentle",
    "af_kore": "clear, poised",
    "af_nicole": "breathy, close-mic whisper",
    "af_nova": "crisp, energetic",
    "af_river": "relaxed, airy",
    "af_sarah": "steady, clean read",
    "af_sky": "light with a bright edge",
    "am_adam": "deep and forceful",
    "am_echo": "mellow, low-key",
    "am_eric": "plain mid-range",
    "am_fenrir": "gravelly intensity",
    "am_liam": "smooth, younger",
    "am_michael": "warm, avuncular",
    "am_onyx": "very deep, resonant",
    "am_puck": "wry, energetic",
    "am_santa": "jolly theatrical bass",
    "bf_alice": "bright RP, precise",
    "bf_emma": "warm British, rounded",
    "bf_isabella": "soft-spoken RP",
    "bf_lily": "light, youthful British",
    "bm_daniel": "measured RP baritone",
    "bm_fable": "storyteller's RP — crisp consonants",
    "bm_george": "deep, formal RP",
    "bm_lewis": "relaxed British mid-depth",
}


def _voice_meta(preset: str) -> EngineVoice:
    return EngineVoice(
        id=preset,
        name=preset.split("_", 1)[1].title(),
        language={"a": "en-US", "b": "en-GB"}[preset[0]],
        gender={"f": "female", "m": "male"}[preset[1]],
        description=_DESCRIPTIONS.get(preset),
    )


class KokoroEngine(TTSEngine):
    engine_id = "kokoro"

    def __init__(self, device: str | None = None) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipelines: dict[str, Any] = {}  # lang_code -> KPipeline
        self._blend_cache: dict[tuple, Any] = {}  # canonical recipe -> weighted voicepack

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

    def warm(self) -> None:
        """Load the default (American English) pipeline — the weights are shared, so any
        lang_code pulls the full model; blends/voicepacks stay lazy per voice."""
        self._pipeline("a")

    def unload(self) -> None:
        """Drop loaded KPipelines and free VRAM (GPU resource manager handoff)."""
        import gc

        self._pipelines.clear()
        self._blend_cache.clear()
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

    def _blend_voicepack(self, lang_code: str, recipe: list[tuple[str, float]]) -> Any:
        """Weighted sum of preset voicepacks (normalized), memoized by canonical recipe."""
        key = tuple(recipe)
        if key in self._blend_cache:
            return self._blend_cache[key]
        pipeline = self._pipeline(lang_code)
        total = sum(w for _, w in recipe) or 1.0
        pack = None
        for preset_id, weight in recipe:
            contribution = pipeline.load_single_voice(preset_id) * (weight / total)
            pack = contribution if pack is None else pack + contribution
        self._blend_cache[key] = pack
        return pack

    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[torch.Tensor, int]:
        blend = settings.get("blend")
        if blend:  # a weighted Kokoro blend (recipe folded into settings by render)
            recipe = [(str(p), float(w)) for p, w in blend]
            families = {p[:1] for p, _ in recipe}
            if len(families) != 1:
                raise SynthesisError(f"kokoro: blend mixes language families {sorted(families)}")
            lang_code = recipe[0][0][0]
            voice_arg: Any = self._blend_voicepack(lang_code, recipe)
        else:
            if voice not in _PRESETS:
                raise SynthesisError(
                    f"kokoro: unknown voice {voice!r}; known presets: {', '.join(_PRESETS)}"
                )
            lang_code, voice_arg = voice[0], voice

        seed = settings.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))  # Kokoro is deterministic; seeded for safety
        speed = float(settings.get("speed", 1.0))

        pipeline = self._pipeline(lang_code)
        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for result in pipeline(text, voice=voice_arg, speed=speed):
                if result.audio is not None:
                    chunks.append(result.audio.detach().cpu())
        if not chunks:
            raise SynthesisError(f"kokoro: produced no audio (text: {text[:80]!r})")
        return torch.cat(chunks), KOKORO_SAMPLE_RATE
