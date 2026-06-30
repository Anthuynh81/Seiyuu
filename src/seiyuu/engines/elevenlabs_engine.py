"""ElevenLabs cloud TTS adapter (premium final renders).

The SDK is imported lazily so listing/cost/key-checks never need it. Output is requested as
`pcm_24000` — headerless 16-bit little-endian PCM at 24 kHz, which IS the canonical rate, so no
resample happens. The `voice` argument is always a CLOUD voice id (a stock voice id for preset
voices, or the IVC handle the slot manager resolves for cloned voices); this adapter never
touches the voice library or creates voices — that is the slot manager's job (voices/cloud.py).

PAID: every call here costs money, so synthesis only runs inside the render path's explicit
cost gate. `requires_validation` is False — re-synthesizing on a whisper miss would silently
spend money; whisper stays an opt-in report for this engine.
"""

from typing import Any

import numpy as np

from seiyuu.engines.audio import CANONICAL_SAMPLE_RATE
from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine

# Per-segment settings forwarded to ElevenLabs VoiceSettings (all part of the frozen
# settings_hash, so changing one re-renders that segment).
_VOICE_SETTING_KEYS = ("stability", "similarity_boost", "style", "use_speaker_boost", "speed")


class ElevenLabsEngine(TTSEngine):
    engine_id = "elevenlabs"
    requires_validation = False  # paid: never auto-retry; whisper is an opt-in report only
    uses_gpu = False  # cloud API; never touches the local GPU

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
        price_per_1k_chars: float | None = None,
        client: Any | None = None,  # injectable for tests
    ) -> None:
        if api_key is None or model_id is None or price_per_1k_chars is None:
            from seiyuu.settings import get_settings

            cfg = get_settings()
            api_key = api_key or cfg.elevenlabs_api_key
            model_id = model_id or cfg.elevenlabs_model_id
            price_per_1k_chars = (
                cfg.elevenlabs_price_per_1k_chars
                if price_per_1k_chars is None
                else price_per_1k_chars
            )
        self._api_key = api_key
        self.model_id = model_id
        self.price_per_1k_chars = price_per_1k_chars
        self._client = client

    @property
    def model_version(self) -> str:
        # Part of the SegmentKey: switching the ElevenLabs model invalidates cached cloud audio.
        return f"elevenlabs-{self.model_id}"

    @property
    def native_sample_rate(self) -> int:
        return CANONICAL_SAMPLE_RATE  # pcm_24000 == canonical, no resample

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise SynthesisError(
                    "ELEVENLABS_API_KEY not set; required for the elevenlabs engine"
                )
            from elevenlabs.client import ElevenLabs

            self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    @property
    def client(self) -> Any:
        """The authenticated SDK client (the cloud-voice/slot manager shares it)."""
        return self._get_client()

    def list_voices(self) -> list[EngineVoice]:
        resp = self._get_client().voices.get_all()
        return [EngineVoice(id=v.voice_id, name=v.name) for v in resp.voices]

    def cost_estimate(self, text: str) -> float:
        """Estimated USD for synthesizing `text` (ElevenLabs bills per character)."""
        return len(text) / 1000 * self.price_per_1k_chars

    def _voice_settings(self, settings: dict[str, Any]) -> Any | None:
        kwargs = {k: settings[k] for k in _VOICE_SETTING_KEYS if k in settings}
        if not kwargs:
            return None
        from elevenlabs.types.voice_settings import VoiceSettings

        return VoiceSettings(**kwargs)

    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[np.ndarray, int]:
        client = self._get_client()
        seed = settings.get("seed")
        try:
            chunks = client.text_to_speech.convert(
                voice,
                text=text,
                model_id=self.model_id,
                output_format="pcm_24000",
                voice_settings=self._voice_settings(settings),
                **({"seed": seed} if seed is not None else {}),
            )
            data = b"".join(chunks)
        except Exception as exc:
            raise SynthesisError(f"elevenlabs synthesis failed (voice={voice}): {exc}") from exc
        if not data:
            raise SynthesisError(f"elevenlabs returned no audio (voice={voice})")
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        return samples, CANONICAL_SAMPLE_RATE
