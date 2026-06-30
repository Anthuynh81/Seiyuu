"""Voice model — the on-disk truth for every voice (preset / blend / cloned).

One pydantic model discriminated by ``kind`` (mirrors how attribute.models uses Segment.type),
matching the SPEC meta.json illustration. ``voice_id`` is the directory name, the value
Characters reference, and the FROZEN SegmentKey.voice_id — never an engine/cloud id. For a
cloned voice ``reference.wav`` is the source of truth; conds/embeddings are disposable caches.
``settings`` is per-engine (``{engine: {tunable: value}}``) so one voice can carry tuned
settings for more than one engine without collision; render uses only the active engine's set.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class VoiceKind(StrEnum):
    PRESET = "preset"  # a single Kokoro preset
    BLEND = "blend"  # a weighted mix of same-accent Kokoro presets
    CLONED = "cloned"  # Chatterbox clone from reference.wav


def today_iso() -> str:
    return datetime.now(UTC).date().isoformat()


class BlendComponent(BaseModel):
    preset_id: str
    weight: float = Field(gt=0)  # normalized at render; only relative weights matter


class VoiceMeta(BaseModel):
    schema_version: int = 1
    voice_id: str
    name: str
    kind: VoiceKind
    engine: str  # 'kokoro' for preset/blend, 'chatterbox' for cloned (M3)
    preset_id: str | None = None  # kind=preset
    blend: list[BlendComponent] | None = None  # kind=blend
    reference_audio: str | None = None  # kind=cloned (e.g. 'reference.wav')
    settings: dict[str, dict[str, float]] = {}  # per-engine tunables -> settings_hash at render
    seed: int = 41172  # pinned per voice; renders must use it
    language: str | None = None
    consent_attested: bool = False  # required True before a cloned voice may be saved/rendered
    source: str = "user_upload"  # user_upload | preset | auto_blend | manual_blend
    created_at: str = Field(default_factory=today_iso)

    @model_validator(mode="after")
    def _check_kind(self) -> "VoiceMeta":
        if self.kind is VoiceKind.PRESET:
            if not self.preset_id:
                raise ValueError(f"voice {self.voice_id}: preset kind requires preset_id")
            if self.blend or self.reference_audio:
                raise ValueError(f"voice {self.voice_id}: preset kind must not set blend/reference")
        elif self.kind is VoiceKind.BLEND:
            if not self.blend or len(self.blend) < 2:
                raise ValueError(f"voice {self.voice_id}: blend kind needs >=2 components")
            families = {c.preset_id[:1] for c in self.blend}
            if len(families) != 1:
                raise ValueError(
                    f"voice {self.voice_id}: blend mixes language families {sorted(families)}"
                )
            if self.preset_id or self.reference_audio:
                raise ValueError(f"voice {self.voice_id}: blend kind must not set preset/reference")
        elif self.kind is VoiceKind.CLONED:
            if not self.reference_audio:
                raise ValueError(f"voice {self.voice_id}: cloned kind requires reference_audio")
            if self.preset_id or self.blend:
                raise ValueError(f"voice {self.voice_id}: cloned kind must not set preset/blend")
        return self

    def engine_settings(self) -> dict[str, float]:
        """The active engine's tunables (what render folds into the segment settings_hash)."""
        return dict(self.settings.get(self.engine, {}))
