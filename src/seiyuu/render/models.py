"""Render manifest: the documented contract between render and assembly.

Scene breaks appear as entries with wav=None — they are never synthesized; assembly turns
them into pauses. M3 multi-voice: per-segment voice identity moves onto RenderedSegment
(additive, optional so the single-voice path stays byte-compatible); manifest-level engine/
voice fields become optional and a per-voice provenance map + assignment snapshot are added.
One paragraph block can now yield several segments (different speakers), so assembly pauses
on block_id transitions, not every segment.
"""

from pydantic import BaseModel, model_validator

from seiyuu.ingest.models import BlockType
from seiyuu.validate import ValidationResult


class RenderedSegment(BaseModel):
    block_id: str
    type: BlockType
    wav: str | None = None  # path relative to the book's output dir
    duration_seconds: float = 0.0
    # M3 multi-voice (optional; populated by both render paths, None on scene_break):
    voice_id: str | None = None
    seed: int | None = None
    settings_hash: str | None = None
    # M4 validation (optional; only LLM-style engines validate, so None for Kokoro/scene_break):
    validation: ValidationResult | None = None
    synth_attempts: int = 1  # synth tries this render (>1 means validation forced retries)

    @model_validator(mode="after")
    def _check_invariants(self) -> "RenderedSegment":
        if self.type is BlockType.SCENE_BREAK:
            if self.wav is not None:
                raise ValueError(f"scene_break {self.block_id} must not carry audio")
            if self.voice_id is not None:
                raise ValueError(f"scene_break {self.block_id} must not carry a voice")
        elif self.wav is None:
            raise ValueError(f"{self.type} segment {self.block_id} is missing its wav")
        return self


class RenderedChapter(BaseModel):
    index: int  # 1-based chapter index within the normalized book
    title: str
    segments: list[RenderedSegment]


class VoiceUse(BaseModel):
    """Provenance for a voice used in a render (manifest voices_used map value)."""

    engine: str
    engine_model_version: str
    kind: str


class RenderManifest(BaseModel):
    book_id: str
    book_title: str | None = None  # for player metadata (album tag)
    # Single-voice fields (optional; None on a multi-voice render, where per-segment + the
    # voices_used map carry the truth):
    engine: str | None = None
    engine_model_version: str | None = None
    voice_id: str | None = None
    settings: dict | None = None
    seed: int | None = None
    chapters: list[RenderedChapter]
    # Multi-voice provenance:
    voices_used: dict[str, VoiceUse] = {}  # voice_id -> engine/model/kind
    assignment: dict | None = None  # VoiceAssignment snapshot
    # M4 validation: count of validated segments that failed whisper (0 == all clean / none
    # validated). Per-segment detail lives on RenderedSegment.validation.
    validation_failures: int = 0
