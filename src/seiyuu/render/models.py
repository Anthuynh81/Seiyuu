"""Render manifest: the documented contract between render and assembly.

Scene breaks appear as entries with wav=None — they are never synthesized;
assembly turns them into long pauses.
"""

from pydantic import BaseModel, model_validator

from seiyuu.ingest.models import BlockType


class RenderedSegment(BaseModel):
    block_id: str
    type: BlockType
    wav: str | None = None  # path relative to the book's output dir
    duration_seconds: float = 0.0

    @model_validator(mode="after")
    def _check_invariants(self) -> "RenderedSegment":
        if self.type is BlockType.SCENE_BREAK and self.wav is not None:
            raise ValueError(f"scene_break {self.block_id} must not carry audio")
        if self.type is not BlockType.SCENE_BREAK and self.wav is None:
            raise ValueError(f"{self.type} segment {self.block_id} is missing its wav")
        return self


class RenderedChapter(BaseModel):
    index: int  # 1-based chapter index within the normalized book
    title: str
    segments: list[RenderedSegment]


class RenderManifest(BaseModel):
    book_id: str
    engine: str
    engine_model_version: str
    voice_id: str
    settings: dict
    seed: int | None
    chapters: list[RenderedChapter]
