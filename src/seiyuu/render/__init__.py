"""Render stage: normalized JSON + TTS engine → cached segments + manifest."""

from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest, VoiceUse
from seiyuu.render.pipeline import (
    MANIFEST_NAME,
    CostEstimate,
    RenderError,
    RenderResult,
    estimate_render_cost,
    render_book,
    render_book_multivoice,
)

__all__ = [
    "MANIFEST_NAME",
    "CostEstimate",
    "RenderError",
    "RenderManifest",
    "RenderResult",
    "RenderedChapter",
    "RenderedSegment",
    "SegmentCache",
    "SegmentKey",
    "VoiceUse",
    "estimate_render_cost",
    "render_book",
    "render_book_multivoice",
]
