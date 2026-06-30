"""Render stage: normalized JSON + TTS engine → cached segments + manifest."""

from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest, VoiceUse
from seiyuu.render.pipeline import (
    MANIFEST_NAME,
    RenderError,
    RenderResult,
    render_book,
    render_book_multivoice,
)

__all__ = [
    "MANIFEST_NAME",
    "RenderError",
    "RenderManifest",
    "RenderResult",
    "RenderedChapter",
    "RenderedSegment",
    "SegmentCache",
    "SegmentKey",
    "VoiceUse",
    "render_book",
    "render_book_multivoice",
]
