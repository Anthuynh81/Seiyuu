"""Render stage: normalized JSON + TTS engine → cached segments + manifest."""

from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest
from seiyuu.render.pipeline import MANIFEST_NAME, RenderError, RenderResult, render_book

__all__ = [
    "MANIFEST_NAME",
    "RenderError",
    "RenderManifest",
    "RenderResult",
    "RenderedChapter",
    "RenderedSegment",
    "SegmentCache",
    "SegmentKey",
    "render_book",
]
