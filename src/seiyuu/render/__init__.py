"""Render stage: normalized JSON + TTS engine → cached segments + manifest."""

from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.render.gate import (
    CostGateError,
    CostQuote,
    check_ceiling,
    hash_assignment,
    issue_quote,
    verify_quote,
)
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest, VoiceUse
from seiyuu.render.pipeline import (
    MANIFEST_NAME,
    CostEstimate,
    RenderError,
    RenderResult,
    estimate_render_cost,
    estimate_render_cost_single,
    render_book,
    render_book_multivoice,
)

__all__ = [
    "MANIFEST_NAME",
    "CostEstimate",
    "CostGateError",
    "CostQuote",
    "RenderError",
    "RenderManifest",
    "RenderResult",
    "RenderedChapter",
    "RenderedSegment",
    "SegmentCache",
    "SegmentKey",
    "VoiceUse",
    "check_ceiling",
    "estimate_render_cost",
    "estimate_render_cost_single",
    "hash_assignment",
    "issue_quote",
    "render_book",
    "render_book_multivoice",
    "verify_quote",
]
