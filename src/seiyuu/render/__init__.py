"""Render stage: normalized JSON + TTS engine → cached segments + manifest."""

from seiyuu.render.align import ensure_words
from seiyuu.render.cache import SegmentCache, SegmentKey, words_sidecar_for_wav
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
    RENDER_MODES,
    CostEstimate,
    RenderError,
    RenderResult,
    estimate_render_cost,
    estimate_render_cost_single,
    manifest_mode,
    manifest_name_for_mode,
    preserve_unarchived_manifest,
    render_book,
    render_book_multivoice,
)

__all__ = [
    "MANIFEST_NAME",
    "RENDER_MODES",
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
    "ensure_words",
    "estimate_render_cost",
    "estimate_render_cost_single",
    "hash_assignment",
    "issue_quote",
    "manifest_mode",
    "manifest_name_for_mode",
    "preserve_unarchived_manifest",
    "render_book",
    "render_book_multivoice",
    "verify_quote",
    "words_sidecar_for_wav",
]
