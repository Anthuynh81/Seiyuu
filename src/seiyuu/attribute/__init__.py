"""Attribution stage: normalized JSON + LLM provider → attributed segments + registry.

Provider SDKs live ONLY under ``attribute/providers/`` behind the ``AttributionLLM``
interface; pipeline code never imports an LLM SDK directly.
"""

from seiyuu.attribute.cache import AttributionCache, ChunkCacheKey
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterMention,
    CharacterRegistry,
    ChunkAttribution,
    FlaggedBlock,
    Segment,
    SegmentType,
)
from seiyuu.attribute.pipeline import (
    ATTRIBUTION_NAME,
    AttributionError,
    attribute_book,
    write_attribution,
)

__all__ = [
    "ATTRIBUTION_NAME",
    "AttributedChapter",
    "AttributionCache",
    "AttributionError",
    "AttributionReport",
    "Character",
    "CharacterMention",
    "CharacterRegistry",
    "ChunkAttribution",
    "ChunkCacheKey",
    "FlaggedBlock",
    "Segment",
    "SegmentType",
    "attribute_book",
    "write_attribution",
]
