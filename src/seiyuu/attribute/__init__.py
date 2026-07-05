"""Attribution stage: normalized JSON + LLM provider → attributed segments + registry.

Provider SDKs live ONLY under ``attribute/providers/`` behind the ``AttributionLLM``
interface; pipeline code never imports an LLM SDK directly.
"""

from seiyuu.attribute.cache import AdjudicationCacheKey, AttributionCache, ChunkCacheKey
from seiyuu.attribute.models import (
    AdjudicationResult,
    AttributedChapter,
    AttributionReport,
    CandidatePair,
    Character,
    CharacterEvidence,
    CharacterMention,
    CharacterRegistry,
    ChunkAttribution,
    FlaggedBlock,
    PairVerdict,
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
    "AdjudicationCacheKey",
    "AdjudicationResult",
    "AttributedChapter",
    "AttributionCache",
    "AttributionError",
    "AttributionReport",
    "CandidatePair",
    "Character",
    "CharacterEvidence",
    "CharacterMention",
    "CharacterRegistry",
    "ChunkAttribution",
    "ChunkCacheKey",
    "FlaggedBlock",
    "PairVerdict",
    "Segment",
    "SegmentType",
    "attribute_book",
    "write_attribution",
]
