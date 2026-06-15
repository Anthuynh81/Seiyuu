"""Attribution stage: normalized JSON + LLM provider → attributed segments + registry.

Provider SDKs live ONLY under ``attribute/providers/`` behind the ``AttributionLLM``
interface; pipeline code never imports an LLM SDK directly.
"""

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

__all__ = [
    "AttributedChapter",
    "AttributionReport",
    "Character",
    "CharacterMention",
    "CharacterRegistry",
    "ChunkAttribution",
    "FlaggedBlock",
    "Segment",
    "SegmentType",
]
