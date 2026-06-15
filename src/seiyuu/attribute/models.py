"""Attribution models — the documented contract between attribution and later stages.

Schema (SPEC.md stage 2): ordered segments
``{type: narration|dialogue|thought, speaker, text, confidence, block_id}`` plus a
running character registry.

Two-phase ``speaker`` convention (intentional, documented):
- As returned by an ``AttributionLLM`` provider, ``Segment.speaker`` holds the raw
  speaker NAME the model wrote (or ``None`` for narration). The provider also surfaces
  per-chunk :class:`CharacterMention`s carrying character metadata.
- After registry resolution (``attribute/registry.py``), ``Segment.speaker`` holds a
  stable character ``id`` from the :class:`CharacterRegistry`. ``attribution.json`` always
  stores resolved segments.

Thoughts (internal monologue) are kept as a distinct segment type so the information is
never lost; which voice renders a thought (character vs. narrator vs. softened variant)
is decided later at voice assignment (M3), not here.
"""

from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator

from seiyuu.ingest.models import BLOCK_ID_PATTERN


class SegmentType(StrEnum):
    NARRATION = "narration"
    DIALOGUE = "dialogue"
    THOUGHT = "thought"


def _clean_optional(value: str | None) -> str | None:
    """Collapse empty/whitespace-only strings to None (LLMs emit ``""`` for absent)."""
    if value is None:
        return None
    value = value.strip()
    return value or None


class Segment(BaseModel):
    block_id: str
    type: SegmentType
    speaker: str | None = None  # raw name (pre-resolution) or character id (post)
    text: str
    confidence: float = 1.0

    @field_validator("speaker")
    @classmethod
    def _clean_speaker(cls, value: str | None) -> str | None:
        return _clean_optional(value)

    @model_validator(mode="after")
    def _check_invariants(self) -> "Segment":
        if not BLOCK_ID_PATTERN.match(self.block_id):
            raise ValueError(f"segment block_id {self.block_id!r} must match ch{{NNN}}_b{{NNNN}}")
        if not self.text.strip():
            raise ValueError(f"segment in {self.block_id} has empty text")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"segment {self.block_id} confidence {self.confidence} not in [0,1]")
        # Narration is the narrator's voice, assigned separately — it never names a speaker.
        if self.type is SegmentType.NARRATION:
            self.speaker = None
        elif self.speaker is None:
            raise ValueError(f"{self.type} segment in {self.block_id} must name a speaker")
        return self


class CharacterMention(BaseModel):
    """A character as a provider reports it within one chunk; merged into the registry."""

    name: str
    aliases: list[str] = []
    gender: str | None = None
    age_hint: str | None = None
    description: str | None = None

    @field_validator("gender", "age_hint", "description")
    @classmethod
    def _clean(cls, value: str | None) -> str | None:
        return _clean_optional(value)


class ChunkAttribution(BaseModel):
    """An ``AttributionLLM`` provider's full output for one chunk; also the cached unit.

    Speakers are still raw names here; registry resolution turns them into character ids.
    """

    segments: list[Segment]
    characters: list[CharacterMention] = []


class Character(BaseModel):
    """A registry record. ``id`` is a stable slug; voices reference characters by it."""

    id: str
    canonical_name: str
    aliases: list[str] = []
    gender: str | None = None
    age_hint: str | None = None
    description: str | None = None
    first_appearance: str | None = None  # block_id where first attributed

    def matches_name(self, name: str) -> bool:
        needle = name.strip().casefold()
        return needle == self.canonical_name.casefold() or any(
            needle == a.casefold() for a in self.aliases
        )


class CharacterRegistry(BaseModel):
    characters: list[Character] = []

    def get(self, character_id: str) -> Character | None:
        return next((c for c in self.characters if c.id == character_id), None)

    def find_by_name(self, name: str) -> Character | None:
        return next((c for c in self.characters if c.matches_name(name)), None)


class FlaggedBlock(BaseModel):
    """A block whose attribution failed the reconstruction invariant after all retries."""

    block_id: str
    chapter_index: int
    reason: str


class AttributedChapter(BaseModel):
    index: int  # 1-based chapter index within the normalized book
    title: str
    segments: list[Segment]  # resolved: speaker is a character id (or None for narration)


class AttributionReport(BaseModel):
    """``attribution.json`` — the documented contract between attribution and voices/render."""

    book_id: str
    provider_id: str
    model_id: str
    prompt_version: str
    registry: CharacterRegistry
    chapters: list[AttributedChapter]
    flagged: list[FlaggedBlock] = []
