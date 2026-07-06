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


class BlockSpeaker(BaseModel):
    """The model's only job per block: name who speaks its dialogue (null if none).

    Text and segment TYPE are derived deterministically (we split on quotes; quoted spans
    are dialogue, prose is narration), so the model never reproduces text or counts spans —
    it just attributes. That makes reconstruction structural and the task small models can do.
    """

    block_id: str
    speaker: str | None = None
    confidence: float = 1.0

    @field_validator("speaker")
    @classmethod
    def _clean_speaker(cls, value: str | None) -> str | None:
        return _clean_optional(value)


class ThoughtVerdict(BaseModel):
    """The model's confirm/thinker decision for ONE deterministic thought candidate.

    Keyed by ``candidate_id`` (``"{block_id}:{start_offset}"``, generated deterministically
    from the italic runs) — the model returns a verdict per candidate, never a slice or
    offset, so it can neither split/rewrite text nor mint a candidate. Verdicts whose
    ``candidate_id`` is not in the generated set are dropped. A candidate becomes a THOUGHT
    only when ``is_thought`` is true with a resolvable ``thinker`` above the confidence floor;
    otherwise it degrades to narration (the same source slice).
    """

    candidate_id: str
    is_thought: bool = False
    thinker: str | None = None  # raw name (pre-resolution); resolved like any speaker
    confidence: float = 1.0

    @field_validator("thinker")
    @classmethod
    def _clean_thinker(cls, value: str | None) -> str | None:
        return _clean_optional(value)


class ChunkLabels(BaseModel):
    """An ``AttributionLLM`` provider's RAW output: a speaker per block + character mentions.

    The provider assembles this into a :class:`ChunkAttribution` (segments with source text).
    ``thoughts`` carries one :class:`ThoughtVerdict` per presented thought candidate; it is
    empty (and unused) on the thought-off v3 path.
    """

    blocks: list[BlockSpeaker] = []
    characters: list[CharacterMention] = []
    thoughts: list[ThoughtVerdict] = []


class ChunkAttribution(BaseModel):
    """Assembled, validated attribution for one chunk; the cached unit.

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


class CharacterEvidence(BaseModel):
    """One character's identifying evidence shown to the LLM adjudicator (read-only view).

    A bounded projection of a :class:`Character`: the adjudicator sees only these fields for
    the two members of a candidate pair, never the free registry, so it can never introduce
    a merge the deterministic generator did not surface.
    """

    id: str
    canonical_name: str
    aliases: list[str] = []
    gender: str | None = None
    age_hint: str | None = None
    description: str | None = None

    @classmethod
    def from_character(cls, char: "Character") -> "CharacterEvidence":
        return cls(
            id=char.id,
            canonical_name=char.canonical_name,
            aliases=list(char.aliases),
            gender=char.gender,
            age_hint=char.age_hint,
            description=char.description,
        )


class CandidatePair(BaseModel):
    """A deterministically-generated merge candidate the LLM only APPROVES/REJECTS.

    ``pair_id`` is derived from the two (sorted) character ids so it is stable across reruns;
    ``generator`` records which rule proposed it (``G1``/``G2``/``G3``). The LLM keys its
    verdict by ``pair_id`` and can never emit an id or name of its own.
    """

    pair_id: str
    generator: str
    a: CharacterEvidence
    b: CharacterEvidence


class PairVerdict(BaseModel):
    """The LLM's approve/reject decision for one :class:`CandidatePair`, keyed by ``pair_id``.

    Verdicts whose ``pair_id`` is not in the generated set are ignored by the adjudicator, so
    a stray or hallucinated id can never cause a merge.
    """

    pair_id: str
    same_person: bool
    confidence: float = 0.0
    justification: str = ""


class AdjudicationResult(BaseModel):
    """Schema-enforced LLM output: one verdict per presented candidate pair."""

    verdicts: list[PairVerdict] = []


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
    registry_notes: list[str] = []  # conservative merges skipped for human review
