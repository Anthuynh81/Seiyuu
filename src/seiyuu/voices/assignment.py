"""Per-book character→voice assignment (output/{book_id}/assignments.json).

Links the attribution registry's character ids to voice ids. The narrator catches narration,
None-speaker segments, and any unmapped speaker so resolution never fails. Draft vs final is a
``stage`` field (one file in M3; the dual-file workflow is M6).
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from seiyuu.attribute.models import Segment, SegmentType
from seiyuu.voices.models import today_iso

ASSIGNMENT_NAME = "assignments.json"


class AssignmentStage(StrEnum):
    DRAFT = "draft"
    FINAL = "final"


class VoiceAssignment(BaseModel):
    schema_version: int = 1
    book_id: str
    stage: AssignmentStage = AssignmentStage.DRAFT
    narrator_voice_id: str
    assignments: dict[str, str] = {}  # character_id (make_character_id slug) -> voice_id
    thought_voice_id: str | None = None  # None -> thoughts use the speaker's own voice
    created_at: str = Field(default_factory=today_iso)


def resolve_voice(segment: Segment, assignment: VoiceAssignment) -> str:
    """The voice_id to render `segment` with. Narrator is the catch-all (never fails)."""
    if segment.type is SegmentType.NARRATION or segment.speaker is None:
        return assignment.narrator_voice_id
    if segment.type is SegmentType.THOUGHT and assignment.thought_voice_id:
        return assignment.thought_voice_id
    return assignment.assignments.get(segment.speaker, assignment.narrator_voice_id)
