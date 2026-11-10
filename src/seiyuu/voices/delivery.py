"""Delivery direction by segment type (design/breeze-default-2026-11-09.md, PR 2).

An engine that takes a spoken-style instruction per line (Breeze: "voice direction") gets one
chosen from the SEGMENT TYPE, so the narrator can share the protagonist's voice and still sound
different from the same person talking out loud:

- narration in a first-person book, and a thought in any book -> the inner-voice text;
- narration in a third-person book -> the narration text (empty by default: a plain clone);
- dialogue -> the dialogue text.

The three texts and an on/off switch are per book: the ``delivery`` block of
``output/{book}/cast_plan.json``. A book with no cast plan uses the defaults below (direction
on). A book counts as first person when the narrator's voice is also a character's voice in the
ASSIGNMENT, which is what the render actually voices (the cast plan's "narrate in X's voice"
reaches it through design-cast).

Like :mod:`seiyuu.voices.emotion`, the instruction is merged into a segment's ``settings`` BEFORE
the FROZEN ``SegmentKey`` is built, so it rides ``settings_hash``'s VALUE and never the key
format. Invariants (each pinned by a test):

- an engine that takes no instruction gets its ``settings`` back UNCHANGED (the same dict), so
  every non-Breeze key is byte-identical to a render before this module existed;
- direction off, an empty text, or a heading -> the same dict too: the key of a plain clone;
- the text is stored and hashed in ONE normal form (single spaces, no edges), because the
  engine strips an instruction before speaking it and the hash would not;
- render and the cost estimate both go through :func:`load_instructions` and
  :func:`delivery_settings`, reading the plan from the book's output folder themselves, so no
  caller can hand them different inputs.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from seiyuu.attribute.models import SegmentType
from seiyuu.voices.assignment import VoiceAssignment

CAST_PLAN_NAME = "cast_plan.json"

# Engines whose adapter reads a per-line instruction from ``settings``. Breeze is the only one;
# the key is mirrored from ``engines/breeze_engine.py`` (a test pins that they agree).
DIRECTED_ENGINES = frozenset({"breeze"})
INSTRUCTION_KEY = "instruction"

MAX_INSTRUCTION_CHARS = 300
DEFAULT_INNER_VOICE = (
    "First-person audiobook narration, the narrator's inner voice: calm and intimate, a little "
    "lower and quieter than speaking out loud, at a steady, unhurried pace."
)
DEFAULT_NARRATION = ""  # third-person narration: a plain clone of the narrator's voice
DEFAULT_DIALOGUE = "Natural, expressive spoken delivery that fits the line and the scene."

_TEXT_FIELDS = ("inner_voice", "narration", "dialogue")


class DeliveryError(ValueError):
    """The book's stored delivery settings cannot be read (names the file)."""


def _one_line(value: Any) -> Any:
    """Single spaces, no edges: the one form that is stored, hashed and sent to the engine."""
    return " ".join(value.split()) if isinstance(value, str) else value


class Delivery(BaseModel):
    """A book's delivery direction. An empty text means a plain clone for that kind of line."""

    enabled: bool = True
    inner_voice: str = Field(default=DEFAULT_INNER_VOICE, max_length=MAX_INSTRUCTION_CHARS)
    narration: str = Field(default=DEFAULT_NARRATION, max_length=MAX_INSTRUCTION_CHARS)
    dialogue: str = Field(default=DEFAULT_DIALOGUE, max_length=MAX_INSTRUCTION_CHARS)

    _normalize = field_validator(*_TEXT_FIELDS, mode="before")(_one_line)


class DeliveryUpdate(BaseModel):
    """One user edit of a book's delivery direction. Omitted fields stay as they are."""

    model_config = {"extra": "forbid"}
    enabled: bool | None = None
    inner_voice: str | None = Field(default=None, max_length=MAX_INSTRUCTION_CHARS)
    narration: str | None = Field(default=None, max_length=MAX_INSTRUCTION_CHARS)
    dialogue: str | None = Field(default=None, max_length=MAX_INSTRUCTION_CHARS)

    _normalize = field_validator(*_TEXT_FIELDS, mode="before")(_one_line)

    def is_empty(self) -> bool:
        return not self.model_fields_set


def apply_delivery_update(delivery: Delivery, update: DeliveryUpdate) -> Delivery:
    """``delivery`` with the fields the update names replaced (``""`` clears a text)."""
    changes = {
        name: getattr(update, name)
        for name in update.model_fields_set
        if getattr(update, name) is not None
    }
    return Delivery.model_validate({**delivery.model_dump(), **changes})


class _PlanFile(BaseModel):
    """The one block of cast_plan.json a render reads; every other key is ignored."""

    delivery: Delivery = Delivery()


def load_delivery(book_output_dir: Path) -> Delivery:
    """The book's delivery settings from ``{book_output_dir}/cast_plan.json``.

    No plan, or a plan written before this block existed -> the defaults (direction on). A file
    that cannot be read fails loudly: guessing would silently re-record or mis-deliver a book.
    """
    path = Path(book_output_dir) / CAST_PLAN_NAME
    if not path.is_file():
        return Delivery()
    try:
        return _PlanFile.model_validate_json(path.read_text(encoding="utf-8")).delivery
    except (ValidationError, UnicodeDecodeError) as exc:
        raise DeliveryError(
            f"delivery: {path} is unreadable, so the delivery direction for this book is "
            f"unknown; fix or delete the file: {exc}"
        ) from exc


def narrator_shares_a_voice(assignment: VoiceAssignment) -> bool:
    """First-person test: the narrator's voice is also the voice of a character."""
    return assignment.narrator_voice_id in assignment.assignments.values()


def instructions_for(delivery: Delivery, assignment: VoiceAssignment) -> dict[SegmentType, str]:
    """The instruction for each segment type; a type with none is absent (plain clone)."""
    if not delivery.enabled:
        return {}
    first_person = narrator_shares_a_voice(assignment)
    table = {
        SegmentType.NARRATION: delivery.inner_voice if first_person else delivery.narration,
        SegmentType.THOUGHT: delivery.inner_voice,
        SegmentType.DIALOGUE: delivery.dialogue,
    }
    return {kind: text for kind, text in table.items() if text}


def load_instructions(book_output_dir: Path, assignment: VoiceAssignment) -> dict[SegmentType, str]:
    """What one render or estimate of this book uses: read once, before the first segment."""
    return instructions_for(load_delivery(book_output_dir), assignment)


def delivery_settings(
    settings: dict[str, Any],
    engine_id: str,
    segment_type: SegmentType,
    instructions: dict[SegmentType, str],
    *,
    heading: bool = False,
) -> dict[str, Any]:
    """Merge the segment's delivery instruction into ``settings`` (render AND estimate).

    Returns ``settings`` UNCHANGED (the same dict) for an engine that takes no instruction, for
    a segment type with no instruction, and for a heading (a chapter title is announced, not
    performed), so those keys stay byte-identical to a plain clone.
    """
    if heading or engine_id not in DIRECTED_ENGINES:
        return settings
    instruction = instructions.get(segment_type)
    if not instruction:
        return settings
    return {**settings, INSTRUCTION_KEY: instruction}
