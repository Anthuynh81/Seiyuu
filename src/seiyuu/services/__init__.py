"""Service layer (M6a): stage logic shared by the CLI, job handlers, and the API.

The CLI used to own real behavior (provider lifecycle, edits, drafting, deletion
guards); anything the M6b API also needs lives here instead, so both frontends stay
thin adapters. Services raise :class:`ServiceError` (or a stage's own loud error);
callers map it to their boundary (ClickException, HTTP status)."""

from seiyuu.services.assignment import (
    CastPreview,
    assignment_path,
    draft_assignment,
    load_assignment,
    save_assignment,
    suggest_assignment,
)
from seiyuu.services.attribution import (
    build_adjudicator,
    build_provider,
    load_report,
    record_edit,
    run_adjudication,
    run_attribution,
    undo_edit,
)
from seiyuu.services.characters import CharactersOverview, CharacterSummary, characters_overview
from seiyuu.services.common import ServiceError
from seiyuu.services.edits import (
    EDITS_NAME,
    EditLog,
    EditOp,
    MergeCharacters,
    ReassignSegment,
    RenameCharacter,
    anchor_op,
    append_edit,
    apply_edits,
    load_edits,
    pop_edit,
    save_edits,
)
from seiyuu.services.voices import VoiceReference, delete_voice, voice_references

__all__ = [
    "EDITS_NAME",
    "CastPreview",
    "CharacterSummary",
    "CharactersOverview",
    "EditLog",
    "EditOp",
    "MergeCharacters",
    "ReassignSegment",
    "RenameCharacter",
    "ServiceError",
    "VoiceReference",
    "anchor_op",
    "append_edit",
    "apply_edits",
    "assignment_path",
    "build_adjudicator",
    "build_provider",
    "characters_overview",
    "delete_voice",
    "draft_assignment",
    "load_assignment",
    "load_edits",
    "load_report",
    "pop_edit",
    "record_edit",
    "run_adjudication",
    "run_attribution",
    "save_assignment",
    "save_edits",
    "suggest_assignment",
    "undo_edit",
    "voice_references",
]
