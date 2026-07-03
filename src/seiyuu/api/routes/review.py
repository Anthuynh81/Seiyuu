"""Character Review writes: the manual-edits overlay and the voice assignment.

Every edit goes through ``record_edit`` — file-locked and anchor-filling — never the
unlocked primitives; the response is the ANCHORED op, the client's proof of what was
recorded. Edit and assignment writes are refused (409 ``render_active``) while a render
job for the book is queued/running: the write would otherwise surface hours later as
that job's quote-drift failure. Assignment writes additionally serialize against voice
deletion through the process-wide voices mutex (shared with M6b-6's DELETE /voices).
"""

import threading
from typing import Annotated

from fastapi import APIRouter, Body, Request
from pydantic import ValidationError

from seiyuu.api.deps import SettingsDep, StoreDep
from seiyuu.api.errors import ApiError
from seiyuu.api.routes.common import effective_report, guard_render_active, status_or_404
from seiyuu.api.schemas import (
    AssignmentDraftRequest,
    AssignmentDraftResponse,
    AssignmentWrite,
    EditRequest,
    MergeRequest,
    RenameRequest,
)
from seiyuu.services import (
    EditLog,
    MergeCharacters,
    ReassignSegment,
    RenameCharacter,
    ServiceError,
    draft_assignment,
    load_assignment,
    load_edits,
    record_edit,
    save_assignment,
    undo_edit,
)
from seiyuu.settings import Settings
from seiyuu.voices import AssignmentStage, VoiceAssignment, VoiceLibrary, VoiceLibraryError

router = APIRouter(tags=["review"])


def _edits_or_500(cfg: Settings, book_id: str) -> EditLog:
    try:
        return load_edits(cfg.books_dir / book_id)
    except ServiceError as exc:
        raise ApiError(500, "corrupt_artifact", str(exc)) from exc


def _voices_mutex(request: Request) -> threading.Lock:
    return request.app.state.voices_mutex


# -- edits --------------------------------------------------------------------------------


@router.get("/books/{book_id}/edits", response_model=EditLog)
def edit_log(book_id: str, cfg: SettingsDep) -> EditLog:
    """The op log verbatim, anchors visible (they are the replay contract, not a secret)."""
    status_or_404(cfg, book_id)
    return _edits_or_500(cfg, book_id)


@router.post("/books/{book_id}/edits", status_code=201)
def record_edit_op(
    book_id: str,
    edit: Annotated[EditRequest, Body()],
    cfg: SettingsDep,
    store: StoreDep,
):
    status = status_or_404(cfg, book_id)
    effective_report(cfg, book_id, status)  # clean 404 (no attribution) / 500 (corrupt)
    guard_render_active(store, book_id)

    if isinstance(edit, RenameRequest):
        op = RenameCharacter(character_id=edit.character_id, new_name=edit.new_name)
    elif isinstance(edit, MergeRequest):
        op = MergeCharacters(loser_id=edit.loser_id, winner_id=edit.winner_id)
    else:
        op = ReassignSegment(
            block_id=edit.block_id, segment_index=edit.segment_index, speaker=edit.speaker
        )
    try:
        anchored = record_edit(cfg.books_dir / book_id, op)
    except ServiceError as exc:
        # record_edit re-reads under its lock, so a corrupt artifact can still surface
        # here (rare TOCTOU vs the pre-check above); everything else is an anchor
        # conflict — the op doesn't apply to the CURRENT effective report.
        if "corrupt" in str(exc):
            raise ApiError(500, "corrupt_artifact", str(exc)) from exc
        raise ApiError(409, "edit_conflict", str(exc)) from exc
    return anchored


@router.delete("/books/{book_id}/edits/last")
def undo_last_edit(book_id: str, cfg: SettingsDep, store: StoreDep):
    status_or_404(cfg, book_id)
    guard_render_active(store, book_id)
    try:
        removed = undo_edit(cfg.books_dir / book_id)
    except ServiceError as exc:
        raise ApiError(500, "corrupt_artifact", str(exc)) from exc
    if removed is None:
        raise ApiError(404, "not_found", f"book {book_id!r} has no manual edits to undo")
    return {"removed": removed}


# -- assignment ---------------------------------------------------------------------------


@router.get("/books/{book_id}/assignment", response_model=VoiceAssignment)
def read_assignment(book_id: str, cfg: SettingsDep) -> VoiceAssignment:
    status_or_404(cfg, book_id)
    try:
        return load_assignment(cfg.output_dir, book_id)
    except ServiceError as exc:
        raise ApiError(404, "not_found", str(exc)) from exc
    except (ValidationError, OSError, ValueError) as exc:
        # load_assignment does NOT wrap the pydantic error — caught explicitly here
        # (scoping doc): a corrupt assignments.json is a server data fault, not a 404.
        path = cfg.output_dir / book_id
        raise ApiError(500, "corrupt_artifact", f"corrupt assignment under {path}: {exc}") from exc


def _existing_voice_ids(cfg: Settings) -> set[str]:
    if not cfg.voices_dir.is_dir():
        return set()
    return {d.name for d in cfg.voices_dir.iterdir() if (d / "meta.json").is_file()}


@router.post(
    "/books/{book_id}/assignment/draft", response_model=AssignmentDraftResponse, status_code=201
)
def draft(
    book_id: str,
    body: AssignmentDraftRequest,
    request: Request,
    cfg: SettingsDep,
    store: StoreDep,
) -> AssignmentDraftResponse:
    """Generate-and-save the deterministic draft. Not a pure read: missing auto voices
    get meta.json files (deterministic, so re-POSTing is idempotent — same voice ids,
    same SegmentKeys)."""
    status = status_or_404(cfg, book_id)
    report, warnings = effective_report(cfg, book_id, status)
    guard_render_active(store, book_id)

    with _voices_mutex(request):
        before = _existing_voice_ids(cfg)
        try:
            assignment = draft_assignment(
                report,
                VoiceLibrary(cfg.voices_dir),
                default_preset=cfg.kokoro_default_voice,
                narrator_voice_id=body.narrator_voice_id,
                thought_voice_id=body.thought_voice_id,
                accent=body.accent,
                stage=AssignmentStage(body.stage),
                overrides=body.overrides,
            )
        except ServiceError as exc:  # unknown character / voice — actionable verbatim
            raise ApiError(422, "invalid", str(exc)) from exc
        save_assignment(assignment, cfg.output_dir)
        created = sorted(_existing_voice_ids(cfg) - before)
    return AssignmentDraftResponse(
        assignment=assignment, created_voice_ids=created, edit_warnings=warnings
    )


@router.put("/books/{book_id}/assignment", response_model=VoiceAssignment)
def write_assignment(
    book_id: str,
    body: AssignmentWrite,
    request: Request,
    cfg: SettingsDep,
    store: StoreDep,
) -> VoiceAssignment:
    """Full-replace write for the per-character voice picker. Changing the assignment
    naturally invalidates outstanding cost quotes (assignment-hash drift) — by design."""
    status = status_or_404(cfg, book_id)
    report, _warnings = effective_report(cfg, book_id, status)
    guard_render_active(store, book_id)

    known = {c.id for c in report.registry.characters}
    unknown = sorted(set(body.assignments) - known)
    if unknown:
        raise ApiError(
            422, "invalid", f"unknown character(s) in this attribution: {', '.join(unknown)}"
        )
    speaking = {
        seg.speaker
        for chapter in report.chapters
        for seg in chapter.segments
        if seg.speaker is not None
    }
    missing = sorted(speaking - set(body.assignments))
    if missing:
        raise ApiError(
            422,
            "invalid",
            "the assignment map must cover every speaking character; "
            f"missing: {', '.join(missing)}",
        )

    # Serialized against DELETE /voices (M6b-6) so a voice can't vanish between this
    # validation and the durable write.
    with _voices_mutex(request):
        library = VoiceLibrary(cfg.voices_dir)
        voice_ids = {body.narrator_voice_id, *body.assignments.values()}
        if body.thought_voice_id is not None:
            voice_ids.add(body.thought_voice_id)
        for voice_id in sorted(voice_ids):
            try:
                library.load(voice_id)
            except VoiceLibraryError as exc:
                raise ApiError(422, "invalid", str(exc)) from exc
        assignment = VoiceAssignment(
            book_id=book_id,
            stage=AssignmentStage(body.stage),
            narrator_voice_id=body.narrator_voice_id,
            assignments=body.assignments,
            thought_voice_id=body.thought_voice_id,
        )
        save_assignment(assignment, cfg.output_dir)
    return assignment
