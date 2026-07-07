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

from fastapi import APIRouter, Body, Request, Response
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
    ReassignRequest,
    RenameRequest,
    SuggestCastResponse,
)
from seiyuu.services import (
    EditLog,
    MergeCharacters,
    ReassignSegment,
    RenameCharacter,
    ServiceError,
    SetEmotion,
    draft_assignment,
    load_assignment,
    load_edits,
    record_edit,
    save_assignment,
    suggest_assignment,
    undo_edit,
)
from seiyuu.services.llm_advisory import resolve_advisory, run_cast_hints
from seiyuu.settings import Settings
from seiyuu.voices import AssignmentStage, VoiceAssignment, VoiceLibrary, VoiceLibraryError

router = APIRouter(tags=["review"])


def _cast_trait_hints(
    cfg: Settings, body: AssignmentDraftRequest, report
) -> dict[str, set[str]] | None:
    """F4: when ``use_llm`` is set on a SMART cast, run the opt-in Layer-2 LLM caster and return
    its per-character trait hints. Off (or on the hash path) it returns None so the deterministic
    keyword bias is used unchanged. Applies the SAME paid gate as attribution: when the resolved
    cast provider is anthropic it is a PAID call on this HTTP path, so it needs confirm_paid + the
    key; the local provider is free but still runs only on this explicit action."""
    if not body.use_llm or body.strategy != "smart":
        return None
    resolved = resolve_advisory(cfg, cfg.cast_provider, cfg.cast_model, body.cast_provider)
    if resolved.is_paid:
        if not body.confirm_paid:
            raise ApiError(
                402,
                "payment_confirmation_required",
                "the LLM caster with cast_provider=anthropic calls the paid Anthropic API; "
                "re-send with confirm_paid=true to approve the spend",
            )
        if not cfg.anthropic_api_key:
            raise ApiError(
                503, "not_ready", "ANTHROPIC_API_KEY not set; required for the anthropic LLM caster"
            )
    override_ids = set(body.overrides or {})
    castable = [c for c in report.registry.characters if c.id not in override_ids]
    try:
        return run_cast_hints(cfg, resolved, castable)
    except Exception as exc:  # a provider/transport failure must not 500 the whole draft
        raise ApiError(502, "upstream_error", f"LLM caster failed: {exc}") from exc


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
    response: Response,
    cfg: SettingsDep,
    store: StoreDep,
):
    response.headers["Location"] = f"/api/books/{book_id}/edits"
    status = status_or_404(cfg, book_id)
    effective_report(cfg, book_id, status)  # clean 404 (no attribution) / 500 (corrupt)
    guard_render_active(store, book_id)

    if isinstance(edit, RenameRequest):
        op = RenameCharacter(character_id=edit.character_id, new_name=edit.new_name)
    elif isinstance(edit, MergeRequest):
        op = MergeCharacters(loser_id=edit.loser_id, winner_id=edit.winner_id)
    elif isinstance(edit, ReassignRequest):
        op = ReassignSegment(
            block_id=edit.block_id, segment_index=edit.segment_index, speaker=edit.speaker
        )
    else:
        op = SetEmotion(
            block_id=edit.block_id, segment_index=edit.segment_index, emotion=edit.emotion
        )
    try:
        anchored = record_edit(cfg.books_dir / book_id, op)
    except ServiceError as exc:
        # record_edit re-reads under its lock, so a corrupt artifact can still surface
        # here (rare TOCTOU vs the pre-check above); everything else is an anchor
        # conflict. FULL phrases, not a bare "corrupt": conflict messages interpolate
        # character/block ids (LLM-derived slugs like "corrupt-one") — the same
        # shadowing gate_code was hardened against.
        message = str(exc)
        if "corrupt attribution" in message or "corrupt edits" in message:
            raise ApiError(500, "corrupt_artifact", message) from exc
        raise ApiError(409, "edit_conflict", message) from exc
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
    response: Response,
    cfg: SettingsDep,
    store: StoreDep,
) -> AssignmentDraftResponse:
    response.headers["Location"] = f"/api/books/{book_id}/assignment"
    """Generate-and-save the deterministic draft. Not a pure read: missing auto voices
    get meta.json files (deterministic, so re-POSTing is idempotent — same voice ids,
    same SegmentKeys)."""
    status = status_or_404(cfg, book_id)
    report, warnings = effective_report(cfg, book_id, status)
    guard_render_active(store, book_id)

    # F4: resolve the LLM trait hints (with the paid gate) BEFORE taking the voices mutex, so a
    # network-bound LLM call never holds the lock. Off/hash -> None (deterministic bias unchanged).
    trait_hints = _cast_trait_hints(cfg, body, report)

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
                strategy=body.strategy,
                recast=body.recast,
                trait_hints=trait_hints,
            )
        except (ServiceError, ValueError) as exc:  # unknown character/voice, exhausted pool
            raise ApiError(422, "invalid", str(exc)) from exc
        save_assignment(assignment, cfg.output_dir)
        created = sorted(_existing_voice_ids(cfg) - before)
    return AssignmentDraftResponse(
        assignment=assignment, created_voice_ids=created, edit_warnings=warnings
    )


@router.post("/books/{book_id}/assignment/suggest", response_model=SuggestCastResponse)
def suggest(
    book_id: str,
    body: AssignmentDraftRequest,
    cfg: SettingsDep,
) -> SuggestCastResponse:
    """PREVIEW the smart cast without saving: the proposed distinct-voice assignment plus
    which auto voices it would create vs (with recast) overwrite. The user commits it via
    POST /assignment/draft with strategy=smart — this endpoint writes nothing, so it is safe
    to call repeatedly and never touches the render cache."""
    status = status_or_404(cfg, book_id)
    report, warnings = effective_report(cfg, book_id, status)
    try:
        preview = suggest_assignment(
            report,
            VoiceLibrary(cfg.voices_dir),
            default_preset=cfg.kokoro_default_voice,
            narrator_voice_id=body.narrator_voice_id,
            thought_voice_id=body.thought_voice_id,
            accent=body.accent,
            stage=AssignmentStage(body.stage),
            overrides=body.overrides,
        )
    except (ServiceError, ValueError) as exc:
        raise ApiError(422, "invalid", str(exc)) from exc
    return SuggestCastResponse(
        assignment=preview.assignment,
        would_create_voice_ids=preview.would_create,
        would_recast_voice_ids=preview.would_recast,
        edit_warnings=warnings,
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
