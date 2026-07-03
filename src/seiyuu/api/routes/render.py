"""Money + render: estimate (pure read), quote minting, the token-gated render job,
and the render/validation/segment-audio reads (scoping doc sections 4-5).

The flow: GET cost-estimate (never touches signing-key state) -> user approves ->
POST quotes (fresh server estimate, signed single-use token) -> POST render with the
token (enqueue does a NON-consuming dry-run so refusals are immediate 402s and never
burn the token; the authoritative consume happens at job start, in the handler).
"""

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import ValidationError

from seiyuu.api.deps import RegistryDep, RunnerDep, SettingsDep, StoreDep
from seiyuu.api.enqueue import enqueue_job
from seiyuu.api.errors import ApiError
from seiyuu.api.money import compute_estimate, gate_code, resolve_single
from seiyuu.api.routes.common import load_book, status_or_404
from seiyuu.api.schemas import (
    CostEstimateOut,
    JobOut,
    QuoteRequest,
    QuoteResponse,
    RenderChapterOut,
    RenderParams,
    RenderSummaryOut,
    SingleSpec,
    ValidationReportOut,
    ValidationRow,
    VoiceUseOut,
)
from seiyuu.duration import estimate_runtime_seconds
from seiyuu.ingest.models import NormalizedBook
from seiyuu.render.gate import (
    FULL_RENDER_CONFIRM_BLOCKS,
    CostGateError,
    CostQuote,
    quote_consumed,
    verify_quote,
)
from seiyuu.render.models import RenderManifest
from seiyuu.repository import BookStatus, JobKind, JobState
from seiyuu.repository.books import MANIFEST_NAME
from seiyuu.services import ServiceError
from seiyuu.settings import Settings
from seiyuu.voices import VoiceLibraryError

router = APIRouter(tags=["render"])


def _check_mode_prerequisites(status: BookStatus, mode: str, *, as_job: bool) -> None:
    """Pure reads report missing stages as 404; job creation as 409 stage_prerequisite."""
    missing = None
    if not status.ingested:
        missing = "run ingest first"
    elif mode == "multivoice":
        if not status.attributed:
            missing = "run attribute first"
        elif not status.assigned:
            missing = "run assign first"
    if missing is None:
        return
    message = f"book {status.book_id!r}: {missing}"
    if as_job:
        raise ApiError(409, "stage_prerequisite", message)
    raise ApiError(404, "not_found", message)


def _check_chapters(book: NormalizedBook, chapters: list[int]) -> tuple[int, ...]:
    for index in chapters:
        if index < 1 or index > len(book.chapters):
            raise ApiError(
                422, "invalid", f"chapter {index} out of range (book has {len(book.chapters)})"
            )
    return tuple(sorted(set(chapters)))


def _estimate_or_http(cfg, registry, book, book_id, *, mode, chapters, single):
    """compute_estimate with the read-path error mapping (422 unknown voice/engine;
    residual ServiceError = corrupt artifact -> 500; marker checks already ran)."""
    try:
        return compute_estimate(
            cfg, registry, book, book_id, mode=mode, chapters=chapters, single=single
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise ApiError(422, "invalid", str(exc)) from exc
    except ServiceError as exc:
        raise ApiError(500, "corrupt_artifact", str(exc)) from exc


def _resolve_single_or_422(cfg: Settings, spec: SingleSpec | None):
    try:
        return resolve_single(cfg, spec)
    except ValueError as exc:  # non-kokoro engine with no explicit voice
        raise ApiError(422, "invalid", str(exc)) from exc


def _preflight_renderability(cfg: Settings, mode: str, single, est_ctx, book_id: str) -> None:
    """Refuse foreseeable dead-ends BEFORE a token exists or is consumed: a paid engine
    with no API key, and clone-consent failures. Both are fully computable here from
    what the estimate already read; without this they surface only AFTER the handler
    burns the single-use approval token. Best-effort UX — the render pipeline's own
    consent/paid gates remain the enforcement."""
    from seiyuu.services import load_assignment
    from seiyuu.voices import VoiceLibrary

    library = VoiceLibrary(cfg.voices_dir)

    def check_consent(voice_id: str) -> str:
        try:
            meta = library.load(voice_id)
        except VoiceLibraryError as exc:
            raise ApiError(422, "invalid", str(exc)) from exc
        try:
            library.verify_consent(meta)
        except VoiceLibraryError as exc:
            raise ApiError(409, "consent_invalid", str(exc)) from exc
        return meta.engine

    if mode == "single":
        paid_engine = single.engine_id == "elevenlabs"
        if library.meta_path(single.voice_id).is_file():
            check_consent(single.voice_id)
    else:
        assignment = load_assignment(cfg.output_dir, book_id)
        voice_ids = {assignment.narrator_voice_id, *assignment.assignments.values()}
        if assignment.thought_voice_id:
            voice_ids.add(assignment.thought_voice_id)
        engines = {check_consent(voice_id) for voice_id in sorted(voice_ids)}
        paid_engine = "elevenlabs" in engines
    if paid_engine and est_ctx.est.total_usd > 0 and not cfg.elevenlabs_api_key:
        raise ApiError(
            503,
            "not_ready",
            "ELEVENLABS_API_KEY not set; the paid segments in this render cannot "
            "synthesize — configure the key before quoting or rendering",
        )


@router.get("/books/{book_id}/cost-estimate", response_model=CostEstimateOut)
def cost_estimate(
    book_id: str,
    cfg: SettingsDep,
    registry: RegistryDep,
    mode: Annotated[str, Query(pattern="^(multivoice|single)$")] = "multivoice",
    chapters: Annotated[list[int], Query()] = [],  # noqa: B006
    engine: Annotated[str | None, Query()] = None,
    voice: Annotated[str | None, Query()] = None,
    speed: Annotated[float, Query(gt=0)] = 1.0,
    seed: int = 41172,
) -> CostEstimateOut:
    """Pure read: exact frozen SegmentKeys vs the segment cache. No network, no GPU,
    no consent check, and NEVER the signing-key state."""
    status = status_or_404(cfg, book_id)
    if mode == "multivoice" and (engine is not None or voice is not None):
        # Mirror the POST bodies' require-single-iff-mode 422: silently discarding these
        # would let the money dialog display an estimate for a different scope.
        raise ApiError(422, "invalid", "engine/voice (and speed/seed) apply only to mode=single")
    _check_mode_prerequisites(status, mode, as_job=False)
    book = load_book(cfg, book_id)
    wanted = _check_chapters(book, chapters)
    single = (
        _resolve_single_or_422(cfg, SingleSpec(engine=engine, voice=voice, speed=speed, seed=seed))
        if mode == "single"
        else None
    )
    ctx = _estimate_or_http(cfg, registry, book, book_id, mode=mode, chapters=wanted, single=single)
    return CostEstimateOut(
        total_usd=ctx.est.total_usd,
        paid_segments=ctx.est.paid_segments,
        cached_segments=ctx.est.cached_segments,
        free_segments=ctx.est.free_segments,
        fingerprint=ctx.est.fingerprint,
        assignment_hash=ctx.assignment_hash,
        mode=mode,
        chapters=list(wanted),
        edit_warnings=ctx.edit_warnings,
    )


@router.post("/books/{book_id}/quotes", response_model=QuoteResponse, status_code=201)
def mint_quote(
    book_id: str,
    body: QuoteRequest,
    cfg: SettingsDep,
    registry: RegistryDep,
) -> QuoteResponse:
    """Mint a signed single-use cost token over a FRESH server-side estimate (the
    ceiling is enforced at issuance; the client's displayed estimate is never trusted)."""
    from seiyuu.render.gate import issue_quote

    status = status_or_404(cfg, book_id)
    _check_mode_prerequisites(status, body.mode, as_job=False)
    book = load_book(cfg, book_id)
    wanted = _check_chapters(book, body.chapters)
    single = _resolve_single_or_422(cfg, body.single) if body.mode == "single" else None
    ctx = _estimate_or_http(
        cfg, registry, book, book_id, mode=body.mode, chapters=wanted, single=single
    )
    if ctx.est.total_usd <= 0:
        raise ApiError(
            409,
            "nothing_to_quote",
            "every segment in scope is free or cached — render without a token",
        )
    _preflight_renderability(cfg, body.mode, single, ctx, book_id)
    try:
        quote = issue_quote(
            ctx.est,
            book_id=book_id,
            chapters=wanted,
            assignment_hash=ctx.assignment_hash,
            max_usd=cfg.render_max_usd,
            ttl_seconds=cfg.cost_quote_ttl_seconds,
            data_dir=cfg.data_dir,
        )
    except CostGateError as exc:
        raise ApiError(402, gate_code(exc), str(exc)) from exc
    return QuoteResponse(
        token=quote.encode(),
        book_id=quote.book_id,
        chapters=list(quote.chapters),
        total_usd=quote.total_usd,
        paid_segments=quote.paid_segments,
        fingerprint=quote.fingerprint,
        assignment_hash=quote.assignment_hash,
        issued_at=quote.issued_at,
        expires_at=quote.expires_at,
        ttl_seconds=cfg.cost_quote_ttl_seconds,
        max_usd_ceiling=cfg.render_max_usd,
    )


@router.post("/books/{book_id}/render", response_model=JobOut, status_code=202)
def render_job(
    book_id: str,
    params: RenderParams,
    request: Request,
    response: Response,
    cfg: SettingsDep,
    registry: RegistryDep,
    store: StoreDep,
    runner: RunnerDep,
) -> JobOut:
    status = status_or_404(cfg, book_id)
    _check_mode_prerequisites(status, params.mode, as_job=True)
    book = load_book(cfg, book_id)
    wanted = _check_chapters(book, params.chapters)

    # An attribute job in flight would regenerate the report under this render's feet —
    # the failure would only surface as the worker's late fingerprint refusal. Refuse now.
    live = store.list_jobs(book_id=book_id, states=[JobState.QUEUED, JobState.RUNNING])
    attribute = next((j for j in live if j.kind is JobKind.ATTRIBUTE), None)
    if attribute is not None:
        raise ApiError(
            409,
            "conflicting_job",
            f"an attribute job for {book_id!r} is {attribute.state.value}; wait for it "
            "or cancel it before rendering",
            detail=JobOut.from_job(attribute).model_dump(mode="json"),
        )

    # Whole-book confirm, server-enforced (free renders are not money-gated, so this is
    # the only whole-book stop; the threshold is exposed in /api/system limits).
    if not wanted:
        speakable = sum(1 for c in book.chapters for b in c.blocks if b.is_speakable)
        if speakable > FULL_RENDER_CONFIRM_BLOCKS and not params.confirm_full:
            raise ApiError(
                409,
                "full_render_confirmation_required",
                f"full-book render: {speakable} speakable segments — a long GPU job; "
                "re-send with confirm_full=true",
                detail={
                    "speakable_blocks": speakable,
                    "runtime_estimate_seconds": estimate_runtime_seconds(
                        book, wpm=cfg.narration_wpm
                    ),
                },
            )

    # Money dry-run: fail tampering/expiry/drift NOW with the granular 402 and the token
    # unburned; the authoritative consume happens at job start (sign-off Q5).
    single = _resolve_single_or_422(cfg, params.single) if params.mode == "single" else None
    ctx = _estimate_or_http(
        cfg, registry, book, book_id, mode=params.mode, chapters=wanted, single=single
    )
    _preflight_renderability(cfg, params.mode, single, ctx, book_id)
    if ctx.est.total_usd > 0:
        if not params.cost_token:
            raise ApiError(
                402,
                "token_required",
                f"this render bills ~${ctx.est.total_usd:.2f} over "
                f"{ctx.est.paid_segments} paid segment(s); mint a token via "
                "POST /api/books/{book_id}/quotes and re-send as cost_token",
                detail={"estimated_usd": ctx.est.total_usd},
            )
        try:
            quote = CostQuote.decode(params.cost_token)
        except CostGateError as exc:
            raise ApiError(422, "invalid", str(exc)) from exc
        try:
            verify_quote(
                quote,
                book_id=book_id,
                chapters=wanted,
                fingerprint=ctx.est.fingerprint,
                assignment_hash=ctx.assignment_hash,
                recomputed_total_usd=ctx.est.total_usd,
                max_usd=cfg.render_max_usd,
                data_dir=cfg.data_dir,
                consume=False,
            )
        except CostGateError as exc:
            raise ApiError(402, gate_code(exc), str(exc)) from exc
        # verify(consume=False) never reads the single-use ledger — probe it so a spent
        # token is an immediate 402, not a failed job minutes later. Best-effort: the
        # handler's atomic consume remains the enforcement.
        if quote_consumed(cfg.data_dir, quote.sig):
            raise ApiError(
                402,
                "quote_used",
                "cost token already used (tokens are single-use); re-run estimate-cost "
                "for a new approval",
            )

    job = enqueue_job(
        store=store,
        runner=runner,
        mutex=request.app.state.enqueue_mutex,
        book_id=book_id,
        kind=JobKind.RENDER,
        params=params.model_dump(),
    )
    response.headers["Location"] = f"/api/jobs/{job.job_id}"
    return JobOut.from_job(job)


# -- render reads -------------------------------------------------------------------------


def _manifest_or_http(cfg: Settings, book_id: str, status: BookStatus) -> RenderManifest:
    if not status.rendered:
        raise ApiError(404, "not_found", f"book {book_id!r} has no render; run render first")
    path = cfg.output_dir / book_id / MANIFEST_NAME
    try:
        return RenderManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError, ValueError) as exc:
        raise ApiError(500, "corrupt_artifact", f"corrupt render manifest {path}: {exc}") from exc


@router.get("/books/{book_id}/render", response_model=RenderSummaryOut)
def render_summary(book_id: str, cfg: SettingsDep) -> RenderSummaryOut:
    status = status_or_404(cfg, book_id)
    manifest = _manifest_or_http(cfg, book_id, status)
    chapters = [
        RenderChapterOut(
            index=ch.index,
            title=ch.title,
            segments=len(ch.segments),
            duration_seconds=round(sum(s.duration_seconds for s in ch.segments), 3),
        )
        for ch in manifest.chapters
    ]
    return RenderSummaryOut(
        book_id=manifest.book_id,
        mode="single" if manifest.engine is not None else "multivoice",
        engine=manifest.engine,
        engine_model_version=manifest.engine_model_version,
        voice_id=manifest.voice_id,
        seed=manifest.seed,
        chapters=chapters,
        total_seconds=round(sum(c.duration_seconds for c in chapters), 3),
        voices_used={
            vid: VoiceUseOut(**use.model_dump()) for vid, use in manifest.voices_used.items()
        },
        validation_failures=manifest.validation_failures,
        assignment_present=manifest.assignment is not None,
    )


@router.get("/books/{book_id}/validation", response_model=ValidationReportOut)
def validation_report(
    book_id: str,
    cfg: SettingsDep,
    all: bool = False,  # noqa: A002 — the documented query param name
) -> ValidationReportOut:
    status = status_or_404(cfg, book_id)
    manifest = _manifest_or_http(cfg, book_id, status)
    rows: list[ValidationRow] = []
    validated = 0
    position_in_block: dict[str, int] = {}
    for chapter in manifest.chapters:
        for seg in chapter.segments:
            # counts ALL of the block's rendered segments, validated or not, so the
            # index addresses the same segment the audio route's ?segment= serves
            seg_index = position_in_block.get(seg.block_id, 0)
            position_in_block[seg.block_id] = seg_index + 1
            if seg.validation is None:
                continue
            validated += 1
            if seg.validation.ok and not all:
                continue
            rows.append(
                ValidationRow(
                    chapter_index=chapter.index,
                    block_id=seg.block_id,
                    segment_index=seg_index,
                    voice_id=seg.voice_id,
                    ok=seg.validation.ok,
                    score=seg.validation.score,
                    expected=seg.validation.expected,
                    transcript=seg.validation.transcript,
                    synth_attempts=seg.synth_attempts,
                )
            )
    return ValidationReportOut(
        validated_segments=validated,
        validation_failures=manifest.validation_failures,
        results=rows,
    )


@router.get("/books/{book_id}/segments/{block_id}/audio")
def segment_audio(
    book_id: str,
    block_id: str,
    cfg: SettingsDep,
    segment: Annotated[int, Query(ge=0)] = 0,
) -> FileResponse:
    """Per-segment WAV playback (Character Review plays flagged blocks; Validation plays
    failures). A multivoice block renders SEVERAL segments (narration + each quoted
    span, possibly different voices) — ``?segment=`` addresses the Nth one, in the same
    order the segment browser and validation report count them, so a failing dialogue
    span is reachable even when the block's narration passed. Resolved via manifest
    lookup ONLY — never a client-supplied path."""
    status = status_or_404(cfg, book_id)
    manifest = _manifest_or_http(cfg, book_id, status)
    in_block = [seg for ch in manifest.chapters for seg in ch.segments if seg.block_id == block_id]
    if not in_block:
        raise ApiError(404, "not_found", f"no rendered segments for block {block_id!r}")
    if segment >= len(in_block):
        raise ApiError(
            404,
            "not_found",
            f"block {block_id!r} has {len(in_block)} rendered segment(s); "
            f"index {segment} is out of range",
        )
    seg = in_block[segment]
    if not seg.wav:
        raise ApiError(
            404,
            "not_found",
            f"segment {segment} of block {block_id!r} has no audio (scene break)",
        )
    book_output_dir = (cfg.output_dir / book_id).resolve()
    wav = (book_output_dir / seg.wav).resolve()
    if not wav.is_file() or not wav.is_relative_to(book_output_dir):
        raise ApiError(
            404,
            "not_found",
            f"audio for {block_id!r}[{segment}] is missing on disk; re-run render",
        )
    return FileResponse(wav, media_type="audio/wav", filename=f"{block_id}_{segment}.wav")
