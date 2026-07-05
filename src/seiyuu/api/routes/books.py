"""Books: EPUB ingest (sync — seconds, CPU-only, and the id isn't known pre-parse),
library/detail aggregates, the attribute job, attribution reads, and file downloads.

Exact ids only — no CLI prefix sugar; ambiguity has no place in a REST id. Every
attribution read goes through ``load_report`` (edits overlay applied) and carries
``edit_warnings``; raw ``attribution.json`` is never served. Book payloads deliberately
omit job progress (polling discipline — poll ``/api/jobs/{id}``).
"""

import os
import re
import secrets
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from seiyuu.api.deps import RunnerDep, SettingsDep, StoreDep
from seiyuu.api.enqueue import enqueue_job
from seiyuu.api.errors import ApiError
from seiyuu.api.routes.common import (
    effective_report,
    load_book,
    status_or_404,
)
from seiyuu.api.schemas import (
    ActiveJobSummary,
    AttributeParams,
    AttributionOut,
    BookCard,
    BookDeletedOut,
    BookDetail,
    BooksOut,
    ChapterDownload,
    ChapterSummary,
    CoverOut,
    DownloadsOut,
    FileDownload,
    IngestResponse,
    JobOut,
    RuntimeEstimateOut,
    SegmentBrowserOut,
    SegmentRow,
)
from seiyuu.duration import estimate_runtime_seconds, format_hms
from seiyuu.ingest import IngestError, parse_epub, write_normalized
from seiyuu.repository import Job, JobKind, JobState, get_book_status, list_books
from seiyuu.repository.books import CHAPTERS_DIR, MANIFEST_NAME, NORMALIZED_NAME, delete_book_trees
from seiyuu.services import ServiceError, characters_overview
from seiyuu.services.characters import CharactersOverview
from seiyuu.services.deletion import detect_paid_artifacts

router = APIRouter(tags=["books"])

_COVER_TYPES = {"cover.jpg": "image/jpeg", "cover.png": "image/png"}
_UPLOAD_CHUNK = 1024 * 1024


def _active_summary(jobs: list[Job]) -> ActiveJobSummary | None:
    live = next((j for j in jobs if j.state is JobState.RUNNING), None) or next(
        (j for j in jobs if j.state is JobState.QUEUED), None
    )
    if live is None:
        return None
    return ActiveJobSummary(job_id=live.job_id, kind=live.kind.value, state=live.state.value)


# -- library and ingest -------------------------------------------------------------------


@router.get("/books", response_model=BooksOut)
def library(cfg: SettingsDep, store: StoreDep) -> BooksOut:
    live = store.list_jobs(states=[JobState.QUEUED, JobState.RUNNING])
    by_book: dict[str, list[Job]] = {}
    for job in live:
        by_book.setdefault(job.book_id, []).append(job)
    return BooksOut(
        books=[
            BookCard(
                **status.model_dump(),
                active_job=_active_summary(by_book.get(status.book_id, [])),
            )
            for status in list_books(books_dir=cfg.books_dir, output_dir=cfg.output_dir)
        ]
    )


def _safe_upload_name(filename: str | None) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename or "book.epub").name).strip()
    if not name or name.startswith("."):
        name = "book.epub"
    return name[-100:]


@router.post("/books", response_model=IngestResponse, status_code=201)
def ingest_book(
    response: Response,
    cfg: SettingsDep,
    store: StoreDep,
    file: Annotated[UploadFile, File()],
    include_item: Annotated[list[str], Form()] = [],  # noqa: B006 — FastAPI default factory
    exclude_item: Annotated[list[str], Form()] = [],  # noqa: B006
    split_level: Annotated[int, Form(ge=1)] = 2,
) -> IngestResponse:
    """Ingest an uploaded EPUB. Identical bytes re-upload is idempotent (the id is
    slug + content sha256[:8]) and answers 200 instead of 201."""
    upload_dir = cfg.data_dir / "uploads" / secrets.token_hex(8)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = upload_dir / _safe_upload_name(file.filename)
    try:
        size = 0
        with tmp.open("wb") as out:
            while chunk := file.file.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > cfg.max_upload_bytes:
                    raise ApiError(
                        413,
                        "payload_too_large",
                        f"upload exceeds the {cfg.max_upload_bytes}-byte limit "
                        "(see /api/system limits.max_upload_bytes)",
                    )
                out.write(chunk)
        try:
            result = parse_epub(
                tmp,
                include_items=tuple(include_item),
                exclude_items=tuple(exclude_item),
                split_level=split_level,
            )
        except IngestError as exc:
            raise ApiError(422, "invalid", str(exc)) from exc
        book = result.book
        book_id = book.book_meta.book_id
        live = store.list_jobs(book_id=book_id, states=[JobState.QUEUED, JobState.RUNNING])
        if live:
            # overwriting normalized.json re-chapters the book a running job already
            # loaded — the failure would surface as that job's confusing late error
            raise ApiError(
                409,
                "conflicting_job",
                f"a {live[0].kind.value} job for {book_id!r} is {live[0].state.value}; "
                "re-ingesting would rewrite the book underneath it",
                detail=JobOut.from_job(live[0]).model_dump(mode="json"),
            )
        existed = (cfg.books_dir / book_id / NORMALIZED_NAME).is_file()
        write_normalized(book, cfg.books_dir)
        response.status_code = 200 if existed else 201
        response.headers["Location"] = f"/api/books/{book_id}"
        return IngestResponse(
            book=get_book_status(book_id, books_dir=cfg.books_dir, output_dir=cfg.output_dir),
            chapters=len(book.chapters),
            blocks=sum(len(c.blocks) for c in book.chapters),
            skipped_items=result.skipped_items,
            dropped_sections=result.dropped_sections,
        )
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


# -- detail and estimates -----------------------------------------------------------------


@router.get("/books/{book_id}", response_model=BookDetail)
def book_detail(book_id: str, cfg: SettingsDep, store: StoreDep) -> BookDetail:
    status = status_or_404(cfg, book_id)
    chapters = None
    runtime = None
    if status.ingested:
        book = load_book(cfg, book_id)
        chapters = [
            ChapterSummary(
                index=i,
                title=c.title,
                blocks=len(c.blocks),
                speakable_blocks=sum(1 for b in c.blocks if b.is_speakable),
            )
            for i, c in enumerate(book.chapters, start=1)
        ]
        runtime = estimate_runtime_seconds(book, wpm=cfg.narration_wpm)

    odir = cfg.output_dir / book_id
    m4b = odir / f"{book_id}.m4b"
    downloads = DownloadsOut(
        m4b=(
            FileDownload(url=f"/api/books/{book_id}/files/m4b", bytes=m4b.stat().st_size)
            if m4b.is_file()
            else None
        ),
        chapter_mp3s=[
            ChapterDownload(
                index=int(p.stem[2:]),
                url=f"/api/books/{book_id}/files/chapters/{int(p.stem[2:])}",
                bytes=p.stat().st_size,
            )
            for p in sorted((odir / CHAPTERS_DIR).glob("ch*.mp3"))
            if p.stem[2:].isdigit()
        ],
    )
    cover = next(
        (
            CoverOut(content_type=ctype, bytes=(odir / name).stat().st_size)
            for name, ctype in _COVER_TYPES.items()
            if (odir / name).is_file()
        ),
        None,
    )
    return BookDetail(
        status=status,
        chapters=chapters,
        runtime_estimate_seconds=runtime,
        active_job=_active_summary(
            store.list_jobs(book_id=book_id, states=[JobState.QUEUED, JobState.RUNNING])
        ),
        recent_jobs=[JobOut.from_job(j) for j in store.list_jobs(book_id=book_id, limit=10)],
        downloads=downloads,
        cover=cover,
    )


def _guard_any_active(store, book_id: str) -> None:
    """Kind-agnostic live-job guard (sign-off D7): refuse deletion while ANY job for the
    book is queued or running. Attribute/render/assemble all read or write the book's trees
    and a queued job starts imminently, so 'no deletion while any work is pending or active'
    is the simplest correct rule. WARMUP jobs never match — their ``book_id`` is the
    ``engine:{id}`` overload — so an engine pre-load never blocks a book deletion."""
    live = store.list_jobs(book_id=book_id, states=[JobState.QUEUED, JobState.RUNNING])
    if live:
        raise ApiError(
            409,
            "conflicting_job",
            f"a {live[0].kind.value} job for {book_id!r} is {live[0].state.value}; cancel "
            "or wait for it before deleting the book",
            detail=JobOut.from_job(live[0]).model_dump(mode="json"),
        )


@router.delete("/books/{book_id}", response_model=BookDeletedOut)
def delete_book(
    book_id: str,
    request: Request,
    cfg: SettingsDep,
    store: StoreDep,
    confirm_paid: Annotated[bool, Query()] = False,
) -> BookDeletedOut:
    """Purge a book from BOTH on-disk roots (``output/{id}`` then ``books/{id}``) and reap
    its terminal job rows. Refused **409 conflicting_job** while any job for the book is
    live. PAID cloud renders (ElevenLabs/Fish) are NEVER discarded automatically: a book that
    owns any answers **402 payment_confirmation_required** and NOTHING is removed until a
    re-send with ``?confirm_paid=true``; a book with only free/local renders deletes in that
    single call. The shared voice library, the global jobs.db file, and every OTHER book are
    left untouched. Guard + purge run under the enqueue mutex so a job cannot be created for
    the book between the live-job check and the purge."""
    status_or_404(cfg, book_id)
    with request.app.state.enqueue_mutex:
        _guard_any_active(store, book_id)
        paid = detect_paid_artifacts(cfg, book_id)
        if paid.paid_segment_count > 0 and not confirm_paid:
            raise ApiError(
                402,
                "payment_confirmation_required",
                f"deleting {book_id!r} discards {paid.paid_segment_count} paid cloud "
                "segment(s) that cost real money to reproduce; re-send with "
                "confirm_paid=true to approve discarding them",
                detail=paid.model_dump(mode="json"),
            )
        result = delete_book_trees(book_id, books_dir=cfg.books_dir, output_dir=cfg.output_dir)
        if result.survivors:
            # Jobs rows are NOT deleted yet, so a retry is idempotent: the book still
            # resolves, the guard passes, and the leftover files are rmtree'd again.
            raise ApiError(
                500,
                "partial_delete",
                f"book {book_id!r} was only partially deleted; {len(result.survivors)} "
                "path(s) could not be removed (a file may be open) — close any process "
                "holding them and retry",
                detail={"survivors": result.survivors},
            )
        jobs_deleted = store.delete_jobs_for_book(book_id)
    return BookDeletedOut(
        book_id=book_id,
        output_removed=result.output_removed,
        books_removed=result.books_removed,
        jobs_rows_deleted=jobs_deleted,
        paid_segments_discarded=paid.paid_segment_count,
    )


@router.get("/books/{book_id}/runtime-estimate", response_model=RuntimeEstimateOut)
def runtime_estimate(
    book_id: str,
    cfg: SettingsDep,
    chapters: Annotated[list[int], Query()] = [],  # noqa: B006
    wpm: Annotated[float | None, Query(gt=0)] = None,
) -> RuntimeEstimateOut:
    status = status_or_404(cfg, book_id)
    if not status.ingested:
        raise ApiError(404, "not_found", f"book {book_id!r} is not ingested; run ingest first")
    book = load_book(cfg, book_id)
    wanted = sorted(set(chapters))
    for index in wanted:
        if index < 1 or index > len(book.chapters):
            raise ApiError(
                422, "invalid", f"chapter {index} out of range (book has {len(book.chapters)})"
            )
    wpm_used = wpm if wpm is not None else cfg.narration_wpm
    seconds = estimate_runtime_seconds(book, wpm=wpm_used, chapters=tuple(wanted))
    return RuntimeEstimateOut(
        seconds=seconds, formatted=format_hms(seconds), wpm_used=wpm_used, chapters=wanted
    )


# -- attribution --------------------------------------------------------------------------


@router.post("/books/{book_id}/attribute", response_model=JobOut, status_code=202)
def attribute_book_job(
    book_id: str,
    params: AttributeParams,
    request: Request,
    response: Response,
    cfg: SettingsDep,
    store: StoreDep,
    runner: RunnerDep,
) -> JobOut:
    status = status_or_404(cfg, book_id)
    if not status.ingested:
        raise ApiError(
            409, "stage_prerequisite", f"book {book_id!r} is not ingested; run ingest first"
        )
    book = load_book(cfg, book_id)
    for index in params.chapters:
        if index < 1 or index > len(book.chapters):
            raise ApiError(
                422, "invalid", f"chapter {index} out of range (book has {len(book.chapters)})"
            )

    # PAID gate on the EFFECTIVE values: a .env default of attribution_hybrid=true (or
    # provider=anthropic) can never silently run paid attribution over HTTP.
    effective_provider = params.provider or cfg.attribution_provider
    effective_hybrid = cfg.attribution_hybrid if params.use_hybrid is None else params.use_hybrid
    if effective_provider == "anthropic" or effective_hybrid:
        if not params.confirm_paid:
            raise ApiError(
                402,
                "payment_confirmation_required",
                "attribution with provider=anthropic or hybrid escalation calls the paid "
                "Anthropic API; re-send with confirm_paid=true to approve the spend "
                "(confirmed-but-uncapped in M6b, per sign-off Q4)",
            )
        if not cfg.anthropic_api_key:
            raise ApiError(
                503, "not_ready", "ANTHROPIC_API_KEY not set; required for anthropic/hybrid"
            )

    job = enqueue_job(
        store=store,
        runner=runner,
        mutex=request.app.state.enqueue_mutex,
        book_id=book_id,
        kind=JobKind.ATTRIBUTE,
        params=params.model_dump(),
    )
    response.headers["Location"] = f"/api/jobs/{job.job_id}"
    return JobOut.from_job(job)


@router.get("/books/{book_id}/attribution", response_model=AttributionOut)
def attribution_report(
    book_id: str,
    cfg: SettingsDep,
    chapters: Annotated[list[int], Query()] = [],  # noqa: B006
) -> AttributionOut:
    status = status_or_404(cfg, book_id)
    report, warnings = effective_report(cfg, book_id, status)
    if chapters:
        wanted = set(chapters)
        report = report.model_copy(
            update={"chapters": [c for c in report.chapters if c.index in wanted]}
        )  # registry / flagged / notes stay full — the filter is a payload trim, not a view
    return AttributionOut(report=report, edit_warnings=warnings)


@router.get("/books/{book_id}/chapters/{index}/segments", response_model=SegmentBrowserOut)
def segment_browser(
    book_id: str,
    index: int,
    cfg: SettingsDep,
    speaker: Annotated[str | None, Query()] = None,  # character id, or literal "narration"
    type: Annotated[str | None, Query(alias="type")] = None,
    low_confidence: bool = False,
) -> SegmentBrowserOut:
    status = status_or_404(cfg, book_id)
    report, warnings = effective_report(cfg, book_id, status)
    chapter = next((c for c in report.chapters if c.index == index), None)
    if chapter is None:
        raise ApiError(404, "not_found", f"chapter {index} has no attribution")
    names = {c.id: c.canonical_name for c in report.registry.characters}

    manifest_path = cfg.output_dir / book_id / MANIFEST_NAME
    block_audio: dict[str, list] = {}
    if manifest_path.is_file():
        from seiyuu.render.models import RenderManifest

        try:
            manifest = RenderManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (ValidationError, OSError, ValueError) as exc:
            raise ApiError(
                500, "corrupt_artifact", f"corrupt render manifest {manifest_path}: {exc}"
            ) from exc
        for man_chapter in manifest.chapters:
            for man_seg in man_chapter.segments:
                block_audio.setdefault(man_seg.block_id, []).append(man_seg)

    # Rendered-audio alignment for the Listen read-along: a MULTIVOICE render emits one
    # manifest segment per attribution segment (1:1 by position); a SINGLE-VOICE render
    # emits one per BLOCK, which every attribution segment of that block shares. Any
    # other shape (re-attribution changed the splits since the render) is ambiguous —
    # has_audio stays truthful but no timing is claimed.
    rows_per_block: dict[str, int] = {}
    for seg in chapter.segments:
        rows_per_block[seg.block_id] = rows_per_block.get(seg.block_id, 0) + 1

    def audio_for(block_id: str, seg_index: int):
        rendered = block_audio.get(block_id)
        if not rendered:
            return (False, None, None, None, None)
        if len(rendered) == rows_per_block[block_id]:
            man_seg = rendered[seg_index]
        elif len(rendered) == 1:
            man_seg = rendered[0]
        else:
            return (any(m.wav for m in rendered), None, None, None, None)
        if not man_seg.wav:
            return (False, None, None, None, None)
        # audio_key = the wav stem (the frozen SegmentKey hash): render identity for the
        # UI and an exact cache-buster — it changes iff the audio content would
        return (
            True,
            rendered.index(man_seg),
            man_seg.duration_seconds,
            man_seg.voice_id,
            Path(man_seg.wav).stem,
        )

    rows: list[SegmentRow] = []
    position_in_block: dict[str, int] = {}
    for seg in chapter.segments:
        # segment_index counts ALL of the block's segments (pre-filter) — it must stay
        # exactly what ReassignSegment expects, regardless of the view's filters.
        seg_index = position_in_block.get(seg.block_id, 0)
        position_in_block[seg.block_id] = seg_index + 1
        if speaker is not None:
            if speaker == "narration":
                if seg.speaker is not None:
                    continue
            elif seg.speaker != speaker:
                continue
        if type is not None and seg.type.value != type:
            continue
        if low_confidence and seg.confidence >= cfg.attribution_confidence_threshold:
            continue
        has_audio, audio_segment, duration, voice_id, audio_key = audio_for(seg.block_id, seg_index)
        rows.append(
            SegmentRow(
                block_id=seg.block_id,
                segment_index=seg_index,
                type=seg.type.value,
                speaker=seg.speaker,
                speaker_name=names.get(seg.speaker) if seg.speaker else None,
                text=seg.text,
                confidence=seg.confidence,
                has_audio=has_audio,
                audio_segment=audio_segment,
                duration_seconds=duration,
                voice_id=voice_id,
                audio_key=audio_key,
            )
        )
    return SegmentBrowserOut(
        chapter_index=index, title=chapter.title, segments=rows, edit_warnings=warnings
    )


@router.get("/books/{book_id}/characters", response_model=CharactersOverview)
def characters(
    book_id: str,
    cfg: SettingsDep,
    sample_lines: Annotated[int, Query(ge=0, le=10)] = 2,
) -> CharactersOverview:
    status = status_or_404(cfg, book_id)
    if not status.attributed:
        raise ApiError(
            404, "not_found", f"book {book_id!r} has no attribution; run attribute first"
        )
    try:
        return characters_overview(
            cfg.books_dir / book_id,
            confidence_threshold=cfg.attribution_confidence_threshold,
            sample_lines=sample_lines,
        )
    except ServiceError as exc:
        raise ApiError(500, "corrupt_artifact", str(exc)) from exc


# -- cover art ----------------------------------------------------------------------------

_COVER_MAGIC = {
    "image/jpeg": (b"\xff\xd8\xff", "cover.jpg"),
    "image/png": (b"\x89PNG", "cover.png"),
}


def _guard_master_active(store, book_id: str) -> None:
    """A master job reads the cover mid-run (and on Windows, unlinking a file ffmpeg
    holds open raises a sharing violation) — refuse cover writes while one is live."""
    live = store.list_jobs(book_id=book_id, states=[JobState.QUEUED, JobState.RUNNING])
    master = next((j for j in live if j.kind is JobKind.MASTER), None)
    if master is not None:
        raise ApiError(
            409,
            "conflicting_job",
            f"a master job for {book_id!r} is {master.state.value}; it reads the cover "
            "mid-run — wait or cancel it before changing cover art",
            detail=JobOut.from_job(master).model_dump(mode="json"),
        )


@router.put("/books/{book_id}/cover", response_model=CoverOut)
def upload_cover(
    book_id: str, cfg: SettingsDep, store: StoreDep, file: Annotated[UploadFile, File()]
) -> CoverOut:
    """Cover art for mastering (replaces the CLI's `master --cover`). Content type AND
    magic bytes are checked; the write is atomic and evicts the other extension so a
    book never carries two covers."""
    status_or_404(cfg, book_id)
    _guard_master_active(store, book_id)
    content_type = (file.content_type or "").lower()
    if content_type not in _COVER_MAGIC:
        raise ApiError(
            415,
            "unsupported_media_type",
            f"cover must be image/jpeg or image/png, got {content_type or 'unknown'}",
        )
    magic, target_name = _COVER_MAGIC[content_type]
    data = file.file.read(cfg.max_upload_bytes + 1)
    if len(data) > cfg.max_upload_bytes:
        raise ApiError(413, "payload_too_large", "cover exceeds the upload limit")
    if not data.startswith(magic):
        raise ApiError(415, "unsupported_media_type", "file content does not match its image type")
    odir = cfg.output_dir / book_id
    odir.mkdir(parents=True, exist_ok=True)
    for other in _COVER_TYPES:
        if other != target_name:
            (odir / other).unlink(missing_ok=True)
    target = odir / target_name
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return CoverOut(content_type=content_type, bytes=len(data))


@router.get("/books/{book_id}/cover")
def get_cover(book_id: str, cfg: SettingsDep) -> FileResponse:
    """Serve the uploaded cover art (the M6c shelf shows books by cover)."""
    status_or_404(cfg, book_id)
    odir = cfg.output_dir / book_id
    for name, ctype in _COVER_TYPES.items():
        path = odir / name
        if path.is_file():
            return FileResponse(path, media_type=ctype)
    raise ApiError(404, "not_found", f"book {book_id!r} has no cover uploaded")


@router.delete("/books/{book_id}/cover", status_code=204)
def delete_cover(book_id: str, cfg: SettingsDep, store: StoreDep) -> None:
    """Idempotent: removing an absent cover is a success, not an error."""
    status_or_404(cfg, book_id)
    _guard_master_active(store, book_id)
    odir = cfg.output_dir / book_id
    for name in _COVER_TYPES:
        (odir / name).unlink(missing_ok=True)


# -- downloads ----------------------------------------------------------------------------


@router.get("/books/{book_id}/files/m4b")
def download_m4b(book_id: str, cfg: SettingsDep) -> FileResponse:
    status_or_404(cfg, book_id)
    path = cfg.output_dir / book_id / f"{book_id}.m4b"
    if not path.is_file():
        raise ApiError(404, "not_found", f"book {book_id!r} has no mastered m4b; run master first")
    return FileResponse(path, media_type="audio/mp4", filename=f"{book_id}.m4b")


@router.get("/books/{book_id}/files/chapters/{index}")
def download_chapter_mp3(book_id: str, index: int, cfg: SettingsDep) -> FileResponse:
    status_or_404(cfg, book_id)
    path = cfg.output_dir / book_id / CHAPTERS_DIR / f"ch{index:03d}.mp3"
    if not path.is_file():
        raise ApiError(404, "not_found", f"no assembled chapter {index}; run assemble first")
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)
