"""Book-scoped helpers shared by the books and review routers."""

import re

from pydantic import ValidationError

from seiyuu.api.errors import ApiError
from seiyuu.api.schemas import JobOut
from seiyuu.ingest.models import NormalizedBook
from seiyuu.repository import BookStatus, JobKind, JobState, JobStore, get_book_status
from seiyuu.repository.books import NORMALIZED_NAME
from seiyuu.services import ServiceError, load_report
from seiyuu.settings import Settings

_BOOK_ID_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def check_book_id(book_id: str) -> None:
    """Path containment: ids are single directory names under the pinned roots."""
    if not _BOOK_ID_OK.match(book_id) or ".." in book_id:
        raise ApiError(422, "invalid", f"invalid book id {book_id!r}")


def status_or_404(cfg: Settings, book_id: str, *, read_meta: bool = False) -> BookStatus:
    """Existence gate on the stage markers. Meta (title/authors) defaults OFF: reading it
    parses the whole normalized.json per call, and this runs at the front of nearly every
    book route — only ``book_detail`` (which serializes the status) opts back in."""
    check_book_id(book_id)
    status = get_book_status(
        book_id, books_dir=cfg.books_dir, output_dir=cfg.output_dir, read_meta=read_meta
    )
    if not any(
        [
            status.ingested,
            status.attributed,
            status.assigned,
            status.rendered,
            status.assembled,
            status.mastered,
        ]
    ):
        raise ApiError(404, "not_found", f"book {book_id!r} not found")
    return status


def load_book(cfg: Settings, book_id: str) -> NormalizedBook:
    path = cfg.books_dir / book_id / NORMALIZED_NAME
    try:
        return NormalizedBook.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError, ValueError) as exc:
        raise ApiError(
            500, "corrupt_artifact", f"corrupt normalized book {path}: {exc}; re-run ingest"
        ) from exc


def effective_report(cfg: Settings, book_id: str, status: BookStatus):
    """(report, edit_warnings) with the doc's error split: missing -> 404, corrupt -> 500."""
    if not status.attributed:
        raise ApiError(
            404, "not_found", f"book {book_id!r} has no attribution; run attribute first"
        )
    try:
        return load_report(cfg.books_dir / book_id)
    except ServiceError as exc:
        raise ApiError(500, "corrupt_artifact", str(exc)) from exc


def guard_render_active(store: JobStore, book_id: str) -> None:
    """409 while a render job for the book is queued/running: an edit or assignment
    write mid-render would only surface HOURS later as the running job's quote-drift
    refusal — refuse early, with the job to cancel in detail (scoping doc section 5)."""
    live = store.list_jobs(book_id=book_id, states=[JobState.QUEUED, JobState.RUNNING])
    render = next((j for j in live if j.kind is JobKind.RENDER), None)
    if render is not None:
        raise ApiError(
            409,
            "render_active",
            f"a render job for {book_id!r} is {render.state.value}; cancel it first — "
            "this write would invalidate its cost quote mid-run",
            detail=JobOut.from_job(render).model_dump(mode="json"),
        )
