"""Series / library voice consistency (F5): declare a series, link a returning character's
voice across its books, and inherit those voices into a sibling book's cast.

All additive: no frozen-format, prompt, or render change. Inheritance is served as the
``overrides`` dict the existing ``POST /assignment/draft`` seam already accepts — this router
never renders and never mutates a book's assignment. The global ``series.json`` is the only
state written here; every write serializes on the process-wide series mutex so a concurrent
read-modify-write can't clobber a link.

Precision over recall: cross-book links are only ever SUGGESTED (``/link-suggestions``) for the
user to confirm, and matching is scoped to a declared series — there is no global name match, so
two same-named characters in unrelated books are never joined onto one voice.
"""

import threading

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, ValidationError

from seiyuu.api.deps import SettingsDep
from seiyuu.api.errors import ApiError
from seiyuu.api.routes.common import effective_report, status_or_404
from seiyuu.services import ServiceError, load_assignment
from seiyuu.settings import Settings
from seiyuu.voices import (
    LinkSuggestion,
    Series,
    SeriesRegistry,
    VoiceAssignment,
    VoiceLibrary,
    identity_key,
    load_registry,
    make_series_id,
    prune_dangling_links,
    resolve_series_overrides,
    save_cast_to_series,
    save_registry,
    seed_voice_links,
    suggest_links,
)

router = APIRouter(tags=["series"])


def _series_mutex(request: Request) -> threading.Lock:
    return request.app.state.series_mutex


def _load_registry(cfg: Settings) -> SeriesRegistry:
    try:
        return load_registry(cfg.data_dir)
    except ValidationError as exc:
        raise ApiError(500, "corrupt_artifact", f"corrupt series.json: {exc}") from exc


def _series_or_404(registry: SeriesRegistry, series_id: str) -> Series:
    series = registry.get(series_id)
    if series is None:
        raise ApiError(404, "not_found", f"series {series_id!r} not found")
    return series


def _report(cfg: Settings, book_id: str):
    status = status_or_404(cfg, book_id)
    report, _warnings = effective_report(cfg, book_id, status)
    return report


def _assignment(cfg: Settings, book_id: str) -> VoiceAssignment:
    try:
        return load_assignment(cfg.output_dir, book_id)
    except ServiceError as exc:
        raise ApiError(409, "stage_prerequisite", str(exc)) from exc


# -- create / list / get ------------------------------------------------------------------


class SeriesCreate(BaseModel):
    name: str = Field(min_length=1)
    book_id: str  # the first book; its cast seeds the series' voice_links
    series_id: str | None = None  # injectable for deterministic tests


class SeriesListOut(BaseModel):
    series: list[Series]


@router.post("/series", response_model=Series, status_code=201)
def create_series(body: SeriesCreate, request: Request, cfg: SettingsDep) -> Series:
    """Create a series SEEDED from a book: its assignment's cast becomes the initial
    ``voice_links`` (``canonical_name -> voice_id``). The book must be attributed and assigned."""
    report = _report(cfg, body.book_id)
    assignment = _assignment(cfg, body.book_id)
    with _series_mutex(request):
        registry = _load_registry(cfg)
        series_id = body.series_id or make_series_id(body.name)
        if registry.get(series_id) is not None:
            raise ApiError(409, "conflict", f"series {series_id!r} already exists")
        series = Series(
            series_id=series_id,
            name=body.name.strip(),
            book_ids=[body.book_id],
            voice_links=seed_voice_links(report, assignment),
        )
        registry.upsert(series)
        save_registry(cfg.data_dir, registry)
    return series


@router.get("/series", response_model=SeriesListOut)
def list_series(cfg: SettingsDep) -> SeriesListOut:
    return SeriesListOut(series=_load_registry(cfg).series)


@router.get("/series/{series_id}", response_model=Series)
def get_series(series_id: str, cfg: SettingsDep) -> Series:
    return _series_or_404(_load_registry(cfg), series_id)


# -- book membership ----------------------------------------------------------------------


class BookRef(BaseModel):
    book_id: str


@router.post("/series/{series_id}/books", response_model=Series)
def add_book(series_id: str, body: BookRef, request: Request, cfg: SettingsDep) -> Series:
    """Attach a book to the series (idempotent). Adding a book does NOT learn its cast — that is
    the explicit ``/save-cast`` action, so precision is preserved."""
    status_or_404(cfg, body.book_id)  # the book must exist
    with _series_mutex(request):
        registry = _load_registry(cfg)
        series = _series_or_404(registry, series_id)
        if body.book_id not in series.book_ids:
            series.book_ids.append(body.book_id)
            registry.upsert(series)
            save_registry(cfg.data_dir, registry)
    return series


@router.delete("/series/{series_id}/books/{book_id}", response_model=Series)
def remove_book(
    series_id: str, book_id: str, request: Request, cfg: SettingsDep, prune: bool = False
) -> Series:
    """Drop a book from the series' membership. Optionally (``?prune=true``) also prune links
    whose voice no longer exists in the library. Voice links are name-keyed, so a plain removal
    leaves them intact (a re-added book re-inherits)."""
    with _series_mutex(request):
        registry = _load_registry(cfg)
        series = _series_or_404(registry, series_id)
        series.book_ids = [b for b in series.book_ids if b != book_id]
        if prune:
            prune_dangling_links(series, VoiceLibrary(cfg.voices_dir))
        registry.upsert(series)
        save_registry(cfg.data_dir, registry)
    return series


# -- link suggestions + inheritance (the draft overrides) ---------------------------------


class LinkSuggestionsOut(BaseModel):
    series_id: str
    book_id: str
    suggestions: list[LinkSuggestion]


@router.get(
    "/series/{series_id}/books/{book_id}/link-suggestions", response_model=LinkSuggestionsOut
)
def link_suggestions(series_id: str, book_id: str, cfg: SettingsDep) -> LinkSuggestionsOut:
    """Cross-book link suggestions for a book joining the series: each character whose name
    matches an existing ``voice_links`` entry, for the user to CONFIRM. Never auto-applied."""
    series = _series_or_404(_load_registry(cfg), series_id)
    report = _report(cfg, book_id)
    return LinkSuggestionsOut(
        series_id=series_id,
        book_id=book_id,
        suggestions=suggest_links(report, series, VoiceLibrary(cfg.voices_dir)),
    )


class OverridesOut(BaseModel):
    series_id: str
    book_id: str
    overrides: dict[str, str]  # char_id -> inherited voice_id (deleted-voice links skipped)


@router.get("/series/{series_id}/books/{book_id}/overrides", response_model=OverridesOut)
def draft_overrides(series_id: str, book_id: str, cfg: SettingsDep) -> OverridesOut:
    """The resolved inheritance for a book: ``{char_id -> voice_id}`` for confirmed links whose
    voice still exists. Feed this straight into ``POST /books/{id}/assignment/draft`` as
    ``overrides`` — inheritance is supplying overrides that win over the auto-cast."""
    series = _series_or_404(_load_registry(cfg), series_id)
    report = _report(cfg, book_id)
    return OverridesOut(
        series_id=series_id,
        book_id=book_id,
        overrides=resolve_series_overrides(report, series, VoiceLibrary(cfg.voices_dir)),
    )


# -- explicit write-back + unlink ---------------------------------------------------------


class SaveCastOut(BaseModel):
    series: Series
    linked_keys: list[str]  # identity keys added or updated by this write-back


@router.post("/series/{series_id}/save-cast", response_model=SaveCastOut)
def save_cast(series_id: str, body: BookRef, request: Request, cfg: SettingsDep) -> SaveCastOut:
    """Explicit save-to-series: fold a book's cast into ``voice_links`` (last-write-wins). The
    only path that grows a series' links from a book — nothing is learned silently."""
    report = _report(cfg, body.book_id)
    assignment = _assignment(cfg, body.book_id)
    with _series_mutex(request):
        registry = _load_registry(cfg)
        series = _series_or_404(registry, series_id)
        linked = save_cast_to_series(series, report, assignment)
        if body.book_id not in series.book_ids:
            series.book_ids.append(body.book_id)
        registry.upsert(series)
        save_registry(cfg.data_dir, registry)
    return SaveCastOut(series=series, linked_keys=sorted(linked))


@router.delete("/series/{series_id}/links", response_model=Series)
def unlink(series_id: str, name: str, request: Request, cfg: SettingsDep) -> Series:
    """Remove a character's voice link by NAME (case-insensitive). Idempotent — unlinking an
    absent name is a success."""
    with _series_mutex(request):
        registry = _load_registry(cfg)
        series = _series_or_404(registry, series_id)
        series.voice_links.pop(identity_key(name), None)
        registry.upsert(series)
        save_registry(cfg.data_dir, registry)
    return series
