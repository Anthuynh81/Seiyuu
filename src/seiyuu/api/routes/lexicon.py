"""Per-book pronunciation lexicon (F3): read, write, preview affected segments, and
deterministic hard-name suggestions.

The lexicon is user INPUT stored at ``books/{id}/lexicon.json``. A grapheme respelling is
spoken as written on every engine; an optional per-entry IPA is honored ONLY on the Kokoro
profile (ignored on validated engines), so the mechanism itself confines IPA to Kokoro — no
engine-scoping field is needed. Editing the lexicon re-synthesizes only the segments whose
normalized text changes; the preview endpoint counts them before a save.
"""

from typing import Annotated

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field, ValidationError

from seiyuu.api.deps import SettingsDep
from seiyuu.api.errors import ApiError
from seiyuu.api.routes.common import load_book, status_or_404
from seiyuu.ingest.models import NormalizedBook
from seiyuu.normalize import normalize_text
from seiyuu.normalize.lexicon import (
    BookLexicon,
    CompiledLexicon,
    LexiconEntry,
    compile_lexicon,
    load_lexicon,
    save_lexicon,
    suggest_terms,
)
from seiyuu.normalize.lexicon import (
    SuggestedTerm as _SuggestedTerm,
)
from seiyuu.normalize.respell import RespellSuggestion
from seiyuu.services.llm_advisory import resolve_advisory, run_respell_suggestions

router = APIRouter(tags=["lexicon"])


class LexiconOut(BaseModel):
    book_id: str
    schema_version: int
    entries: list[LexiconEntry]
    suggestions: list[_SuggestedTerm]


class LexiconWrite(BaseModel):
    entries: list[LexiconEntry] = Field(default_factory=list)


class LexiconSaved(BaseModel):
    book_id: str
    schema_version: int
    entries: list[LexiconEntry]
    affected_blocks: int  # speakable blocks whose normalized text changed vs the previous save
    total_speakable_blocks: int


class LexiconPreviewOut(BaseModel):
    affected_blocks: int
    total_speakable_blocks: int


class RespellSuggestRequest(BaseModel):
    """F3 explicit-action request. ``terms`` is the list to respell; empty means "use the
    deterministic hard-name suggestions for this book". ``provider`` overrides cfg.respell_provider
    ("local" free / "anthropic" PAID); ``confirm_paid`` is the paid gate for anthropic."""

    terms: list[str] = Field(default_factory=list)
    provider: str | None = None
    confirm_paid: bool = False


class RespellSuggestOut(BaseModel):
    provider: str
    model: str
    suggestions: list[RespellSuggestion]


def _speakable_texts(book: NormalizedBook) -> list[str]:
    return [b.text for c in book.chapters for b in c.blocks if b.is_speakable]


def _affected_blocks(book: NormalizedBook, old: CompiledLexicon, new: CompiledLexicon) -> int:
    """Count speakable blocks whose normalized synthesis text changes between two lexicons.
    Checks BOTH the shared (default) profile and the Kokoro profile so an IPA-only edit — which
    only alters the Kokoro output — is still counted."""
    changed = 0
    for text in _speakable_texts(book):
        for profile in ("default", "kokoro"):
            if normalize_text(text, profile=profile, lexicon=old) != normalize_text(
                text, profile=profile, lexicon=new
            ):
                changed += 1
                break
    return changed


def _load_lexicon_or_500(cfg, book_id: str) -> BookLexicon:
    try:
        return load_lexicon(cfg.books_dir / book_id, book_id=book_id)
    except ValidationError as exc:
        raise ApiError(500, "corrupt_artifact", f"corrupt lexicon for {book_id!r}: {exc}") from exc


@router.get("/books/{book_id}/lexicon", response_model=LexiconOut)
def get_lexicon(book_id: str, cfg: SettingsDep) -> LexiconOut:
    status = status_or_404(cfg, book_id)
    lexicon = _load_lexicon_or_500(cfg, book_id)
    suggestions: list[_SuggestedTerm] = []
    if status.ingested:
        book = load_book(cfg, book_id)
        suggestions = suggest_terms(
            _speakable_texts(book), existing_terms=[e.term for e in lexicon.entries]
        )
    return LexiconOut(
        book_id=book_id,
        schema_version=lexicon.schema_version,
        entries=lexicon.entries,
        suggestions=suggestions,
    )


def _reject_duplicate_terms(entries: list[LexiconEntry]) -> None:
    seen: set[str] = set()
    for entry in entries:
        # case-insensitive terms collapse; a case-sensitive term is distinct by exact spelling
        key = entry.term if entry.case_sensitive else entry.term.casefold()
        if key in seen:
            raise ApiError(422, "invalid", f"duplicate lexicon term {entry.term!r}")
        seen.add(key)


@router.put("/books/{book_id}/lexicon", response_model=LexiconSaved)
def put_lexicon(book_id: str, cfg: SettingsDep, body: LexiconWrite) -> LexiconSaved:
    status = status_or_404(cfg, book_id)
    _reject_duplicate_terms(body.entries)
    previous = _load_lexicon_or_500(cfg, book_id)
    new = BookLexicon(book_id=book_id, entries=body.entries)

    affected = total = 0
    if status.ingested:
        book = load_book(cfg, book_id)
        total = len(_speakable_texts(book))
        affected = _affected_blocks(book, compile_lexicon(previous), compile_lexicon(new))

    save_lexicon(cfg.books_dir / book_id, new)
    return LexiconSaved(
        book_id=book_id,
        schema_version=new.schema_version,
        entries=new.entries,
        affected_blocks=affected,
        total_speakable_blocks=total,
    )


@router.post("/books/{book_id}/lexicon/suggest-respellings", response_model=RespellSuggestOut)
def suggest_respellings_route(
    book_id: str,
    cfg: SettingsDep,
    body: Annotated[RespellSuggestRequest, Body()],
) -> RespellSuggestOut:
    """F3: ADVISORY LLM respellings for hard terms, on an explicit action. Writes nothing — the
    user accepts a suggestion into the lexicon separately (PUT), which stays the source of truth.

    The deterministic hard-name surfacer stays the free default (GET /lexicon.suggestions); this
    is the opt-in enrichment. When the resolved provider is anthropic it is a PAID call and needs
    confirm_paid + the key; local Ollama is free but still runs only on this explicit action and
    acquires the GPU through the resource manager."""
    status = status_or_404(cfg, book_id)
    if not status.ingested:
        raise ApiError(
            409, "stage_prerequisite", f"book {book_id!r} is not ingested; run ingest first"
        )

    terms = [t.strip() for t in body.terms if t.strip()]
    if not terms:
        # Fall back to the deterministic hard-name list (same surface as GET /lexicon).
        book = load_book(cfg, book_id)
        lexicon = _load_lexicon_or_500(cfg, book_id)
        terms = [
            s.term
            for s in suggest_terms(
                _speakable_texts(book), existing_terms=[e.term for e in lexicon.entries]
            )
        ]

    resolved = resolve_advisory(cfg, cfg.respell_provider, cfg.respell_model, body.provider)
    if resolved.is_paid:
        if not body.confirm_paid:
            raise ApiError(
                402,
                "payment_confirmation_required",
                "the LLM respell suggester with provider=anthropic calls the paid Anthropic API; "
                "re-send with confirm_paid=true to approve the spend",
            )
        if not cfg.anthropic_api_key:
            raise ApiError(
                503, "not_ready", "ANTHROPIC_API_KEY not set; required for the anthropic suggester"
            )

    if not terms:
        return RespellSuggestOut(
            provider=resolved.provider_id, model=resolved.model, suggestions=[]
        )
    try:
        suggestions = run_respell_suggestions(cfg, resolved, terms)
    except Exception as exc:  # a provider/transport failure must not 500 opaquely
        raise ApiError(502, "upstream_error", f"LLM respell suggester failed: {exc}") from exc
    return RespellSuggestOut(
        provider=resolved.provider_id, model=resolved.model, suggestions=suggestions
    )


@router.post("/books/{book_id}/lexicon/preview", response_model=LexiconPreviewOut)
def preview_lexicon(
    book_id: str,
    cfg: SettingsDep,
    body: Annotated[LexiconWrite, Body()],
) -> LexiconPreviewOut:
    """Affected-segment count for a PROPOSED lexicon vs the current save, WITHOUT persisting —
    so a user sees how many segments a term change re-synthesizes before committing."""
    status = status_or_404(cfg, book_id)
    if not status.ingested:
        raise ApiError(
            409, "stage_prerequisite", f"book {book_id!r} is not ingested; run ingest first"
        )
    _reject_duplicate_terms(body.entries)
    previous = _load_lexicon_or_500(cfg, book_id)
    proposed = BookLexicon(book_id=book_id, entries=body.entries)
    book = load_book(cfg, book_id)
    return LexiconPreviewOut(
        affected_blocks=_affected_blocks(
            book, compile_lexicon(previous), compile_lexicon(proposed)
        ),
        total_speakable_blocks=len(_speakable_texts(book)),
    )
