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
