"""Series / library voice consistency (F5): reuse a character's voice across books.

A GLOBAL registry at ``<data_dir>/series.json`` (the home for non-per-book state, alongside
``jobs.db``) maps a cross-book character IDENTITY to a library ``voice_id``. Registry character
ids differ across books, so the identity key is ``casefold(canonical_name)`` (a NAME, not an id).

Inheritance is deliberately NOT a new render path: :func:`resolve_series_overrides` turns a
series' ``voice_links`` into the ``overrides`` dict the existing ``draft_assignment`` seam
already validates and lets win over the auto-cast. A ``voice_id`` from book A is a legal value
in book B's assignment because the voice library is already global and book-independent — no
new voice storage, no re-consent, no frozen-format change.

Precision over recall (mirrors the alias adjudicator): cross-book matching is scoped to a
DECLARED series and surfaced as SUGGESTIONS the user confirms — two same-named characters in
unrelated books are never auto-merged onto one voice, because there is no global name match.
A linked voice that has since been deleted from the library is SKIPPED (the book degrades to a
fresh cast), never an error — the same validation the override path already does.
"""

import secrets
from pathlib import Path

from pydantic import BaseModel, Field

from seiyuu.attribute.models import AttributionReport
from seiyuu.repository.atomic import atomic_write_text
from seiyuu.voices.assignment import VoiceAssignment
from seiyuu.voices.library import VoiceLibrary, slugify

SERIES_NAME = "series.json"
SERIES_SCHEMA_VERSION = 1


def identity_key(name: str) -> str:
    """The cross-book identity for a character NAME: whitespace-stripped, casefolded. Registry
    ids are per-book, so consistency keys on the (canonical) name, not the id."""
    return name.strip().casefold()


class Series(BaseModel):
    """One declared series. ``voice_links`` maps ``identity_key(canonical_name) -> voice_id``;
    ``book_ids`` is a plain membership list (last-write-wins, no cascading deletes)."""

    series_id: str
    name: str
    book_ids: list[str] = Field(default_factory=list)
    voice_links: dict[str, str] = Field(default_factory=dict)


class SeriesRegistry(BaseModel):
    """The global ``series.json`` holder — every declared series in one file."""

    schema_version: int = SERIES_SCHEMA_VERSION
    series: list[Series] = Field(default_factory=list)

    def get(self, series_id: str) -> Series | None:
        return next((s for s in self.series if s.series_id == series_id), None)

    def upsert(self, series: Series) -> None:
        """Replace an existing series with the same id, else append."""
        for i, existing in enumerate(self.series):
            if existing.series_id == series.series_id:
                self.series[i] = series
                return
        self.series.append(series)


# -- persistence (mirrors ingest.write_normalized / lexicon save) --------------------------


def series_path(data_dir: Path) -> Path:
    return Path(data_dir) / SERIES_NAME


def load_registry(data_dir: Path) -> SeriesRegistry:
    """Load ``<data_dir>/series.json``; an absent file is an empty registry (not an error).
    A corrupt file raises loudly (pydantic ValidationError)."""
    path = series_path(data_dir)
    if not path.is_file():
        return SeriesRegistry()
    return SeriesRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def save_registry(data_dir: Path, registry: SeriesRegistry) -> Path:
    return atomic_write_text(series_path(data_dir), registry.model_dump_json(indent=2))


def make_series_id(name: str, *, suffix: str | None = None) -> str:
    """A stable-ish slug from ``name`` plus a short hex disambiguator, so two series can share
    a display name. ``suffix`` is injectable for deterministic tests."""
    return f"{slugify(name) or 'series'}_{suffix or secrets.token_hex(3)}"


# -- seeding + write-back ------------------------------------------------------------------


def seed_voice_links(report: AttributionReport, assignment: VoiceAssignment) -> dict[str, str]:
    """A book's cast projected onto the cross-book key space: for each character that has an
    assigned voice, ``identity_key(canonical_name) -> voice_id``. Narrator/thought voices are
    NOT character links and are excluded. Used both to SEED a new series from its first book and
    to WRITE BACK a later book's cast (``dict.update``, last-write-wins)."""
    links: dict[str, str] = {}
    for char in report.registry.characters:
        voice_id = assignment.assignments.get(char.id)
        if voice_id is not None:
            links[identity_key(char.canonical_name)] = voice_id
    return links


def save_cast_to_series(
    series: Series, report: AttributionReport, assignment: VoiceAssignment
) -> list[str]:
    """Explicit write-back: fold a book's cast into ``series.voice_links`` (last-write-wins),
    returning the identity keys added or updated. Mutates ``series`` in place; the caller
    persists. This is the ONLY path that grows a series' links from a book — nothing is learned
    silently on draft/confirm."""
    changed: list[str] = []
    for key, voice_id in seed_voice_links(report, assignment).items():
        if series.voice_links.get(key) != voice_id:
            changed.append(key)
        series.voice_links[key] = voice_id
    return changed


# -- the core: inheritance as overrides ----------------------------------------------------


def _character_keys(canonical_name: str, aliases: list[str]) -> list[str]:
    """Identity keys to try for one character: its canonical name first (the primary link
    key), then its aliases (a guarded, alias-aware enhancement). Canonical wins on a tie."""
    keys = [identity_key(canonical_name)]
    keys.extend(identity_key(a) for a in aliases)
    return keys


def resolve_series_overrides(
    report: AttributionReport, series: Series, library: VoiceLibrary
) -> dict[str, str]:
    """PURE inheritance resolver: ``{char_id -> voice_id}`` for every character in ``report``
    whose identity matches ``series.voice_links`` AND whose linked voice STILL EXISTS in the
    library. A missing/deleted linked voice is SKIPPED (the book degrades to a fresh cast),
    never an error — the same existence check the override path does. The result is fed straight
    into ``draft_assignment(overrides=...)``, so inheritance is literally supplying overrides
    that win over the auto-cast; there is no new render machinery."""
    overrides: dict[str, str] = {}
    for char in report.registry.characters:
        voice_id = next(
            (
                series.voice_links[key]
                for key in _character_keys(char.canonical_name, char.aliases)
                if key in series.voice_links
            ),
            None,
        )
        if voice_id is None:
            continue  # this character is not linked in the series
        if not library.meta_path(voice_id).is_file():
            continue  # linked voice was deleted from the library -> degrade to a fresh cast
        overrides[char.id] = voice_id
    return overrides


class LinkSuggestion(BaseModel):
    """A within-series name match surfaced for the user to CONFIRM (never auto-applied). It is
    an inheritance candidate: character ``character_id`` in the joining book matches an existing
    ``voice_links`` entry. ``voice_exists`` is False when the linked voice was deleted — the UI
    shows it as unavailable and :func:`resolve_series_overrides` skips it."""

    character_id: str
    canonical_name: str
    identity_key: str
    voice_id: str
    voice_exists: bool


def suggest_links(
    report: AttributionReport, series: Series, library: VoiceLibrary
) -> list[LinkSuggestion]:
    """Cross-book link suggestions for a book joining ``series``: every character whose identity
    matches an existing ``voice_links`` entry, returned for confirmation. Scoped to THIS series'
    links only — there is no global name match, so a same-named character in an unrelated book is
    never proposed. Deterministic order (by character id)."""
    suggestions: list[LinkSuggestion] = []
    for char in report.registry.characters:
        match = next(
            (
                (key, series.voice_links[key])
                for key in _character_keys(char.canonical_name, char.aliases)
                if key in series.voice_links
            ),
            None,
        )
        if match is None:
            continue
        key, voice_id = match
        suggestions.append(
            LinkSuggestion(
                character_id=char.id,
                canonical_name=char.canonical_name,
                identity_key=key,
                voice_id=voice_id,
                voice_exists=library.meta_path(voice_id).is_file(),
            )
        )
    suggestions.sort(key=lambda s: s.character_id)
    return suggestions


# -- membership drop on book deletion ------------------------------------------------------


def drop_book(registry: SeriesRegistry, book_id: str) -> list[str]:
    """Remove ``book_id`` from every series' ``book_ids`` (mutates in place). Returns the ids of
    the series that changed. Voice links are name-keyed (not book-keyed) and harmless, so they
    are left intact — a re-added book re-inherits them. Deleting a book must leave no dangling
    MEMBERSHIP behind; this is what enforces that."""
    affected: list[str] = []
    for series in registry.series:
        if book_id in series.book_ids:
            series.book_ids = [b for b in series.book_ids if b != book_id]
            affected.append(series.series_id)
    return affected


def prune_dangling_links(series: Series, library: VoiceLibrary) -> list[str]:
    """Drop ``voice_links`` whose ``voice_id`` no longer exists in the library (mutates in
    place). Returns the pruned identity keys. Optional housekeeping — inheritance already SKIPS a
    dangling link, so this only tidies the file."""
    dangling = [k for k, v in series.voice_links.items() if not library.meta_path(v).is_file()]
    for key in dangling:
        del series.voice_links[key]
    return dangling


def drop_book_everywhere(data_dir: Path, book_id: str) -> list[str]:
    """Load the global registry, drop ``book_id`` from every series' membership, and persist iff
    anything changed. Called from the book-delete path (API + CLI) so a deleted book leaves no
    series membership dangling. Returns the affected series ids."""
    registry = load_registry(data_dir)
    affected = drop_book(registry, book_id)
    if affected:
        save_registry(data_dir, registry)
    return affected
