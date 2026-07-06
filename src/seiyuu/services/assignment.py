"""Assignment service: character→voice drafting and the durable assignments.json write.

Extracted from cli.py so the API and job handlers share one implementation. Drafting is
deterministic: each character gets an auto-blend voice keyed by its character id, so
re-running reproduces the same draft voices and therefore the same segment cache keys.
"""

from pathlib import Path
from typing import NamedTuple

from seiyuu.attribute.models import AttributionReport
from seiyuu.repository import atomic_write_text
from seiyuu.services.common import ServiceError
from seiyuu.voices import (
    ASSIGNMENT_NAME,
    AssignmentStage,
    BlendComponent,
    VoiceAssignment,
    VoiceKind,
    VoiceLibrary,
    VoiceMeta,
    auto_blend_recipe,
    cast_book,
    slugify,
)


def _reserved_presets(library: VoiceLibrary, overrides: dict[str, str] | None) -> set[str]:
    """Presets consumed by the override voices, to reserve out of the smart caster's
    distinct-voice budget (``cast_book(taken=...)``) so an auto voice can never be cast onto
    a preset an override already uses — otherwise two characters render identical audio.

    Only Kokoro preset/blend voices contribute presets; cloned/cloud overrides have none.
    Missing override voices are skipped here (the override-validation loop reports them with a
    clean ServiceError)."""
    taken: set[str] = set()
    for voice_id in (overrides or {}).values():
        if not library.meta_path(voice_id).is_file():
            continue
        meta = library.load(voice_id)
        if meta.kind is VoiceKind.PRESET and meta.preset_id:
            taken.add(meta.preset_id)
        elif meta.kind is VoiceKind.BLEND and meta.blend:
            taken.update(c.preset_id for c in meta.blend)
    return taken


def _auto_voice_meta(
    voice_id: str, name: str, recipe: list[tuple[str, float]], book_id: str
) -> VoiceMeta:
    """A draft VoiceMeta from a smart-cast recipe: a single preset stays a PRESET voice, a
    2-preset recipe is a BLEND (the pool-exhaustion fallback). Tagged like the hash draft so
    the UI groups a book's auto cast identically."""
    common = dict(
        voice_id=voice_id,
        name=name,
        engine="kokoro",
        tags=["auto", book_id],  # the UI maps book_id -> title
    )
    if len(recipe) == 1:
        return VoiceMeta(
            kind=VoiceKind.PRESET, preset_id=recipe[0][0], source="auto_cast", **common
        )
    return VoiceMeta(
        kind=VoiceKind.BLEND,
        blend=[BlendComponent(preset_id=p, weight=w) for p, w in recipe],
        source="auto_cast",
        **common,
    )


def draft_assignment(
    report: AttributionReport,
    library: VoiceLibrary,
    *,
    default_preset: str,
    narrator_voice_id: str | None = None,
    thought_voice_id: str | None = None,
    accent: str = "a",
    stage: AssignmentStage = AssignmentStage.DRAFT,
    overrides: dict[str, str] | None = None,
    strategy: str = "hash",
    recast: bool = False,
) -> VoiceAssignment:
    """Build a VoiceAssignment, creating any missing draft voices in ``library``.

    Narrator is an explicit existing voice or an auto preset for ``default_preset``;
    ``overrides`` maps character ids to explicit voice ids (e.g. cloud voices for the
    final stage) and is validated against the report and the library.

    ``strategy`` selects the auto-caster: ``"hash"`` (legacy) draws each character's blend
    from its own ``name|gender`` hash in isolation; ``"smart"`` runs the book-level
    :func:`cast_book` so no two characters share a voice (and characters covered by
    ``overrides`` are reserved out of its distinct-voice budget). Both are skip-if-exists on
    ``{char_id}_auto``; ``recast=True`` (smart only) OVERWRITES existing auto voices to apply
    the new cast — this changes their blend recipe, so those voices' rendered segments
    re-render (settings_hash drift). ``recast`` on the ``hash`` strategy is a no-op.
    """
    if narrator_voice_id is None:
        narrator_voice_id = f"narrator_{slugify(default_preset)}"
        if not library.meta_path(narrator_voice_id).is_file():
            library.save(
                VoiceMeta(
                    voice_id=narrator_voice_id,
                    name="Narrator",
                    kind=VoiceKind.PRESET,
                    engine="kokoro",
                    preset_id=default_preset,
                    source="preset",
                    tags=["auto"],  # shared across books — no book tag
                )
            )
    elif not library.meta_path(narrator_voice_id).is_file():
        raise ServiceError(f"narrator voice {narrator_voice_id!r} not in the library")
    if thought_voice_id is not None and not library.meta_path(thought_voice_id).is_file():
        raise ServiceError(f"thought voice {thought_voice_id!r} not in the library")

    known_ids = {c.id for c in report.registry.characters}
    override_ids = set(overrides or {})
    assignments: dict[str, str] = {}

    if strategy == "smart":
        # The smart caster reserves overridden characters out of its distinct-voice budget
        # (F5 seam): overrides win below anyway, so casting them would waste voices.
        castable = [c for c in report.registry.characters if c.id not in override_ids]
        taken = _reserved_presets(library, overrides)
        cast = cast_book(castable, narrator_preset=default_preset, accent=accent, taken=taken)
        for char in castable:
            voice_id = f"{char.id}_auto"
            if recast or not library.meta_path(voice_id).is_file():
                library.save(
                    _auto_voice_meta(voice_id, char.canonical_name, cast[char.id], report.book_id)
                )
            assignments[char.id] = voice_id
    else:  # "hash": legacy per-character isolated blend (collision-blind)
        for char in report.registry.characters:
            voice_id = f"{char.id}_auto"
            if not library.meta_path(voice_id).is_file():
                recipe = auto_blend_recipe(char.canonical_name, char.gender, accent=accent)
                blend = [BlendComponent(preset_id=p, weight=w) for p, w in recipe]
                library.save(
                    VoiceMeta(
                        voice_id=voice_id,
                        name=char.canonical_name,
                        kind=VoiceKind.BLEND,
                        engine="kokoro",
                        blend=blend,
                        source="auto_blend",
                        tags=["auto", report.book_id],  # the UI maps book_id -> title
                    )
                )
            assignments[char.id] = voice_id

    for char_id, voice_id in (overrides or {}).items():
        if char_id not in known_ids:
            raise ServiceError(f"assignment override: unknown character {char_id!r}")
        if not library.meta_path(voice_id).is_file():
            raise ServiceError(f"assignment override: voice {voice_id!r} not in the library")
        assignments[char_id] = voice_id

    return VoiceAssignment(
        book_id=report.book_id,
        narrator_voice_id=narrator_voice_id,
        assignments=assignments,
        thought_voice_id=thought_voice_id,
        stage=stage,
    )


class CastPreview(NamedTuple):
    """A smart-cast PROPOSAL — nothing is written. ``would_create`` are ``{char_id}_auto``
    voices that don't exist yet; ``would_recast`` already exist and a ``recast`` apply would
    OVERWRITE them (re-rendering their segments). The UI shows both so applying is never a
    silent no-op on an already-drafted book."""

    assignment: VoiceAssignment
    would_create: list[str]
    would_recast: list[str]


def suggest_assignment(
    report: AttributionReport,
    library: VoiceLibrary,
    *,
    default_preset: str,
    narrator_voice_id: str | None = None,
    thought_voice_id: str | None = None,
    accent: str = "a",
    stage: AssignmentStage = AssignmentStage.DRAFT,
    overrides: dict[str, str] | None = None,
) -> CastPreview:
    """Compute the smart cast as a preview WITHOUT saving or creating any voice.

    Same shape as a ``strategy="smart"`` :func:`draft_assignment`, but read-only: it returns
    the proposed ``VoiceAssignment`` plus which auto voices it would create vs overwrite, so a
    UI can show the cost before the user commits via the draft (apply) endpoint.
    """
    if narrator_voice_id is None:
        narrator_voice_id = f"narrator_{slugify(default_preset)}"
    elif not library.meta_path(narrator_voice_id).is_file():
        raise ServiceError(f"narrator voice {narrator_voice_id!r} not in the library")
    if thought_voice_id is not None and not library.meta_path(thought_voice_id).is_file():
        raise ServiceError(f"thought voice {thought_voice_id!r} not in the library")

    known_ids = {c.id for c in report.registry.characters}
    override_ids = set(overrides or {})
    castable = [c for c in report.registry.characters if c.id not in override_ids]
    taken = _reserved_presets(library, overrides)
    cast_book(  # validates pool budget
        castable, narrator_preset=default_preset, accent=accent, taken=taken
    )

    assignments: dict[str, str] = {}
    would_create: list[str] = []
    would_recast: list[str] = []
    for char in castable:
        voice_id = f"{char.id}_auto"
        (would_recast if library.meta_path(voice_id).is_file() else would_create).append(voice_id)
        assignments[char.id] = voice_id
    for char_id, voice_id in (overrides or {}).items():
        if char_id not in known_ids:
            raise ServiceError(f"assignment override: unknown character {char_id!r}")
        if not library.meta_path(voice_id).is_file():
            raise ServiceError(f"assignment override: voice {voice_id!r} not in the library")
        assignments[char_id] = voice_id

    proposed = VoiceAssignment(
        book_id=report.book_id,
        narrator_voice_id=narrator_voice_id,
        assignments=assignments,
        thought_voice_id=thought_voice_id,
        stage=stage,
    )
    return CastPreview(proposed, sorted(would_create), sorted(would_recast))


def assignment_path(output_dir: Path, book_id: str) -> Path:
    return Path(output_dir) / book_id / ASSIGNMENT_NAME


def save_assignment(assignment: VoiceAssignment, output_dir: Path) -> Path:
    return atomic_write_text(
        assignment_path(output_dir, assignment.book_id), assignment.model_dump_json(indent=2)
    )


def load_assignment(output_dir: Path, book_id: str) -> VoiceAssignment:
    path = assignment_path(output_dir, book_id)
    if not path.is_file():
        raise ServiceError(f"no assignment at {path}; run `seiyuu assign {book_id}` first")
    return VoiceAssignment.model_validate_json(path.read_text(encoding="utf-8"))
