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


def _reserved_presets(
    library: VoiceLibrary,
    overrides: dict[str, str] | None,
    *,
    narrator_voice_id: str | None = None,
    thought_voice_id: str | None = None,
) -> set[str]:
    """Presets consumed by the override voices (and the narrator / thought voice), to reserve
    out of the smart caster's distinct-voice budget (``cast_book(taken=...)``) so an auto voice
    can never be cast onto a preset an override already uses — otherwise two characters render
    identical audio. The narrator/thought voice is reserved the same way: ``cast_book`` only
    knows ``default_preset``, so a hand-picked (or kept) narrator on another preset would
    otherwise be free for a character to share.

    Only Kokoro preset/blend voices contribute presets; cloned/cloud voices have none.
    Missing voices are skipped here (the validation paths report them with a clean
    ServiceError, or the auto narrator is not created yet in a read-only preview)."""
    taken: set[str] = set()
    voice_ids = [*(overrides or {}).values(), narrator_voice_id, thought_voice_id]
    for voice_id in voice_ids:
        if voice_id is None or not library.meta_path(voice_id).is_file():
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


class KeptCast(NamedTuple):
    """The draft arguments after folding a saved assignment in (:func:`merge_existing_cast`)."""

    narrator_voice_id: str | None
    thought_voice_id: str | None
    overrides: dict[str, str]


def merge_existing_cast(
    existing: VoiceAssignment | None,
    report: AttributionReport,
    library: VoiceLibrary,
    *,
    narrator_voice_id: str | None,
    thought_voice_id: str | None,
    overrides: dict[str, str] | None,
) -> KeptCast:
    """Fold the SAVED assignment's manual picks into a draft request so re-drafting never
    silently discards a hand-picked voice. Read-only (only checks voice existence).

    - narrator / thought voice: the saved one is kept when the request leaves it ``None``.
    - characters: every saved MANUAL pick (``voice != f"{char_id}_auto"``) is added to the
      overrides. Request overrides always win over a saved pick.
    - Auto picks are NOT folded: the caster recomputes them, so hash/smart/``recast``
      semantics are unchanged for auto voices.
    - A saved pick is dropped (no error) when its character is no longer in
      ``report.registry`` (re-attribution / merge) or its voice was deleted from the library —
      the caster then drafts that character like any other.

    ``existing=None`` (no assignments.json yet) returns the request unchanged.
    """
    merged: dict[str, str] = dict(overrides or {})
    if existing is None:
        return KeptCast(narrator_voice_id, thought_voice_id, merged)

    def _exists(voice_id: str) -> bool:
        return library.meta_path(voice_id).is_file()

    if narrator_voice_id is None and _exists(existing.narrator_voice_id):
        narrator_voice_id = existing.narrator_voice_id
    if (
        thought_voice_id is None
        and existing.thought_voice_id is not None
        and _exists(existing.thought_voice_id)
    ):
        thought_voice_id = existing.thought_voice_id

    known_ids = {c.id for c in report.registry.characters}
    for char_id, voice_id in existing.assignments.items():
        if char_id in merged or char_id not in known_ids:
            continue
        if voice_id == f"{char_id}_auto" or not _exists(voice_id):
            continue
        merged[char_id] = voice_id
    return KeptCast(narrator_voice_id, thought_voice_id, merged)


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
    trait_hints: dict[str, set[str]] | None = None,
) -> VoiceAssignment:
    """Build a VoiceAssignment FROM SCRATCH, creating any missing draft voices in ``library``.

    Narrator is an explicit existing voice or an auto preset for ``default_preset``;
    ``overrides`` maps character ids to explicit voice ids (e.g. cloud voices for the
    final stage) and is validated against the report and the library.

    This function never reads the saved assignments.json: anything not passed in is reset
    (narrator -> auto preset, thought voice -> None, every non-overridden character ->
    ``{char_id}_auto``). Callers that must preserve the user's manual picks (the API draft/
    suggest routes, ``seiyuu assign`` without ``--reset``, ``convert --multivoice``) fold the
    saved assignment into the arguments first via :func:`merge_existing_cast`.

    ``strategy`` selects the auto-caster: ``"hash"`` (legacy) draws each character's blend
    from its own ``name|gender`` hash in isolation; ``"smart"`` runs the book-level
    :func:`cast_book` so no two characters share a voice (and characters covered by
    ``overrides`` are reserved out of its distinct-voice budget). Both are skip-if-exists on
    ``{char_id}_auto``; ``recast=True`` (smart only) OVERWRITES existing auto voices to apply
    the new cast — this changes their blend recipe, so those voices' rendered segments
    re-render (settings_hash drift). ``recast`` on the ``hash`` strategy is a no-op.

    ``trait_hints`` is the OPTIONAL F4 Layer-2 LLM signal (``{char_id -> trait tags}``) forwarded
    to ``cast_book`` on the ``smart`` path only; it biases the tie-breaker without ever affecting
    distinctness or determinism, and is ignored by the ``hash`` strategy.
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
        taken = _reserved_presets(
            library,
            overrides,
            narrator_voice_id=narrator_voice_id,
            thought_voice_id=thought_voice_id,
        )
        cast = cast_book(
            castable,
            narrator_preset=default_preset,
            accent=accent,
            taken=taken,
            trait_hints=trait_hints,
        )
        for char in castable:
            voice_id = f"{char.id}_auto"
            if recast or not library.meta_path(voice_id).is_file():
                library.save(
                    _auto_voice_meta(voice_id, char.canonical_name, cast[char.id], report.book_id)
                )
            assignments[char.id] = voice_id
    else:  # "hash": legacy per-character isolated blend (collision-blind)
        for char in report.registry.characters:
            # Skip characters the overrides cover (F5 inheritance or an explicit pick): the
            # override loop below sets their assignment, so creating a throwaway {id}_auto voice
            # here would just orphan it. Mirrors the smart path's ``castable`` filter.
            if char.id in override_ids:
                continue
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
    trait_hints: dict[str, set[str]] | None = None,
) -> CastPreview:
    """Compute the smart cast as a preview WITHOUT saving or creating any voice.

    Same shape as a ``strategy="smart"`` :func:`draft_assignment`, but read-only: it returns
    the proposed ``VoiceAssignment`` plus which auto voices it would create vs overwrite, so a
    UI can show the cost before the user commits via the draft (apply) endpoint. ``trait_hints``
    is the same OPTIONAL F4 Layer-2 signal forwarded to ``cast_book`` (tie-breaker only).
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
    taken = _reserved_presets(
        library, overrides, narrator_voice_id=narrator_voice_id, thought_voice_id=thought_voice_id
    )
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


def load_existing_assignment(output_dir: Path, book_id: str) -> VoiceAssignment | None:
    """The saved assignment, or None when there is none yet (nothing to keep). A corrupt
    file raises (pydantic ``ValidationError`` / ``OSError`` / ``ValueError`` for bad UTF-8) —
    callers decide how to surface it, never silently treat it as absent."""
    if not assignment_path(output_dir, book_id).is_file():
        return None
    return load_assignment(output_dir, book_id)


def load_assignment(output_dir: Path, book_id: str) -> VoiceAssignment:
    path = assignment_path(output_dir, book_id)
    if not path.is_file():
        raise ServiceError(f"no assignment at {path}; run `seiyuu assign {book_id}` first")
    return VoiceAssignment.model_validate_json(path.read_text(encoding="utf-8"))
