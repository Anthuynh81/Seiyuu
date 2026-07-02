"""Assignment service: character→voice drafting and the durable assignments.json write.

Extracted from cli.py so the API and job handlers share one implementation. Drafting is
deterministic: each character gets an auto-blend voice keyed by its character id, so
re-running reproduces the same draft voices and therefore the same segment cache keys.
"""

from pathlib import Path

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
    slugify,
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
) -> VoiceAssignment:
    """Build a VoiceAssignment, creating any missing draft voices in ``library``.

    Narrator is an explicit existing voice or an auto preset for ``default_preset``;
    ``overrides`` maps character ids to explicit voice ids (e.g. cloud voices for the
    final stage) and is validated against the report and the library.
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
                )
            )
    elif not library.meta_path(narrator_voice_id).is_file():
        raise ServiceError(f"narrator voice {narrator_voice_id!r} not in the library")
    if thought_voice_id is not None and not library.meta_path(thought_voice_id).is_file():
        raise ServiceError(f"thought voice {thought_voice_id!r} not in the library")

    assignments: dict[str, str] = {}
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
                )
            )
        assignments[char.id] = voice_id

    known_ids = {c.id for c in report.registry.characters}
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
