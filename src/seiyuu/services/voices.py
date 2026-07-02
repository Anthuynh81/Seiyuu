"""Voice deletion with a referential guard.

Deleting a voice removes its whole directory — including reference.wav, the consent-
attested source of truth for a clone — so the operation must be loud and safe: refuse
while any book's assignments.json still references the voice (a later render would fail
with a missing voice, or worse, silently re-draft a different one)."""

import shutil
from pathlib import Path

from pydantic import BaseModel

from seiyuu.services.common import ServiceError
from seiyuu.voices import ASSIGNMENT_NAME, VoiceAssignment, VoiceLibrary, VoiceLibraryError


class VoiceReference(BaseModel):
    book_id: str
    role: str  # 'narrator' | 'thought' | 'character:<id>'


def voice_references(voice_id: str, output_dir: Path) -> list[VoiceReference]:
    """Every place a voice is referenced across all books' assignments.json."""
    output_dir = Path(output_dir)
    refs: list[VoiceReference] = []
    if not output_dir.is_dir():
        return refs
    for assign_path in sorted(output_dir.glob(f"*/{ASSIGNMENT_NAME}")):
        try:
            assignment = VoiceAssignment.model_validate_json(
                assign_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            # FAIL CLOSED: an unreadable assignment may still reference the voice, and
            # the caller is about to rmtree a consent-attested reference.wav
            raise ServiceError(
                f"cannot verify voice references: {assign_path} is unreadable ({exc}); "
                f"fix or remove that file first"
            ) from exc
        book_id = assign_path.parent.name
        if assignment.narrator_voice_id == voice_id:
            refs.append(VoiceReference(book_id=book_id, role="narrator"))
        if assignment.thought_voice_id == voice_id:
            refs.append(VoiceReference(book_id=book_id, role="thought"))
        for char_id, vid in assignment.assignments.items():
            if vid == voice_id:
                refs.append(VoiceReference(book_id=book_id, role=f"character:{char_id}"))
    return refs


def delete_voice(voice_id: str, *, library: VoiceLibrary, output_dir: Path) -> Path:
    """Delete a voice directory outright, refusing while assignments still reference it."""
    voice_dir = library.dir_for(voice_id)
    # rmtree containment: in M6b voice_id arrives from an HTTP client — a traversal like
    # '../books' must never resolve outside the library before we delete anything
    if voice_dir.resolve().parent != library.voices_dir.resolve():
        raise ServiceError(f"invalid voice id {voice_id!r}")
    # library.load enforces EXACT id equality with meta.json — NTFS resolves 'V1' (or
    # 'v1.'/'v1 ') to the real voices/v1 dir, but the reference scan below is string-
    # based, so a case-variant id would sail past the guard and delete a referenced voice
    try:
        library.load(voice_id)
    except VoiceLibraryError as exc:
        raise ServiceError(str(exc)) from exc
    refs = voice_references(voice_id, output_dir)
    if refs:
        where = ", ".join(f"{r.book_id} ({r.role})" for r in refs[:5])
        more = f" and {len(refs) - 5} more" if len(refs) > 5 else ""
        raise ServiceError(
            f"voice {voice_id!r} is still assigned in: {where}{more}; "
            f"reassign those books first (`seiyuu assign`)"
        )
    shutil.rmtree(voice_dir)
    return voice_dir
