"""Render-mode activation: per-mode manifest archives + the manifest.json active pointer.

A completed render both archives its manifest (manifest.single.json / manifest.multi.json)
and promotes it to manifest.json — the single file Listen, assembly, mastering, and
GET /render read. Switching modes is therefore a pure atomic copy of the chosen archive
over the active pointer: no synthesis, no cache touch, instant fallback between a finished
single-voice render and a finished multivoice one. Pre-feature books (only a bare
manifest.json) migrate lazily: that file counts as its own mode's render until an archive
exists, and is preserved as the archive before anything overwrites it.
"""

from pathlib import Path

from pydantic import BaseModel

from seiyuu.render.models import RenderManifest
from seiyuu.render.pipeline import (
    MANIFEST_NAME,
    RENDER_MODES,
    manifest_mode,
    manifest_name_for_mode,
    preserve_unarchived_manifest,
)
from seiyuu.repository import Job, JobKind, JobState, JobStore, atomic_write_text
from seiyuu.services.common import ServiceError

# Job kinds that read or rewrite output/{id}/manifest.json mid-run; moving the active
# pointer under any of them would swap the audio truth they are streaming from.
_MANIFEST_JOB_KINDS = (JobKind.RENDER, JobKind.ASSEMBLE, JobKind.MASTER)


class RenderModeUnavailable(ServiceError):
    """The requested mode has no archived render to activate."""


class RenderModeConflict(ServiceError):
    """A live render/assemble/master job owns the manifest right now."""

    def __init__(self, message: str, job: Job) -> None:
        super().__init__(message)
        self.job = job


class RenderModeSwitch(BaseModel):
    """What the switch activated — enough for the API/CLI to echo."""

    book_id: str
    mode: str  # 'single' | 'multi' — now the active mode
    chapters: int  # chapter count of the activated manifest
    changed: bool  # False when this mode was already active (no-op switch)


def _read_manifest(path: Path) -> RenderManifest | None:
    """The manifest at ``path``, or None when absent/unreadable (read-only derivations
    treat a torn file as no signal; the write paths fail loudly instead)."""
    if not path.is_file():
        return None
    try:
        return RenderManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def available_render_modes(book_output_dir: Path, *, active_mode: str | None = None) -> list[str]:
    """Modes with an activatable render, in canonical (single, multi) order: every mode
    archive on disk, plus the active manifest.json's own mode (a pre-feature book has no
    archive yet — its active render still counts). Pass ``active_mode`` when the caller
    already parsed manifest.json, to skip re-reading it."""
    book_output_dir = Path(book_output_dir)
    modes = {m for m in RENDER_MODES if (book_output_dir / manifest_name_for_mode(m)).is_file()}
    if active_mode is None:
        active = _read_manifest(book_output_dir / MANIFEST_NAME)
        active_mode = manifest_mode(active) if active is not None else None
    if active_mode is not None:
        modes.add(active_mode)
    return [m for m in RENDER_MODES if m in modes]


def activate_render_mode(
    output_dir: Path, book_id: str, mode: str, *, store: JobStore
) -> RenderModeSwitch:
    """Point manifest.json at ``mode``'s archived render — an atomic copy, no synthesis.

    Refuses (``RenderModeConflict``) while a render/assemble/master job for the book is
    queued or running, and (``RenderModeUnavailable``) when the mode was never rendered.
    A pre-feature book whose only copy is a manifest.json of the requested mode is already
    active: the archive is materialized from it and the switch reports ``changed=False``.
    A pre-feature manifest.json of the OTHER mode is preserved as its archive before the
    pointer moves — the fallback this switch exists for must never be lost.
    """
    if mode not in RENDER_MODES:
        raise ServiceError(f"unknown render mode {mode!r} (expected one of {RENDER_MODES})")
    live = store.list_jobs(book_id=book_id, states=[JobState.QUEUED, JobState.RUNNING])
    conflict = next((j for j in live if j.kind in _MANIFEST_JOB_KINDS), None)
    if conflict is not None:
        raise RenderModeConflict(
            f"a {conflict.kind.value} job for {book_id!r} is {conflict.state.value}; wait for "
            f"it or cancel it before switching the active render mode",
            conflict,
        )

    book_output_dir = Path(output_dir) / book_id
    archive = book_output_dir / manifest_name_for_mode(mode)
    active_path = book_output_dir / MANIFEST_NAME
    active = _read_manifest(active_path)
    active_is_mode = active is not None and manifest_mode(active) == mode

    if not archive.is_file():
        if active_is_mode:
            # lazy migration: manifest.json IS this mode's only copy and it is already
            # active — materialize the archive so later switches see it; nothing else moves
            atomic_write_text(archive, active_path.read_text(encoding="utf-8"))
            return RenderModeSwitch(
                book_id=book_id, mode=mode, chapters=len(active.chapters), changed=False
            )
        label = "single-voice" if mode == "single" else "multivoice"
        raise RenderModeUnavailable(
            f"book {book_id!r} has no {label} render to activate; render that mode first"
        )

    try:
        raw = archive.read_text(encoding="utf-8")
        manifest = RenderManifest.model_validate_json(raw)
    except (OSError, ValueError) as exc:  # pydantic ValidationError is a ValueError
        raise ServiceError(
            f"book {book_id!r}: mode archive {archive.name} is unreadable: {exc}; "
            f"re-render that mode"
        ) from exc
    if manifest_mode(manifest) != mode:
        raise ServiceError(
            f"book {book_id!r}: mode archive {archive.name} holds a "
            f"{manifest_mode(manifest)} manifest — the archive is inconsistent; "
            f"re-render that mode"
        )
    preserve_unarchived_manifest(book_output_dir, exclude_mode=mode)
    atomic_write_text(active_path, raw)
    return RenderModeSwitch(
        book_id=book_id, mode=mode, chapters=len(manifest.chapters), changed=not active_is_mode
    )
