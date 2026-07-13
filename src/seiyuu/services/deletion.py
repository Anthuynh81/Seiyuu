"""Book-deletion services (F3): paid-artifact detection and the purge summary.

Book deletion must never discard PAID cloud renders (ElevenLabs/Fish) on an automatic
path (CLAUDE.md's absolute rule), so the DELETE route and the ``seiyuu delete`` CLI gate
deletion behind an explicit second confirm whenever this module reports a paid segment.
Detection is deliberately engine-SDK-free: it reads only on-disk FACTS — the render
manifest's engine fields and the cache's frozen ``SegmentKey`` sidecars — and never
instantiates an engine, so it is cheap and can never trigger a paid call.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from seiyuu.api.schemas import PaidArtifacts
from seiyuu.render.models import RenderManifest
from seiyuu.settings import Settings

# Engines whose cached segments cost real money to reproduce. No clean engine-CLASS fact
# exists for paid-ness (the price is an instance field and ``cost_estimate`` needs the
# SDK), so this is the sanctioned small module constant — it mirrors CLAUDE.md's paid TTS
# lineup and stays a FACT so detection never imports an engine adapter.
PAID_ENGINES = frozenset({"elevenlabs", "fish"})


def _load_manifest(path: Path) -> RenderManifest | None:
    """The render manifest at ``path``, or None when there is no readable one. A torn or
    placeholder manifest is treated as 'no signal', not an error."""
    if not path.is_file():
        return None
    try:
        return RenderManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fallback_manifests(odir: Path) -> list[RenderManifest]:
    """One manifest per MODE for the fallback paid scan: both mode archives, plus a
    pre-feature manifest.json standing in for its mode when that archive is missing.
    Filenames are local literals mirroring ``render.pipeline.manifest_name_for_mode``
    (same rationale as ``repository.books``: this module must stay too light to ever
    import an engine); ``tests/test_book_deletion.py`` asserts they stay in sync. Never
    two copies of the same render — the active pointer duplicates an archive, and
    counting both would double-report its paid segments."""
    by_mode: dict[str, RenderManifest] = {}
    for mode in ("single", "multi"):
        manifest = _load_manifest(odir / f"manifest.{mode}.json")
        if manifest is not None:
            by_mode[mode] = manifest
    active = _load_manifest(odir / "manifest.json")
    if active is not None:
        by_mode.setdefault("single" if active.engine is not None else "multi", active)
    return list(by_mode.values())


def _manifest_paid_segments(manifest: RenderManifest) -> tuple[int, set[str], set[str]]:
    """Fallback paid signal when the cache dir is gone: count the manifest's rendered
    segments whose engine is paid, plus the paid engines and voice ids. A segment's engine is
    ``voices_used[voice_id].engine`` on a multi-voice render, else the single-voice
    ``manifest.engine``. Scene breaks (no wav) never count."""
    voice_engine = {vid: use.engine for vid, use in manifest.voices_used.items()}
    count = 0
    engines: set[str] = set()
    voice_ids: set[str] = set()
    for chapter in manifest.chapters:
        for seg in chapter.segments:
            if seg.wav is None:
                continue
            engine = voice_engine.get(seg.voice_id) if seg.voice_id else None
            engine = engine or manifest.engine
            if engine in PAID_ENGINES:
                count += 1
                engines.add(engine)
                if seg.voice_id is not None:
                    voice_ids.add(seg.voice_id)
    return count, engines, voice_ids


def detect_paid_artifacts(cfg: Settings, book_id: str) -> PaidArtifacts:
    """Report the PAID cloud work a deletion of ``book_id`` would discard. The cache's frozen
    ``SegmentKey`` sidecars are the AUTHORITATIVE signal (sign-off D8): ALWAYS scan
    ``output/{id}/cache/`` for sidecars naming a paid engine, no matter what the current
    manifest says. The cache is content-addressed and never pruned, so a paid render followed
    by a free re-render leaves the paid sidecars behind even after the manifest is overwritten
    to a free engine — the short-circuit that trusted the manifest would silently delete them
    off the gate. The manifest is used only ADDITIVELY/as a fallback: it proves paid work when
    the cache dir is absent (already pruned) and enriches the reported engine list.
    ``estimated_usd`` stays best-effort and is None — no per-segment cost is stored to derive
    it from."""
    odir = cfg.output_dir / book_id
    cache_dir = odir / "cache"
    paid_count = 0
    engines: set[str] = set()
    voice_ids: set[str] = set()
    if cache_dir.is_dir():
        for sidecar in sorted(cache_dir.glob("*.json")):
            if sidecar.name.endswith(".validation.json"):
                continue  # a validation verdict, not a SegmentKey
            try:
                key = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # torn sidecar: its wav is unmatchable, skip it
            if not isinstance(key, dict):
                continue
            engine = key.get("engine")
            voice_id = key.get("voice_id")
            if engine is None or voice_id is None:
                continue  # not a SegmentKey sidecar (e.g. a future .words.json)
            if engine in PAID_ENGINES:
                paid_count += 1
                engines.add(engine)
                voice_ids.add(voice_id)

    # Fallback ONLY when the cache scan found no paid segment: the content-addressed cache is
    # authoritative and is never pruned, so a paid render leaves its paid sidecars behind even
    # after a free re-render overwrites the manifest — those MUST gate. But if the cache dir is
    # gone entirely the manifests are the last proof of paid work, so count paid segments
    # across the per-mode manifests (both modes are about to be discarded) so the gate still
    # fires instead of silently discarding paid audio.
    if paid_count == 0:
        for manifest in _fallback_manifests(odir):
            count, manifest_engines, manifest_voice_ids = _manifest_paid_segments(manifest)
            paid_count += count
            engines |= manifest_engines
            voice_ids |= manifest_voice_ids

    return PaidArtifacts(
        paid_segment_count=paid_count,
        engines=sorted(engines),
        paid_voice_ids=sorted(voice_ids),
        estimated_usd=None,
    )


class PurgeSummary(BaseModel):
    """Pre-deletion preview: which on-disk roots exist and what paid work is at stake. Used
    by the CLI to print the plan before the destructive step."""

    book_id: str
    output_exists: bool
    books_exists: bool
    paid: PaidArtifacts


def compute_purge_manifest(cfg: Settings, book_id: str) -> PurgeSummary:
    """A read-only summary of what deleting ``book_id`` would touch (both roots + paid
    artifacts), computed without removing anything."""
    return PurgeSummary(
        book_id=book_id,
        output_exists=(cfg.output_dir / book_id).is_dir(),
        books_exists=(cfg.books_dir / book_id).is_dir(),
        paid=detect_paid_artifacts(cfg, book_id),
    )
