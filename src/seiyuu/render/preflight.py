"""Render pre-flight: cheap, offline checks that a render can actually START.

A signed cost token is single-use: once consumed, a refusal means a fresh estimate and a
fresh spend approval. The CLI used to consume the token and only THEN let ``render_book``
discover dead-ends that have nothing to do with money (a clone without consent, a
chapter-subset merge into a manifest with a different voice identity, a missing
ElevenLabs key, the GPU held by another process). These checks run BEFORE consumption,
mirroring the API's order (enqueue dry-run + ``_preflight_renderability`` -> consume at
job start).

Best-effort UX, never the enforcement: every check here either re-runs inside
``render_book`` / ``render_book_multivoice`` or fails the same way there, so skipping the
pre-flight can never let a render do something it otherwise couldn't. Everything raises
``RenderError`` naming the book (and voice/chapter) that failed. No synthesis, no model
load, no network.
"""

from pathlib import Path
from typing import Any

from seiyuu.attribute.models import AttributionReport
from seiyuu.engines import SynthesisError, TTSEngine, voices_dir_kwargs
from seiyuu.ingest.models import NormalizedBook
from seiyuu.render import pipeline as _pipeline
from seiyuu.render.pipeline import (
    CostEstimate,
    RenderError,
    _check_multivoice_merge_identity,
    _check_single_merge_identity,
    _effective_single_args,
    _manifest_merge_base,
)
from seiyuu.voices import VoiceAssignment, VoiceKind, VoiceLibrary, VoiceLibraryError, VoiceMeta
from seiyuu.voices.assignment import resolve_voice

_PROBE_TEXT = "pre-flight"


def is_cloud_preset(meta: VoiceMeta) -> bool:
    """True when ``render_voice_args`` hands the engine a real cloud id (``preset_id``)
    rather than the library voice_id — mirrors its PRESET branch exactly."""
    return meta.kind is VoiceKind.PRESET and bool(meta.preset_id)


def preflight_single(
    book: NormalizedBook,
    engine: TTSEngine,
    voice_id: str,
    book_output_dir: Path,
    est: CostEstimate,
    *,
    settings: dict[str, Any] | None = None,
    seed: int | None = None,
    chapters: tuple[int, ...] = (),
    library: VoiceLibrary | None = None,
) -> None:
    """Everything ``render_book`` would refuse before its first synthesis, checked up front.

    Always: chapter range, library-voice load + consent, and the chapter-subset merge
    identity. When ``est`` bills money: the paid engine can authenticate, and a library
    ElevenLabs voice the single-voice path cannot address is refused. ``render_book``
    resolves a saved voice through ``render_voice_args``: a PRESET cloud voice (``voice
    add-cloud``) goes to the engine as its ``preset_id`` — the raw cloud id, which works —
    but a CLONED/BLEND ElevenLabs voice goes as the library ``voice_id``, which the cloud
    has never heard of (only multivoice resolves a clone to its cloud handle).
    """
    book_id = book.book_meta.book_id
    wanted = set(chapters)
    unknown = wanted - set(range(1, len(book.chapters) + 1))
    if unknown:
        raise RenderError(
            f"{book_id}: no such chapter(s) {sorted(unknown)} (book has {len(book.chapters)})"
        )
    try:
        _engine_voice, effective_settings, effective_seed, meta = _effective_single_args(
            library, voice_id, settings or {}, seed
        )
        if meta is not None:
            library.verify_consent(meta)  # type: ignore[union-attr]  # meta implies library
    except VoiceLibraryError as exc:
        raise RenderError(f"{book_id}: voice {voice_id!r}: {exc}") from exc
    merge_base = _manifest_merge_base(
        Path(book_output_dir),
        book_id=book_id,
        wanted=wanted,
        total_chapters=len(book.chapters),
        multivoice=False,
    )
    if merge_base is not None:
        _check_single_merge_identity(
            merge_base,
            book_id=book_id,
            engine=engine,
            voice_id=voice_id,
            settings=effective_settings,
            seed=effective_seed,
        )
    if est.total_usd <= 0:
        return
    if meta is not None and meta.engine == "elevenlabs" and not is_cloud_preset(meta):
        raise RenderError(
            f"{book_id}: voice {voice_id!r} is a {meta.kind.value} ElevenLabs library voice; "
            "the single-voice path only addresses cloud presets (a raw cloud voice id or a "
            "`voice add-cloud` voice) — use --multivoice for cloned ElevenLabs voices"
        )
    _check_paid_engine_ready(engine, book_id=book_id)
    _check_gpu_free([engine], est, book_id=book_id)


def preflight_multivoice(
    report: AttributionReport,
    book: NormalizedBook,
    library: VoiceLibrary,
    assignment: VoiceAssignment,
    book_output_dir: Path,
    est: CostEstimate,
    *,
    chapters: tuple[int, ...] = (),
) -> None:
    """Everything ``render_book_multivoice`` would refuse before its first synthesis.

    Always: the chapter-subset merge identity, and load + consent for every voice an
    in-scope segment resolves to (exactly the voices the render's ``meta_for`` gates).
    When ``est`` bills money: every paid engine in scope can authenticate, and the GPU is
    not held by another process if local engines have uncached work too.
    """
    book_id = book.book_meta.book_id
    wanted = set(chapters)
    merge_base = _manifest_merge_base(
        Path(book_output_dir),
        book_id=book_id,
        wanted=wanted,
        total_chapters=len(book.chapters),
        multivoice=True,
    )
    if merge_base is not None:
        _check_multivoice_merge_identity(merge_base, assignment, book_id=book_id)

    in_book = set(range(1, len(book.chapters) + 1))
    voice_ids: set[str] = set()
    for att_chapter in report.chapters:
        if att_chapter.index not in in_book or (wanted and att_chapter.index not in wanted):
            continue
        voice_ids.update(resolve_voice(seg, assignment) for seg in att_chapter.segments)
    engine_ids: set[str] = set()
    for voice_id in sorted(voice_ids):
        try:
            meta = library.load(voice_id)
            library.verify_consent(meta)
        except VoiceLibraryError as exc:
            raise RenderError(f"{book_id}: voice {voice_id!r}: {exc}") from exc
        engine_ids.add(meta.engine)
    if est.total_usd <= 0:
        return
    # constructed exactly as the estimate/render construct them (the pipeline's own
    # get_engine seam) — adapters load no weights at construction
    engines = [
        _pipeline.get_engine(eid, **voices_dir_kwargs(eid, library.voices_dir))
        for eid in sorted(engine_ids)
    ]
    for engine in engines:
        _check_paid_engine_ready(engine, book_id=book_id)
    _check_gpu_free(engines, est, book_id=book_id)


def _check_paid_engine_ready(engine: TTSEngine, *, book_id: str) -> None:
    """A paid engine that cannot authenticate would fail at its first segment — after the
    token is burned and possibly after hours of local synthesis. Cloud adapters expose
    their authenticated SDK client as ``client`` (raising SynthesisError without a key);
    building it makes no request."""
    if engine.cost_estimate(_PROBE_TEXT) <= 0 or not hasattr(type(engine), "client"):
        return
    try:
        engine.client  # noqa: B018 — property access IS the check
    except SynthesisError as exc:
        raise RenderError(
            f"{book_id}: engine {engine.engine_id} cannot authenticate: {exc}; the paid "
            "segments in this render cannot synthesize"
        ) from exc


def _check_gpu_free(engines: list[TTSEngine], est: CostEstimate, *, book_id: str) -> None:
    """Refuse up front when another seiyuu PROCESS holds the card and a local engine has
    uncached work: the render would hit GpuBusyError at its first local segment.

    Probe-and-release of the same OS lock the GPU manager claims (data_dir/gpu.lock). A
    manager in THIS process that already has a resident model owns the card, so the probe
    is skipped (re-locking our own lock would falsely report busy on Windows). Racy by
    nature — a competitor can still claim the card after the probe; the manager's own
    claim stays the enforcement."""
    if est.free_segments <= 0 or not any(e.uses_gpu for e in engines):
        return
    from seiyuu.gpu import get_gpu_manager
    from seiyuu.repository.lock import FileLockHandle
    from seiyuu.settings import get_settings

    if get_gpu_manager().resident is not None:
        return
    probe = FileLockHandle(get_settings().data_dir / "gpu.lock")
    if not probe.try_acquire():
        raise RenderError(
            f"{book_id}: another seiyuu process holds the GPU (lock: {probe.path}). Is the "
            "API server running, or another seiyuu command? Stop it or wait for its job "
            "to finish, then retry."
        )
    probe.release()
