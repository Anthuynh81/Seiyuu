"""Single-voice render: normalized JSON → cached canonical segment WAVs + manifest.

Only speakable blocks (paragraph, heading) become synthesis segments; scene
breaks pass through to the manifest as pause markers. Every synthesis call
goes through the segment cache.
"""

import hashlib
import json
import queue
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf

if TYPE_CHECKING:
    from seiyuu.api.concurrency import BorrowBroker
    from seiyuu.normalize.lexicon import CompiledLexicon

from seiyuu.attribute.models import AttributionReport, EmotionVerdict
from seiyuu.engines import TTSEngine, get_engine, voices_dir_kwargs
from seiyuu.gpu import get_gpu_manager
from seiyuu.ingest.models import BlockType, NormalizedBook
from seiyuu.normalize import normalize_text, profile_for
from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest, VoiceUse
from seiyuu.repository import atomic_write_text
from seiyuu.validate import (
    SegmentWords,
    ValidationResult,
    Validator,
    WordTiming,
    resample_to_whisper,
)
from seiyuu.voices import (
    VoiceAssignment,
    VoiceLibrary,
    VoiceLibraryError,
    ensure_cloud_voice,
    map_emotion,
    render_voice_args,
    resolve_voice,
)
from seiyuu.voices.models import VoiceMeta

MANIFEST_NAME = "manifest.json"

# Per-mode manifest archives: a completed render writes its manifest to the mode archive
# AND promotes the same content to MANIFEST_NAME (the ACTIVE pointer every consumer —
# Listen, assemble, master, GET /render — reads). Both modes' renders coexist; switching
# is an atomic copy of an archive over the pointer, never a re-render.
RENDER_MODES = ("single", "multi")


def manifest_name_for_mode(mode: str) -> str:
    """Archive filename for a render mode: manifest.single.json / manifest.multi.json."""
    if mode not in RENDER_MODES:
        raise ValueError(f"unknown render mode {mode!r} (expected one of {RENDER_MODES})")
    return f"manifest.{mode}.json"


def manifest_mode(manifest: RenderManifest) -> str:
    """'single' iff the manifest carries the single-voice engine identity, else 'multi'."""
    return "single" if manifest.engine is not None else "multi"


def preserve_unarchived_manifest(book_output_dir: Path, *, exclude_mode: str) -> None:
    """Lazy migration for pre-feature books: if manifest.json parses to a mode OTHER than
    ``exclude_mode`` and that mode's archive is missing, copy it (byte-for-byte, atomic)
    to that archive BEFORE the caller overwrites manifest.json — activating one mode must
    never discard the other mode's only completed render. An absent/unreadable
    manifest.json preserves nothing: there is no provable render to keep."""
    active_path = Path(book_output_dir) / MANIFEST_NAME
    if not active_path.is_file():
        return
    try:
        raw = active_path.read_text(encoding="utf-8")
        existing_mode = manifest_mode(RenderManifest.model_validate_json(raw))
    except (OSError, ValueError):
        return
    if existing_mode == exclude_mode:
        return
    archive = Path(book_output_dir) / manifest_name_for_mode(existing_mode)
    if not archive.is_file():
        atomic_write_text(archive, raw)


def _write_mode_manifests(book_output_dir: Path, manifest: RenderManifest, mode: str) -> Path:
    """Persist a completed render: mode archive first, then promote the SAME content to
    manifest.json (rendering a mode activates it — you rendered it to hear it). Two atomic
    writes, archive before active, so a crash between them leaves the archive (the
    recoverable truth) rather than an active pointer with no archive behind it."""
    preserve_unarchived_manifest(book_output_dir, exclude_mode=mode)
    payload = manifest.model_dump_json(indent=2)
    atomic_write_text(Path(book_output_dir) / manifest_name_for_mode(mode), payload)
    active_path = Path(book_output_dir) / MANIFEST_NAME
    atomic_write_text(active_path, payload)
    return active_path


class RenderError(Exception):
    """Loud render failure naming book/chapter/block."""


@dataclass
class CostEstimate:
    total_usd: float
    paid_segments: int  # uncached segments that will cost money
    cached_segments: int  # already rendered, free to reuse
    free_segments: int  # uncached but local (free)
    # Identity of the paid work (hash over paid-engine SegmentKey hashes, cached or not).
    # Cache growth never changes it; any text/voice/settings/seed change does. The cost
    # gate binds its signed quotes to this.
    fingerprint: str = ""


def _paid_fingerprint(paid_key_hashes: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(paid_key_hashes)).encode("utf-8")).hexdigest()


def _emotion_settings(
    settings: dict[str, Any],
    engine_id: str,
    emotion: "EmotionVerdict | None",
    apply_emotion: bool,
) -> dict[str, Any]:
    """Merge the F2 emotion override into ``settings`` (identically at render AND estimate).

    Returns ``settings`` UNCHANGED (the same dict) when ``apply_emotion`` is off, the emotion
    is None/neutral, or the engine has no emotion knob — so the FROZEN SegmentKey is byte-
    identical to a no-emotion render (cache-stable). When on and non-neutral, the override
    folds into ``settings_hash``'s value. This single helper is the parity guarantee: render
    and the cost estimate must call it with the same arguments or the gate authorizes a
    different bill than render runs up.
    """
    if not apply_emotion or emotion is None:
        return settings
    override = map_emotion(engine_id, emotion)
    return {**settings, **override} if override else settings


def _emotions_by_segment(chapter) -> list["EmotionVerdict | None"]:
    """The chapter's per-segment emotions, normalized to one entry per segment (F2).

    Empty (v3/v4 report) or a length mismatch degrades to all-None, so render/estimate treat a
    pre-emotion book exactly as today.
    """
    emotions = getattr(chapter, "segment_emotions", None) or []
    if len(emotions) == len(chapter.segments):
        return list(emotions)
    return [None] * len(chapter.segments)


def _effective_single_args(
    library: "VoiceLibrary | None",
    voice_id: str,
    settings: dict[str, Any],
    seed: int | None,
) -> tuple[str, dict[str, Any], int | None, VoiceMeta | None]:
    """Resolve the (engine_voice, settings, seed, meta) the single-voice SegmentKey and
    synthesis actually use for a possibly-saved library voice.

    For a bare preset id (no library directory) the caller's values are returned verbatim
    and meta is None. For a SAVED library voice they come from ``render_voice_args(meta)``
    + ``meta.seed`` — a Kokoro preset addresses the engine by preset_id, a blend folds its
    recipe into settings. Shared by ``render_book`` and ``estimate_render_cost_single`` so
    the FROZEN SegmentKey they build can never drift. Does NOT verify consent — the
    read-only estimate must not; ``render_book`` gates consent separately.
    """
    if library is not None and (
        library.meta_path(voice_id).is_file() or library.reference_path(voice_id).is_file()
    ):
        # load() also refuses a reference.wav-only dir (no meta = never attested)
        meta = library.load(voice_id)
        engine_voice, effective_settings = render_voice_args(meta)
        return engine_voice, effective_settings, meta.seed, meta
    return voice_id, settings, seed, None


def _gate_paid(
    engine: TTSEngine,
    text: str,
    allow_paid: bool,
    spent_usd: float,
    max_paid_usd: float | None,
    *,
    book_id,
    block_id,
    voice,
) -> float:
    """Refuse a paid synthesis unless explicitly authorized AND within the approved budget;
    returns the updated running paid total. No automatic code path may bill, and an approved
    render may never bill past its approval — if the segment cache changed under us mid-run
    (eviction, concurrent job), fail loudly instead of overspending."""
    cost = engine.cost_estimate(text)
    if cost <= 0:
        return spent_usd
    if not allow_paid:
        raise RenderError(
            f"refusing paid synthesis (~${cost:.4f}) book={book_id} block={block_id} "
            f"voice={voice} engine={engine.engine_id} without cost confirmation; confirm the "
            f"estimate first (CLI: `seiyuu estimate-cost`, then render with --confirm-cost)"
        )
    spent_usd += cost
    # a cent of slack: the estimate's total is rounded, per-segment costs are not
    if max_paid_usd is not None and spent_usd > max_paid_usd + 0.01:
        raise RenderError(
            f"paid synthesis (${spent_usd:.4f} so far) would exceed the approved budget "
            f"(${max_paid_usd:.4f}) at book={book_id} block={block_id} voice={voice}; the "
            f"segment cache changed since the estimate — re-run estimate-cost and re-approve"
        )
    return spent_usd


@dataclass
class RenderResult:
    manifest: RenderManifest
    manifest_path: Path
    synthesized: int
    cache_hits: int
    validation_failures: int = 0

    @property
    def total_audio_seconds(self) -> float:
        return sum(s.duration_seconds for c in self.manifest.chapters for s in c.segments)


def _validate_capturing_words(
    validator: Validator, wav_path: Path, text: str
) -> tuple[ValidationResult, list[WordTiming] | None]:
    """One transcription pass that yields BOTH the verdict and the word timings when the
    validator supports it (the real ``Validator``); otherwise the verdict alone. This is the
    F2 piggyback — a `requires_validation` engine already transcribes every attempt, so its
    word alignment costs no extra whisper pass. Scripted test validators exposing only
    ``validate`` transparently fall back to no words."""
    if hasattr(validator, "validate_with_words"):
        return validator.validate_with_words(wav_path, text)
    return validator.validate(wav_path, text), None


@dataclass
class _PendingSegment:
    """One `requires_validation` segment in flight through the synth→whisper overlap.

    Its manifest row is a placeholder (None) at ``rows[slot]`` until the verdict lands.
    ``where``/``flag_where`` carry the owning loop's own error/progress phrasing so both
    render loops share the overlap machinery verbatim. A ``revalidate_wav`` job is a
    cache hit with a missing verdict: score that wav once, never retry, cache nothing
    but the verdict."""

    rows: list  # the chapter's RenderedSegment slots (None until finalized)
    slot: int
    block_id: str
    block_type: BlockType
    voice_id: str
    seed: int | None
    key: SegmentKey
    text: str
    engine: TTSEngine
    engine_voice: str
    settings: dict[str, Any]
    where: str  # "book=… chapter=… block=…" in the owning loop's error shape
    flag_where: str  # the validation-failure say() fragment in the owning loop's shape
    revalidate_wav: Path | None = None
    attempts: int = 0
    audio: Any = None  # the attempt currently being scored
    best_audio: Any = None
    best_result: ValidationResult | None = None
    best_words: list[WordTiming] | None = None


@dataclass
class _PlannedSynthesis:
    """One uncached multivoice segment awaiting its ENGINE GROUP's synthesis pass.

    The chapter plan partitions uncached segments by engine so each engine synthesizes
    all its segments under one residency (mixing engines segment-to-segment thrashes
    the single GPU with catastrophic reloads). The manifest row lands back at ``slot``
    in reading order — the FROZEN per-segment cache key makes synthesis order
    irrelevant to caching."""

    slot: int
    block_id: str
    block_type: BlockType
    voice_id: str
    meta: VoiceMeta
    engine: TTSEngine
    text: str
    settings: dict[str, Any]
    key: SegmentKey
    engine_voice: str
    where: str
    flag_where: str


class _ValidationWorker:
    """The single background whisper thread: CPU-scores segment N while the GPU
    synthesizes N+1. FIFO; verdicts are consumed on the render thread via collect()."""

    _STOP = object()

    def __init__(self, validator: Validator, cache_dir: Path) -> None:
        self._validator = validator
        self._cache_dir = Path(cache_dir)
        self._todo: queue.Queue[Any] = queue.Queue()
        self._done: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None

    def submit(self, job: _PendingSegment) -> None:
        if self._thread is None:  # lazy: a render with no validated engine never starts it
            self._thread = threading.Thread(target=self._run, name="seiyuu-validation", daemon=True)
            self._thread.start()
        self._todo.put(job)

    def collect(self, *, wait: bool) -> tuple | None:
        """The next finished (job, result, words, exc), or None when none is ready — after
        ~0.2s when waiting, so the caller can re-check cancel/lend between waits."""
        try:
            return self._done.get(block=wait, timeout=0.2 if wait else None)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Idempotent shutdown. Unscored queue entries are discarded (an aborting render
        will never read their verdicts); the in-progress transcription finishes first."""
        if self._thread is None:
            return
        try:
            while True:
                self._todo.get_nowait()
        except queue.Empty:
            pass
        self._todo.put(self._STOP)
        self._thread.join(timeout=60.0)
        self._thread = None

    def _run(self) -> None:
        while True:
            job = self._todo.get()
            if job is self._STOP:
                return
            try:
                result, words = self._score(job)
                self._done.put((job, result, words, None))
            except BaseException as exc:
                self._done.put((job, None, None, exc))

    def _score(self, job: _PendingSegment) -> tuple[ValidationResult, list[WordTiming] | None]:
        validator = self._validator
        if job.revalidate_wav is not None:
            return _validate_capturing_words(validator, job.revalidate_wav, job.text)
        if getattr(validator, "accepts_arrays", False):
            # real Validator: hand whisper the waveform, skip the tmp-wav round trip
            source = resample_to_whisper(job.audio.samples, job.audio.sample_rate)
            return _validate_capturing_words(validator, source, job.text)
        tmp = self._cache_dir / f"_validate.{job.key.key_hash}.tmp.wav"
        try:
            job.audio.save(tmp)
            return _validate_capturing_words(validator, tmp, job.text)
        finally:
            tmp.unlink(missing_ok=True)


class _ValidationOverlap:
    """Render-thread coordinator for overlapped validation (shared by both render loops).

    Whisper scoring is the ONLY thing on the worker thread; synthesis (GPU, under
    acquire), retry dispatch with the next seed, cache writes, and manifest-row
    finalize all stay on the render thread, so cache and manifest access remain
    single-threaded. Retries keep the best-scoring attempt exactly like the old serial
    helper; a persistent failure is flagged for review, never dropped. At most
    ``_MAX_IN_FLIGHT`` segments are buffered (bounds audio memory), and drain() runs at
    every chapter boundary so a chapter's rows are complete before it is emitted."""

    _MAX_IN_FLIGHT = 2

    def __init__(
        self,
        *,
        validator: Validator | None,
        cache: SegmentCache,
        gpu: Any,
        book_output_dir: Path,
        say: Callable[[str], None],
        check: Callable[[], None],
        lend: Callable[[], None],
        max_retries: int,
    ) -> None:
        self._validator = validator
        self._cache = cache
        self._gpu = gpu
        self._book_output_dir = Path(book_output_dir)
        self._say, self._check, self._lend = say, check, lend
        self._max_retries = max_retries
        self._worker = (
            _ValidationWorker(validator, cache.cache_dir) if validator is not None else None
        )
        self._pending: dict[str, _PendingSegment] = {}
        self.synthesized = 0
        self.validation_failures = 0

    def handles(self, engine: TTSEngine) -> bool:
        return self._validator is not None and engine.requires_validation

    def wait_for_key(self, key: SegmentKey) -> None:
        """Park until an identical in-flight segment finalizes — its cached wav then
        serves the new occurrence as a cache hit, preserving the serial loop's
        no-duplicate-synthesis behavior for repeated identical blocks."""
        while key.key_hash in self._pending:
            self._service_step()

    def submit_fresh(self, job: _PendingSegment) -> None:
        """First synthesis attempt now (on this thread), verdict off-thread."""
        while len(self._pending) >= self._MAX_IN_FLIGHT:
            self._service_step()
        self._pending[job.key.key_hash] = job
        self._synth_attempt(job)
        self.poll()

    def submit_revalidate(self, job: _PendingSegment) -> None:
        while len(self._pending) >= self._MAX_IN_FLIGHT:
            self._service_step()
        self._pending[job.key.key_hash] = job
        self._worker.submit(job)
        self.poll()

    def poll(self) -> None:
        """Consume already-finished verdicts without blocking (called between segments)."""
        while self._worker is not None:
            item = self._worker.collect(wait=False)
            if item is None:
                return
            self._handle(item)

    def drain(self) -> None:
        """Block — servicing retries — until every in-flight segment has its row."""
        while self._pending:
            self._service_step()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    def _service_step(self) -> None:
        # The render thread is idle while whisper works: honor cancel and lend the
        # resident engine to a waiting audition, then handle at most one verdict.
        self._check()
        self._lend()
        item = self._worker.collect(wait=True)
        if item is not None:
            self._handle(item)

    def _handle(self, item: tuple) -> None:
        job, result, words, exc = item
        if exc is not None:
            del self._pending[job.key.key_hash]
            if isinstance(exc, RenderError):
                raise exc
            raise RenderError(f"synthesis failed: {job.where}: {exc}") from exc
        if job.best_result is None or result.score > job.best_result.score:
            job.best_audio, job.best_result, job.best_words = job.audio, result, words
        if job.revalidate_wav is None and not result.ok and job.attempts <= self._max_retries:
            self._synth_attempt(job)  # retry with the next seed; the engine is still resident
            return
        self._finalize(job)

    def _synth_attempt(self, job: _PendingSegment) -> None:
        i = job.attempts
        job.attempts = i + 1
        seed = job.seed if (i == 0 or job.seed is None) else job.seed + i
        synth = {**job.settings, **({"seed": seed} if seed is not None else {})}
        try:
            ctx = (
                self._gpu.acquire(job.engine, job.engine.engine_id)
                if job.engine.uses_gpu
                else nullcontext()
            )
            with ctx:
                job.audio = job.engine.synthesize(job.text, job.engine_voice, synth)
        except RenderError:
            raise
        except Exception as exc:
            raise RenderError(f"synthesis failed: {job.where}: {exc}") from exc
        self._worker.submit(job)

    def _finalize(self, job: _PendingSegment) -> None:
        result = job.best_result
        if job.revalidate_wav is not None:
            wav_path = job.revalidate_wav
            duration = sf.info(str(wav_path)).duration
            self._cache.put_validation(job.key, result)
        else:
            wav_path = self._cache.put(job.key, job.best_audio)
            self._cache.put_validation(job.key, result)
            duration = job.best_audio.duration_seconds
            if job.best_words is not None:  # F2 piggyback: validated engines cache words inline
                self._cache.put_words(
                    job.key, SegmentWords(words=job.best_words, audio_duration=duration)
                )
            self.synthesized += 1
        if not result.ok:
            self.validation_failures += 1
            self._say(
                f"  ! validation failed (score {result.score}) {job.flag_where} "
                f"after {job.attempts} attempt(s) — flagged for review"
            )
        job.rows[job.slot] = RenderedSegment(
            block_id=job.block_id,
            type=job.block_type,
            wav=wav_path.relative_to(self._book_output_dir).as_posix(),
            duration_seconds=round(duration, 3),
            voice_id=job.voice_id,
            seed=job.seed,
            settings_hash=job.key.settings_hash,
            validation=result,
            synth_attempts=job.attempts,
        )
        del self._pending[job.key.key_hash]


def _manifest_merge_base(
    book_output_dir: Path, *, book_id: str, wanted: set[int], total_chapters: int, multivoice: bool
) -> RenderManifest | None:
    """The existing SAME-MODE manifest a chapter-SUBSET render must merge into (None →
    plain overwrite).

    A subset is a non-empty ``wanted`` that does not cover all ``total_chapters`` (1-based).
    Full-book renders, and subset renders of a book with no manifest yet, keep the historical
    overwrite-wholesale behavior. A subset render over a prior manifest must NOT clobber it:
    chapters outside the subset would vanish from the render summary, Listen, and assembly
    even though their WAVs still sit in cache. The merge base comes from the mode's ARCHIVE
    (manifest.single.json / manifest.multi.json); a pre-feature book with only manifest.json
    contributes it iff it parses to the SAME mode (lazy migration). A different-mode
    manifest.json is simply not a merge base — cross-mode subset renders start fresh in
    their own archive, so the mode-mismatch refusal below guards only a hand-edited archive
    whose content contradicts its name (unreachable via normal flows).
    """
    if not wanted or wanted == set(range(1, total_chapters + 1)):
        return None
    mode = "multi" if multivoice else "single"
    manifest_path = Path(book_output_dir) / manifest_name_for_mode(mode)
    from_archive = manifest_path.is_file()
    if not from_archive:
        manifest_path = Path(book_output_dir) / MANIFEST_NAME
        if not manifest_path.is_file():
            return None
    try:
        existing = RenderManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pydantic ValidationError is a ValueError
        raise RenderError(
            f"{book_id}: cannot merge a chapter-subset render into unreadable manifest "
            f"{manifest_path}: {exc}; delete it or render the full book"
        ) from exc
    if manifest_mode(existing) != mode:
        if not from_archive:
            # pre-feature manifest.json of the OTHER mode: not a merge base for this mode;
            # it stays in place as that mode's archive fallback (preserved at write time)
            return None
        raise RenderError(
            f"{book_id}: mode archive {manifest_path.name} holds a {manifest_mode(existing)} "
            f"manifest, not {mode} — the archive is inconsistent; delete it or render the "
            f"full book"
        )
    return existing


def _check_single_merge_identity(
    existing: RenderManifest,
    *,
    book_id: str,
    engine: TTSEngine,
    voice_id: str,
    settings: dict[str, Any],
    seed: int | None,
) -> None:
    """Refuse a single-voice subset merge whose voice identity differs from the manifest's.

    The single-voice manifest stores ONE engine/model/voice/settings/seed at top level for
    every chapter; merging chapters rendered under a different identity would make those
    fields lie about the carried-over half. Settings are JSON-round-tripped because the
    existing side was parsed from disk (they are JSON-serializable by SegmentKey contract).
    """
    ours = {
        "engine": engine.engine_id,
        "engine_model_version": engine.model_version,
        "voice_id": voice_id,
        "settings": json.loads(json.dumps(settings)),
        "seed": seed,
    }
    theirs = {
        "engine": existing.engine,
        "engine_model_version": existing.engine_model_version,
        "voice_id": existing.voice_id,
        "settings": existing.settings,
        "seed": existing.seed,
    }
    if ours != theirs:
        fields = ", ".join(sorted(k for k in ours if ours[k] != theirs[k]))
        raise RenderError(
            f"{book_id}: refusing single-voice chapter-subset render: {fields} differ(s) from "
            f"the existing manifest (existing voice={existing.voice_id!r} "
            f"engine={existing.engine!r}, new voice={voice_id!r} engine={engine.engine_id!r}); "
            f"render the full book to change the voice"
        )


# The assignment-snapshot fields that determine which voice renders which segment — exactly
# what resolve_voice reads. stage/created_at churn from a re-saved assignment must not block
# a resume.
_ASSIGNMENT_IDENTITY = ("narrator_voice_id", "assignments", "thought_voice_id")


def _check_multivoice_merge_identity(
    existing: RenderManifest, assignment: VoiceAssignment, *, book_id: str
) -> None:
    """Refuse a multivoice subset merge whose voice map differs from the manifest snapshot's."""
    snapshot = existing.assignment or {}
    ours = assignment.model_dump(mode="json")
    fields = [k for k in _ASSIGNMENT_IDENTITY if snapshot.get(k) != ours.get(k)]
    if fields:
        raise RenderError(
            f"{book_id}: refusing multivoice chapter-subset render: the voice assignment "
            f"({', '.join(fields)}) differs from the existing manifest's assignment snapshot; "
            f"chapters outside the subset were rendered with the OLD voices — render the "
            f"full book to apply the new assignment"
        )


def _merge_manifests(
    existing: RenderManifest, new: RenderManifest, *, total_chapters: int
) -> RenderManifest:
    """Merge a chapter-subset render into the previous whole-book manifest.

    Newly rendered chapters replace their old entries by index; untouched chapters carry
    over — but only up to ``total_chapters``: a re-upload of the same file with different
    split settings keeps the book_id yet can SHRINK the chapter set, leaving ghost entries
    in the old manifest that must not survive the merge. Aggregates are recomputed over the
    SURVIVING chapter set: validation_failures from the per-segment verdicts, voices_used
    as a union restricted to voices a surviving segment actually used (new provenance wins
    on overlap). Identity compatibility was enforced before synthesis started.
    """
    by_index = {ch.index: ch for ch in existing.chapters if ch.index <= total_chapters}
    by_index.update((ch.index, ch) for ch in new.chapters)
    chapters = [by_index[i] for i in sorted(by_index)]
    surviving_voices = {seg.voice_id for ch in chapters for seg in ch.segments if seg.voice_id}
    merged_voices = {**existing.voices_used, **new.voices_used}
    return new.model_copy(
        update={
            "chapters": chapters,
            "voices_used": {v: use for v, use in merged_voices.items() if v in surviving_voices},
            "validation_failures": sum(
                1
                for chapter in chapters
                for seg in chapter.segments
                if seg.validation is not None and not seg.validation.ok
            ),
        }
    )


def render_book(
    book: NormalizedBook,
    engine: TTSEngine,
    voice_id: str,
    book_output_dir: Path,
    *,
    settings: dict[str, Any] | None = None,
    seed: int | None = None,
    chapters: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    gpu=None,
    library: VoiceLibrary | None = None,
    validator: Validator | None = None,
    validation_max_retries: int = 2,
    allow_paid: bool = False,
    max_paid_usd: float | None = None,
    check_cancel: Callable[[], None] | None = None,
    broker: "BorrowBroker | None" = None,
    lexicon: "CompiledLexicon | None" = None,
    force: bool = False,
    release_gpu: bool = True,
) -> RenderResult:
    """Render a book (or a 1-based subset of `chapters`) with one voice.

    ``release_gpu=False`` (server only) leaves the model lazily resident at the end —
    the manager's design: a competitor acquire evicts it, lifespan teardown frees it —
    so back-to-back renders and follow-up auditions re-acquire warm instead of paying
    a full reload. The True default frees the card for the CLI, where an out-of-process
    Ollama attribution stage may need the VRAM next.

    A chapter-subset render MERGES into the book's existing SAME-MODE manifest archive
    (same voice identity required — refused up front otherwise) instead of clobbering it;
    a full-book render, or a subset with no prior single-voice manifest, overwrites
    wholesale as before. The completed manifest lands in manifest.single.json AND becomes
    the active manifest.json.

    When ``library`` is given and ``voice_id`` refers to a library voice (a directory with
    meta.json or reference.wav), consent is verified before any synthesis — a cloned voice
    must never render ungated just because it came through the single-voice path. Bare
    engine preset ids (no library directory) have nothing to verify. GPU engines are
    acquired through the resource manager (one heavy model resident at a time) and freed
    at the end. ``check_cancel`` (when given) is called between chapters and between
    blocks and may raise to abort cooperatively; synthesized segments are already cached
    and no manifest is written, so a re-run resumes from the cache.

    ``broker`` (F1, server only) lets a waiting audition borrow this render's resident
    engine between synthesis units: the engine is published once and offered at each
    ``check`` yield point (after cancel), and closed before the GPU is freed. Default None
    keeps the CLI/tests unchanged.
    """
    settings = settings or {}
    book_output_dir = Path(book_output_dir)
    cache = SegmentCache(book_output_dir / "cache")
    say = progress or (lambda _msg: None)
    check = check_cancel or (lambda: None)
    gpu = gpu or get_gpu_manager()

    def lend() -> None:
        # Offer this render's own resident instance to a waiting audition; parks here until
        # the audition finishes one segment (never while synthesizing, always after cancel).
        if broker is not None and engine.uses_gpu:
            broker.serve(engine.engine_id, engine)

    if broker is not None and engine.uses_gpu:
        broker.publish(engine.engine_id, engine)

    # The engine voice arg / settings / seed the adapter actually synthesizes with. For a
    # bare preset id (no library dir) these are the caller's values verbatim. For a SAVED
    # library voice they come from render_voice_args(meta): a Kokoro preset addresses the
    # engine by preset_id, a blend folds its recipe into settings — passing voice_id verbatim
    # (as the bare path does) crashes those. The FROZEN SegmentKey still keys on voice_id.
    try:
        engine_voice, effective_settings, effective_seed, meta = _effective_single_args(
            library, voice_id, settings, seed
        )
        if meta is not None:
            library.verify_consent(meta)
    except VoiceLibraryError as exc:
        raise RenderError(str(exc)) from exc

    wanted = set(chapters)
    unknown = wanted - set(range(1, len(book.chapters) + 1))
    if unknown:
        raise RenderError(
            f"{book.book_meta.book_id}: no such chapter(s) {sorted(unknown)} "
            f"(book has {len(book.chapters)})"
        )
    merge_base = _manifest_merge_base(
        book_output_dir,
        book_id=book.book_meta.book_id,
        wanted=wanted,
        total_chapters=len(book.chapters),
        multivoice=False,
    )
    if merge_base is not None:
        _check_single_merge_identity(
            merge_base,
            book_id=book.book_meta.book_id,
            engine=engine,
            voice_id=voice_id,
            settings=effective_settings,
            seed=effective_seed,
        )

    rendered_chapters: list[RenderedChapter] = []
    profile = profile_for(engine.engine_id)
    synthesized = cache_hits = validation_failures = 0
    paid_spent = 0.0
    overlap = _ValidationOverlap(
        validator=validator, cache=cache, gpu=gpu, book_output_dir=book_output_dir,
        say=say, check=check, lend=lend, max_retries=validation_max_retries,
    )  # fmt: skip
    try:
        for ci, chapter in enumerate(book.chapters, start=1):
            if wanted and ci not in wanted:
                continue
            check()
            lend()
            say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}")
            segments: list[RenderedSegment | None] = []
            for block in chapter.blocks:
                check()
                lend()
                overlap.poll()  # consume any finished verdicts between segments
                if block.type is BlockType.SCENE_BREAK:
                    segments.append(RenderedSegment(block_id=block.id, type=block.type))
                    continue
                text = normalize_text(block.text, profile=profile, lexicon=lexicon)
                key = SegmentKey.build(
                    engine=engine.engine_id,
                    engine_model_version=engine.model_version,
                    voice_id=voice_id,
                    settings=effective_settings,
                    seed=effective_seed,
                    normalized_text=text,
                )
                overlap.wait_for_key(key)  # an identical segment may still be in flight
                # force: a re-render bypasses the cache HIT and re-synthesizes, overwriting the
                # same key_hash. In-scope only — the chapter-skip above already excludes the rest.
                wav_path = None if force else cache.get(key)
                if wav_path is not None:
                    cache_hits += 1
                    validation = cache.get_validation(key)
                    if validation is None and overlap.handles(engine):
                        # A cache hit with no stored verdict (crash between the wav write and
                        # the verdict write, or a pre-M4 segment) must not ship unvalidated:
                        # re-score the cached wav off-thread; its row (counted and flagged
                        # exactly like a fresh render) lands when the verdict does.
                        segments.append(None)
                        overlap.submit_revalidate(
                            _PendingSegment(
                                rows=segments,
                                slot=len(segments) - 1,
                                block_id=block.id,
                                block_type=block.type,
                                voice_id=voice_id,
                                seed=effective_seed,
                                key=key,
                                text=text,
                                engine=engine,
                                engine_voice=engine_voice,
                                settings=effective_settings,
                                revalidate_wav=wav_path,
                                attempts=1,
                                where=(
                                    f"book={book.book_meta.book_id} chapter={ci} "
                                    f"({chapter.title!r}) block={block.id} "
                                    f"engine={engine.engine_id} voice={voice_id}"
                                ),
                                flag_where=f"block={block.id}",
                            )  # fmt: skip
                        )
                        continue
                    duration = sf.info(str(wav_path)).duration
                    if validation is not None and not validation.ok:
                        validation_failures += 1
                        say(
                            f"  ! validation failed (score {validation.score}) block={block.id} "
                            f"after 1 attempt(s) — flagged for review"
                        )
                    segments.append(
                        RenderedSegment(
                            block_id=block.id,
                            type=block.type,
                            wav=wav_path.relative_to(book_output_dir).as_posix(),
                            duration_seconds=round(duration, 3),
                            voice_id=voice_id,
                            seed=effective_seed,
                            settings_hash=key.settings_hash,
                            validation=validation,
                            synth_attempts=1,
                        )
                    )
                    continue
                paid_spent = _gate_paid(
                    engine, text, allow_paid, paid_spent, max_paid_usd,
                    book_id=book.book_meta.book_id, block_id=block.id, voice=voice_id,
                )  # fmt: skip
                if overlap.handles(engine):
                    # validated engine: synthesize now, score on the worker — the GPU moves
                    # on to the next segment while whisper (CPU) scores this one
                    segments.append(None)
                    overlap.submit_fresh(
                        _PendingSegment(
                            rows=segments,
                            slot=len(segments) - 1,
                            block_id=block.id,
                            block_type=block.type,
                            voice_id=voice_id,
                            seed=effective_seed,
                            key=key,
                            text=text,
                            engine=engine,
                            engine_voice=engine_voice,
                            settings=effective_settings,
                            where=(
                                f"book={book.book_meta.book_id} chapter={ci} "
                                f"({chapter.title!r}) block={block.id} "
                                f"engine={engine.engine_id} voice={voice_id}"
                            ),
                            flag_where=f"block={block.id}",
                        )  # fmt: skip
                    )
                    continue
                # deterministic engine (or no validator): synthesize once, no verdict
                synth = {
                    **effective_settings,
                    **({"seed": effective_seed} if effective_seed is not None else {}),
                }
                try:
                    ctx = (
                        gpu.acquire(engine, engine.engine_id) if engine.uses_gpu else nullcontext()
                    )
                    with ctx:
                        audio = engine.synthesize(text, engine_voice, synth)
                except RenderError:
                    raise
                except Exception as exc:
                    raise RenderError(
                        f"synthesis failed: book={book.book_meta.book_id} "
                        f"chapter={ci} ({chapter.title!r}) block={block.id} "
                        f"engine={engine.engine_id} voice={voice_id}: {exc}"
                    ) from exc
                wav_path = cache.put(key, audio)
                synthesized += 1
                segments.append(
                    RenderedSegment(
                        block_id=block.id,
                        type=block.type,
                        wav=wav_path.relative_to(book_output_dir).as_posix(),
                        duration_seconds=round(audio.duration_seconds, 3),
                        voice_id=voice_id,
                        seed=effective_seed,
                        settings_hash=key.settings_hash,
                        validation=None,
                        synth_attempts=1,
                    )
                )
            overlap.drain()  # every deferred row for this chapter, before it is emitted
            rendered_chapters.append(
                RenderedChapter(index=ci, title=chapter.title, segments=segments)
            )
    finally:
        # Stop lending BEFORE the engine is unloaded: an in-flight request gets an
        # immediate None (soft retry), never an about-to-be-freed instance.
        if broker is not None:
            broker.close()
        overlap.stop()
        if engine.uses_gpu and release_gpu:
            # free only what this render could have loaded: a cloud-only render must not
            # evict another consumer's resident model from the shared manager
            gpu.free_all()

    synthesized += overlap.synthesized
    validation_failures += overlap.validation_failures
    manifest = RenderManifest(
        book_id=book.book_meta.book_id,
        book_title=book.book_meta.title,
        engine=engine.engine_id,
        engine_model_version=engine.model_version,
        voice_id=voice_id,
        settings=effective_settings,
        seed=effective_seed,
        chapters=rendered_chapters,
        validation_failures=validation_failures,
    )
    if merge_base is not None:
        manifest = _merge_manifests(merge_base, manifest, total_chapters=len(book.chapters))
    manifest_path = _write_mode_manifests(book_output_dir, manifest, "single")
    return RenderResult(
        manifest=manifest,
        manifest_path=manifest_path,
        synthesized=synthesized,
        cache_hits=cache_hits,
        validation_failures=validation_failures,
    )


def render_book_multivoice(
    report: AttributionReport,
    book: NormalizedBook,
    library: VoiceLibrary,
    assignment: VoiceAssignment,
    book_output_dir: Path,
    *,
    chapters: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    gpu=None,
    validator: Validator | None = None,
    validation_max_retries: int = 2,
    allow_paid: bool = False,
    max_paid_usd: float | None = None,
    cloud_max_slots: int = 10,
    check_cancel: Callable[[], None] | None = None,
    broker: "BorrowBroker | None" = None,
    lexicon: "CompiledLexicon | None" = None,
    apply_emotion: bool = False,
    force: bool = False,
    engine_provider: Callable[[str], TTSEngine] | None = None,
    release_gpu: bool = True,
) -> RenderResult:
    """Multi-voice render: attribution segments + per-character voices → cached WAVs + manifest.

    ``engine_provider`` (server only) supplies engine instances by id — the API handler
    passes the process-lifetime registry's ``get`` so a model warmed by a warmup job or
    audition is REUSED (the GPU manager compares consumers by identity; a fresh instance
    always evicts the warm one and cold-loads multi-GB weights). Default None constructs
    engines directly, keeping the CLI and tests unchanged. ``release_gpu=False`` (server
    only, same rationale as ``render_book``) leaves the last engine lazily resident
    instead of unloading at the end.

    Reads the attribution report (segments + resolved speaker ids), the normalized book
    (scene-break pause markers + reading order), the voice library, and the assignment. Each
    chapter renders in two passes: a plan pass resolves every segment's voice/engine/key in
    reading order and emits cache-hit rows, then the UNCACHED segments synthesize grouped
    by engine — one residency per engine per chapter instead of a catastrophic model reload
    at every voice alternation. Rows land back in reading order; the FROZEN per-segment
    cache key makes synthesis order irrelevant to caching. Paid segments are still gated
    per segment before synthesis (the cumulative budget cap is order-insensitive).

    A chapter-subset render MERGES into the book's existing SAME-MODE manifest archive
    (same voice assignment required — refused up front otherwise) instead of clobbering it;
    a full-book render, or a subset with no prior multivoice manifest, overwrites wholesale
    as before. The completed manifest lands in manifest.multi.json AND becomes the active
    manifest.json.

    ``check_cancel`` (when given) is called between chapters and between segments and may
    raise to abort cooperatively; synthesized segments are already cached and no manifest
    is written, so a re-run resumes from the cache. The GPU is freed either way.

    ``broker`` (F1, server only) lends the render's OWN resident engine to a waiting
    audition between segments. The lendable engine varies segment to segment (this is a
    multi-engine loop), so it is (re)published right after each engine is chosen and the
    currently-resident one is offered at every ``check`` yield point (after cancel).
    """
    book_output_dir = Path(book_output_dir)
    cache = SegmentCache(book_output_dir / "cache")
    say = progress or (lambda _msg: None)
    check = check_cancel or (lambda: None)
    gpu = gpu or get_gpu_manager()

    wanted = set(chapters)
    attributed = {ch.index: ch for ch in report.chapters}
    merge_base = _manifest_merge_base(
        book_output_dir,
        book_id=book.book_meta.book_id,
        wanted=wanted,
        total_chapters=len(book.chapters),
        multivoice=True,
    )
    if merge_base is not None:
        _check_multivoice_merge_identity(merge_base, assignment, book_id=book.book_meta.book_id)
    engines: dict[str, TTSEngine] = {}
    metas: dict[str, VoiceMeta] = {}
    voices_used: dict[str, VoiceUse] = {}
    # The engine currently resident and free to lend between segments (F1). Updated after
    # each engine is chosen; None until the first GPU segment establishes residency.
    lent_id: str | None = None
    lent_engine: TTSEngine | None = None

    def lend() -> None:
        if broker is not None and lent_engine is not None:
            broker.serve(lent_id, lent_engine)

    def engine_for(engine_id: str) -> TTSEngine:
        if engine_id not in engines:
            if engine_provider is not None:
                engines[engine_id] = engine_provider(engine_id)
            else:
                extra = voices_dir_kwargs(engine_id, library.voices_dir)
                engines[engine_id] = get_engine(engine_id, **extra)
        return engines[engine_id]

    def meta_for(voice_id: str) -> VoiceMeta:
        if voice_id not in metas:
            meta = library.load(voice_id)
            try:
                # once per voice per render: cloned voices need consent bound to the
                # actual reference audio (hash), not just a flippable bool
                library.verify_consent(meta)
            except VoiceLibraryError as exc:
                raise RenderError(str(exc)) from exc
            metas[voice_id] = meta
        return metas[voice_id]

    rendered_chapters: list[RenderedChapter] = []
    synthesized = cache_hits = validation_failures = 0
    paid_spent = 0.0
    used_gpu = False
    overlap = _ValidationOverlap(
        validator=validator, cache=cache, gpu=gpu, book_output_dir=book_output_dir,
        say=say, check=check, lend=lend, max_retries=validation_max_retries,
    )  # fmt: skip
    try:
        for ci, chapter in enumerate(book.chapters, start=1):
            if (wanted and ci not in wanted) or ci not in attributed:
                continue
            check()
            lend()
            say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}")
            att_chapter = attributed[ci]
            emotions = _emotions_by_segment(att_chapter)  # F2: index-aligned to segments
            by_block: dict[str, list] = {}
            for seg, emotion in zip(att_chapter.segments, emotions, strict=True):
                by_block.setdefault(seg.block_id, []).append((seg, emotion))

            # PASS 1 — plan (reading order, no synthesis): resolve each segment's voice/
            # engine/key, emit cache-hit rows immediately, and partition the UNCACHED
            # segments by engine. Missing-verdict cache hits go to the whisper worker now,
            # so they score while pass 2 synthesizes.
            rendered: list[RenderedSegment | None] = []
            groups: dict[str, list[_PlannedSynthesis]] = {}  # engine_id → first-appearance order
            for block in chapter.blocks:
                if block.type is BlockType.SCENE_BREAK:
                    rendered.append(RenderedSegment(block_id=block.id, type=block.type))
                    continue
                for seg, emotion in by_block.get(block.id, []):
                    check()
                    lend()
                    overlap.poll()  # consume any finished verdicts between segments
                    voice_id = resolve_voice(seg, assignment)
                    meta = meta_for(voice_id)  # verifies consent on first sight
                    engine = engine_for(meta.engine)
                    text = normalize_text(
                        seg.text, profile=profile_for(meta.engine), lexicon=lexicon
                    )
                    engine_voice, settings = render_voice_args(meta)
                    # F2: fold the per-segment emotion override in BEFORE the FROZEN SegmentKey
                    # (and before synthesis), so it rides settings_hash. No-op when apply_emotion
                    # is off / neutral / Kokoro — key stays byte-identical to today.
                    settings = _emotion_settings(settings, meta.engine, emotion, apply_emotion)
                    key = SegmentKey.build(
                        engine=meta.engine,
                        engine_model_version=engine.model_version,
                        voice_id=voice_id,
                        settings=settings,
                        seed=meta.seed,
                        normalized_text=text,
                    )
                    voices_used.setdefault(
                        voice_id,
                        VoiceUse(
                            engine=meta.engine,
                            engine_model_version=engine.model_version,
                            kind=meta.kind.value,
                        ),
                    )
                    where = (
                        f"book={book.book_meta.book_id} chapter={ci} "
                        f"block={block.id} voice={voice_id} engine={meta.engine}"
                    )
                    flag_where = f"block={block.id} voice={voice_id}"
                    overlap.wait_for_key(key)  # an identical revalidation may be in flight
                    # force: a re-render bypasses the cache HIT and re-synthesizes, overwriting the
                    # same key_hash. In-scope only — the chapter-skip above excludes the rest.
                    wav_path = None if force else cache.get(key)
                    if wav_path is None:
                        rendered.append(None)
                        groups.setdefault(meta.engine, []).append(
                            _PlannedSynthesis(
                                slot=len(rendered) - 1,
                                block_id=block.id,
                                block_type=block.type,
                                voice_id=voice_id,
                                meta=meta,
                                engine=engine,
                                text=text,
                                settings=settings,
                                key=key,
                                engine_voice=engine_voice,
                                where=where,
                                flag_where=flag_where,
                            )  # fmt: skip
                        )
                        continue
                    cache_hits += 1
                    validation = cache.get_validation(key)
                    if validation is None and overlap.handles(engine):
                        # A cache hit with no stored verdict (crash between the wav write
                        # and the verdict write, or a pre-M4 segment) must not ship
                        # unvalidated: re-score the cached wav off-thread; its row (counted
                        # and flagged exactly like a fresh render) lands with the verdict.
                        rendered.append(None)
                        overlap.submit_revalidate(
                            _PendingSegment(
                                rows=rendered,
                                slot=len(rendered) - 1,
                                block_id=block.id,
                                block_type=block.type,
                                voice_id=voice_id,
                                seed=meta.seed,
                                key=key,
                                text=text,
                                engine=engine,
                                engine_voice=engine_voice,
                                settings=settings,
                                revalidate_wav=wav_path,
                                attempts=1,
                                where=where,
                                flag_where=flag_where,
                            )  # fmt: skip
                        )
                        continue
                    duration = sf.info(str(wav_path)).duration
                    if validation is not None and not validation.ok:
                        validation_failures += 1
                        say(
                            f"  ! validation failed (score {validation.score}) "
                            f"block={block.id} voice={voice_id} after 1 attempt(s) — "
                            f"flagged for review"
                        )
                    rendered.append(
                        RenderedSegment(
                            block_id=block.id,
                            type=block.type,
                            wav=wav_path.relative_to(book_output_dir).as_posix(),
                            duration_seconds=round(duration, 3),
                            voice_id=voice_id,
                            seed=meta.seed,
                            settings_hash=key.settings_hash,
                            validation=validation,
                            synth_attempts=1,
                        )
                    )

            # PASS 2 — synthesize per engine group under ONE residency: all of an engine's
            # segments run back-to-back, so the manager unloads/reloads at most once per
            # engine per chapter instead of at every voice alternation (SPEC's deferred
            # voice-grouped synthesis). Draining at each group boundary keeps validation
            # retries on the still-resident engine.
            for engine_id, planned in groups.items():
                for p in planned:
                    check()
                    lend()
                    overlap.poll()
                    if broker is not None and p.engine.uses_gpu:
                        # this group's engine is what will be resident; offer it at the
                        # next yield point (lend() reads these via closure)
                        broker.publish(engine_id, p.engine)
                        lent_id, lent_engine = engine_id, p.engine
                    overlap.wait_for_key(p.key)  # an identical twin may be in flight
                    wav_path = None if force else cache.get(p.key)
                    if wav_path is not None:
                        # rendered by an identical earlier segment this run: reuse it
                        cache_hits += 1
                        validation = cache.get_validation(p.key)
                        duration = sf.info(str(wav_path)).duration
                        if validation is not None and not validation.ok:
                            validation_failures += 1
                            say(
                                f"  ! validation failed (score {validation.score}) "
                                f"{p.flag_where} after 1 attempt(s) — flagged for review"
                            )
                        rendered[p.slot] = RenderedSegment(
                            block_id=p.block_id,
                            type=p.block_type,
                            wav=wav_path.relative_to(book_output_dir).as_posix(),
                            duration_seconds=round(duration, 3),
                            voice_id=p.voice_id,
                            seed=p.meta.seed,
                            settings_hash=p.key.settings_hash,
                            validation=validation,
                            synth_attempts=1,
                        )
                        continue
                    paid_spent = _gate_paid(
                        p.engine, p.text, allow_paid, paid_spent, max_paid_usd,
                        book_id=book.book_meta.book_id, block_id=p.block_id, voice=p.voice_id,
                    )  # fmt: skip
                    try:
                        synth_voice = p.engine_voice
                        if engine_id == "elevenlabs":  # resolve/create the cloud voice
                            synth_voice = ensure_cloud_voice(
                                p.meta, p.engine.client, library, max_slots=cloud_max_slots
                            )
                    except RenderError:
                        raise
                    except Exception as exc:
                        raise RenderError(f"synthesis failed: {p.where}: {exc}") from exc
                    used_gpu = used_gpu or p.engine.uses_gpu
                    if overlap.handles(p.engine):
                        # validated engine: synthesize now, score on the worker — the GPU
                        # moves on to the next segment while whisper (CPU) scores this one
                        overlap.submit_fresh(
                            _PendingSegment(
                                rows=rendered,
                                slot=p.slot,
                                block_id=p.block_id,
                                block_type=p.block_type,
                                voice_id=p.voice_id,
                                seed=p.meta.seed,
                                key=p.key,
                                text=p.text,
                                engine=p.engine,
                                engine_voice=synth_voice,
                                settings=p.settings,
                                where=p.where,
                                flag_where=p.flag_where,
                            )  # fmt: skip
                        )
                        continue
                    # deterministic or cloud engine: synthesize once, no verdict
                    synth = {
                        **p.settings,
                        **({"seed": p.meta.seed} if p.meta.seed is not None else {}),
                    }
                    try:
                        ctx = (
                            gpu.acquire(p.engine, p.engine.engine_id)
                            if p.engine.uses_gpu
                            else nullcontext()
                        )
                        with ctx:
                            audio = p.engine.synthesize(p.text, synth_voice, synth)
                    except RenderError:
                        raise
                    except Exception as exc:
                        raise RenderError(f"synthesis failed: {p.where}: {exc}") from exc
                    wav_path = cache.put(p.key, audio)
                    synthesized += 1
                    rendered[p.slot] = RenderedSegment(
                        block_id=p.block_id,
                        type=p.block_type,
                        wav=wav_path.relative_to(book_output_dir).as_posix(),
                        duration_seconds=round(audio.duration_seconds, 3),
                        voice_id=p.voice_id,
                        seed=p.meta.seed,
                        settings_hash=p.key.settings_hash,
                        validation=None,
                        synth_attempts=1,
                    )
                overlap.drain()  # same-engine retries finish while the engine is resident

            overlap.drain()  # any pass-1 revalidations left, before the chapter is emitted
            rendered_chapters.append(
                RenderedChapter(index=ci, title=chapter.title, segments=rendered)
            )
    finally:
        # Stop lending BEFORE the engine is unloaded: an in-flight request gets an
        # immediate None (soft retry), never an about-to-be-freed instance.
        if broker is not None:
            broker.close()
        overlap.stop()
        # free the GPU for the next stage/process — but only if this render acquired it;
        # a cloud-only render must not evict another consumer's resident model
        if used_gpu and release_gpu:
            gpu.free_all()

    synthesized += overlap.synthesized
    validation_failures += overlap.validation_failures
    manifest = RenderManifest(
        book_id=book.book_meta.book_id,
        book_title=book.book_meta.title,
        chapters=rendered_chapters,
        voices_used=voices_used,
        assignment=assignment.model_dump(mode="json"),
        validation_failures=validation_failures,
    )
    if merge_base is not None:
        manifest = _merge_manifests(merge_base, manifest, total_chapters=len(book.chapters))
    manifest_path = _write_mode_manifests(book_output_dir, manifest, "multi")
    return RenderResult(
        manifest=manifest,
        manifest_path=manifest_path,
        synthesized=synthesized,
        cache_hits=cache_hits,
        validation_failures=validation_failures,
    )


def estimate_render_cost(
    report: AttributionReport,
    book: NormalizedBook,
    library: VoiceLibrary,
    assignment: VoiceAssignment,
    book_output_dir: Path,
    *,
    chapters: tuple[int, ...] = (),
    lexicon: "CompiledLexicon | None" = None,
    apply_emotion: bool = False,
    force: bool = False,
) -> CostEstimate:
    """Pre-flight cost of a multi-voice render: USD over the segments that aren't already cached.

    Read-only and offline — builds the same FROZEN SegmentKey as render_book_multivoice to count
    cache hits exactly, and sums each uncached segment's engine.cost_estimate (no API key or
    synthesis needed). What this returns is what the cost gate will let render bill.
    """
    book_output_dir = Path(book_output_dir)
    cache = SegmentCache(book_output_dir / "cache")
    wanted = set(chapters)
    attributed = {ch.index: ch for ch in report.chapters}
    engines: dict[str, TTSEngine] = {}
    metas: dict[str, VoiceMeta] = {}

    def engine_for(engine_id: str) -> TTSEngine:
        if engine_id not in engines:
            extra = voices_dir_kwargs(engine_id, library.voices_dir)
            engines[engine_id] = get_engine(engine_id, **extra)
        return engines[engine_id]

    def meta_for(voice_id: str) -> VoiceMeta:
        if voice_id not in metas:
            metas[voice_id] = library.load(voice_id)
        return metas[voice_id]

    total = 0.0
    paid = cached = free = 0
    paid_hashes: list[str] = []
    for ci, chapter in enumerate(book.chapters, start=1):
        if (wanted and ci not in wanted) or ci not in attributed:
            continue
        att_chapter = attributed[ci]
        emotions = _emotions_by_segment(att_chapter)  # F2: index-aligned to segments
        by_block: dict[str, list] = {}
        for seg, emotion in zip(att_chapter.segments, emotions, strict=True):
            by_block.setdefault(seg.block_id, []).append((seg, emotion))
        for block in chapter.blocks:
            if block.type is BlockType.SCENE_BREAK:
                continue
            for seg, emotion in by_block.get(block.id, []):
                # resolved id, NOT meta.voice_id: must build the EXACT SegmentKey the render
                # loop will use, or the gate authorizes a different bill than render runs up
                voice_id = resolve_voice(seg, assignment)
                meta = meta_for(voice_id)
                engine = engine_for(meta.engine)
                text = normalize_text(seg.text, profile=profile_for(meta.engine), lexicon=lexicon)
                _, settings = render_voice_args(meta)
                # F2 parity: identical emotion merge as render_book_multivoice, so the gate
                # authorizes exactly the SegmentKeys render will bill.
                settings = _emotion_settings(settings, meta.engine, emotion, apply_emotion)
                key = SegmentKey.build(
                    engine=meta.engine,
                    engine_model_version=engine.model_version,
                    voice_id=voice_id,
                    settings=settings,
                    seed=meta.seed,
                    normalized_text=text,
                )
                cost = engine.cost_estimate(text)
                if cost > 0:
                    paid_hashes.append(key.key_hash)  # paid identity, cached or not
                # force: price forced-in-scope segments as work (a HIT no longer discounts them),
                # so the quote authorizes the paid re-synthesis. Fingerprint is unchanged above.
                if not force and cache.get(key) is not None:
                    cached += 1
                    continue
                if cost > 0:
                    total += cost
                    paid += 1
                else:
                    free += 1
    return CostEstimate(
        total_usd=round(total, 4),
        paid_segments=paid,
        cached_segments=cached,
        free_segments=free,
        fingerprint=_paid_fingerprint(paid_hashes),
    )


def estimate_render_cost_single(
    book: NormalizedBook,
    engine: TTSEngine,
    voice_id: str,
    book_output_dir: Path,
    *,
    settings: dict[str, Any] | None = None,
    seed: int | None = None,
    chapters: tuple[int, ...] = (),
    library: VoiceLibrary | None = None,
    lexicon: "CompiledLexicon | None" = None,
    force: bool = False,
) -> CostEstimate:
    """Pre-flight cost of a SINGLE-VOICE render (M6a) — the counterpart the M5 gate lacked,
    so a paid engine could be authorized with no estimate at all. Same loop and FROZEN
    SegmentKey as render_book, so cache hits are counted exactly; free engines total 0.

    When ``library`` is given and ``voice_id`` is a SAVED library voice, the key is built
    from the voice's stored settings + pinned seed (via the shared ``_effective_single_args``)
    exactly as ``render_book`` now does — otherwise the estimate would key on the caller's
    raw settings/seed and never match what render actually caches for saved voices."""
    settings = settings or {}
    cache = SegmentCache(Path(book_output_dir) / "cache")
    profile = profile_for(engine.engine_id)
    _engine_voice, settings, seed, _meta = _effective_single_args(library, voice_id, settings, seed)
    wanted = set(chapters)
    total = 0.0
    paid = cached = free = 0
    paid_hashes: list[str] = []
    for ci, chapter in enumerate(book.chapters, start=1):
        if wanted and ci not in wanted:
            continue
        for block in chapter.blocks:
            if block.type is BlockType.SCENE_BREAK:
                continue
            text = normalize_text(block.text, profile=profile, lexicon=lexicon)
            key = SegmentKey.build(
                engine=engine.engine_id,
                engine_model_version=engine.model_version,
                voice_id=voice_id,
                settings=settings,
                seed=seed,
                normalized_text=text,
            )
            cost = engine.cost_estimate(text)
            if cost > 0:
                paid_hashes.append(key.key_hash)
            # force: price forced-in-scope segments as work (see estimate_render_cost).
            if not force and cache.get(key) is not None:
                cached += 1
                continue
            if cost > 0:
                total += cost
                paid += 1
            else:
                free += 1
    return CostEstimate(
        total_usd=round(total, 4),
        paid_segments=paid,
        cached_segments=cached,
        free_segments=free,
        fingerprint=_paid_fingerprint(paid_hashes),
    )
