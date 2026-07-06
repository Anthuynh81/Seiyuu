"""Single-voice render: normalized JSON → cached canonical segment WAVs + manifest.

Only speakable blocks (paragraph, heading) become synthesis segments; scene
breaks pass through to the manifest as pause markers. Every synthesis call
goes through the segment cache.
"""

import hashlib
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf

if TYPE_CHECKING:
    from seiyuu.api.concurrency import BorrowBroker
    from seiyuu.normalize.lexicon import CompiledLexicon

from seiyuu.attribute.models import AttributionReport
from seiyuu.engines import TTSEngine, get_engine
from seiyuu.gpu import get_gpu_manager
from seiyuu.ingest.models import BlockType, NormalizedBook
from seiyuu.normalize import normalize_text, profile_for
from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest, VoiceUse
from seiyuu.repository import atomic_write_text
from seiyuu.validate import SegmentWords, ValidationResult, Validator, WordTiming
from seiyuu.voices import (
    VoiceAssignment,
    VoiceLibrary,
    VoiceLibraryError,
    ensure_cloud_voice,
    render_voice_args,
    resolve_voice,
)
from seiyuu.voices.models import VoiceMeta

MANIFEST_NAME = "manifest.json"


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


def _synthesize_validated(
    engine: TTSEngine,
    text: str,
    voice_arg: str,
    settings: dict[str, Any],
    seed: int | None,
    *,
    validator: Validator | None,
    max_retries: int,
    cache_dir: Path,
) -> tuple[Any, ValidationResult | None, int, list[WordTiming] | None]:
    """Synthesize one segment, returning (audio, validation, attempts, words).

    Deterministic engines (or when no validator is supplied) synthesize once and skip
    validation (and produce no words — those engines align lazily on first Listen). LLM-style
    engines (`requires_validation`) transcribe each attempt and, on a failure, retry with a
    fresh seed up to `max_retries` more times, keeping the best-scoring attempt. The kept
    attempt's word timings (F2) ride out of the SAME transcription used for its score, so the
    seek points target exactly the audio that ships. A persistent failure is returned (not
    raised) so the caller can flag it for review rather than silently ship — or drop — it.
    """
    base = dict(settings)
    if not engine.requires_validation or validator is None:
        synth = {**base, **({"seed": seed} if seed is not None else {})}
        return engine.synthesize(text, voice_arg, synth), None, 1, None

    tmp = Path(cache_dir) / "_validate.tmp.wav"
    best_audio: Any = None
    best_result: ValidationResult | None = None
    best_words: list[WordTiming] | None = None
    attempts = 0
    try:
        for i in range(max_retries + 1):
            attempts = i + 1
            attempt_seed = seed if (i == 0 or seed is None) else seed + i
            synth = {**base, **({"seed": attempt_seed} if attempt_seed is not None else {})}
            audio = engine.synthesize(text, voice_arg, synth)
            audio.save(tmp)
            result, words = _validate_capturing_words(validator, tmp, text)
            if best_result is None or result.score > best_result.score:
                best_audio, best_result, best_words = audio, result, words
            if result.ok:
                break
    finally:
        tmp.unlink(missing_ok=True)
    return best_audio, best_result, attempts, best_words


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
) -> RenderResult:
    """Render a book (or a 1-based subset of `chapters`) with one voice.

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

    rendered_chapters: list[RenderedChapter] = []
    profile = profile_for(engine.engine_id)
    synthesized = cache_hits = validation_failures = 0
    paid_spent = 0.0
    try:
        for ci, chapter in enumerate(book.chapters, start=1):
            if wanted and ci not in wanted:
                continue
            check()
            lend()
            say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}")
            segments: list[RenderedSegment] = []
            for block in chapter.blocks:
                check()
                lend()
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
                wav_path = cache.get(key)
                if wav_path is not None:
                    cache_hits += 1
                    duration = sf.info(str(wav_path)).duration
                    validation = cache.get_validation(key)
                    attempts = 1
                    if validation is None and engine.requires_validation and validator is not None:
                        # A cache hit with no stored verdict (crash between the wav write and
                        # the verdict write, or a pre-M4 segment) must not ship unvalidated:
                        # re-validate the cached wav and persist the verdict so it is counted
                        # and flagged exactly like a fresh render.
                        validation, _words = _validate_capturing_words(validator, wav_path, text)
                        cache.put_validation(key, validation)
                else:
                    paid_spent = _gate_paid(
                        engine, text, allow_paid, paid_spent, max_paid_usd,
                        book_id=book.book_meta.book_id, block_id=block.id, voice=voice_id,
                    )  # fmt: skip
                    try:
                        ctx = (
                            gpu.acquire(engine, engine.engine_id)
                            if engine.uses_gpu
                            else nullcontext()
                        )
                        with ctx:
                            audio, validation, attempts, words = _synthesize_validated(
                                engine, text, engine_voice, effective_settings, effective_seed,
                                validator=validator, max_retries=validation_max_retries,
                                cache_dir=cache.cache_dir,
                            )  # fmt: skip
                    except RenderError:
                        raise
                    except Exception as exc:
                        raise RenderError(
                            f"synthesis failed: book={book.book_meta.book_id} "
                            f"chapter={ci} ({chapter.title!r}) block={block.id} "
                            f"engine={engine.engine_id} voice={voice_id}: {exc}"
                        ) from exc
                    wav_path = cache.put(key, audio)
                    if validation is not None:
                        cache.put_validation(key, validation)
                    duration = audio.duration_seconds
                    if words is not None:  # F2 piggyback: validated engines cache words inline
                        cache.put_words(key, SegmentWords(words=words, audio_duration=duration))
                    synthesized += 1
                if validation is not None and not validation.ok:
                    validation_failures += 1
                    say(
                        f"  ! validation failed (score {validation.score}) block={block.id} "
                        f"after {attempts} attempt(s) — flagged for review"
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
                        synth_attempts=attempts,
                    )
                )
            rendered_chapters.append(
                RenderedChapter(index=ci, title=chapter.title, segments=segments)
            )
    finally:
        # Stop lending BEFORE the engine is unloaded: an in-flight request gets an
        # immediate None (soft retry), never an about-to-be-freed instance.
        if broker is not None:
            broker.close()
        if engine.uses_gpu:
            # free only what this render could have loaded: a cloud-only render must not
            # evict another consumer's resident model from the shared manager
            gpu.free_all()

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
    manifest_path = book_output_dir / MANIFEST_NAME
    atomic_write_text(manifest_path, manifest.model_dump_json(indent=2))
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
) -> RenderResult:
    """Multi-voice render: attribution segments + per-character voices → cached WAVs + manifest.

    Reads the attribution report (segments + resolved speaker ids), the normalized book
    (scene-break pause markers + reading order), the voice library, and the assignment. Each
    segment's voice is resolved, its engine acquired through the GPU manager (one heavy model
    resident at a time), its text normalized per the engine profile, and synthesized through
    the FROZEN SegmentKey. Segments are emitted in reading order; the per-segment cache key
    makes that order-independent for caching.

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
            extra = {"voices_dir": library.voices_dir} if engine_id == "chatterbox" else {}
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
    try:
        for ci, chapter in enumerate(book.chapters, start=1):
            if (wanted and ci not in wanted) or ci not in attributed:
                continue
            check()
            lend()
            say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}")
            by_block: dict[str, list] = {}
            for seg in attributed[ci].segments:
                by_block.setdefault(seg.block_id, []).append(seg)

            rendered: list[RenderedSegment] = []
            for block in chapter.blocks:
                if block.type is BlockType.SCENE_BREAK:
                    rendered.append(RenderedSegment(block_id=block.id, type=block.type))
                    continue
                for seg in by_block.get(block.id, []):
                    check()
                    lend()
                    voice_id = resolve_voice(seg, assignment)
                    meta = meta_for(voice_id)  # verifies consent on first sight
                    engine = engine_for(meta.engine)
                    if broker is not None and engine.uses_gpu:
                        # this segment's engine is what will be resident; offer it at the
                        # next yield point (lend() reads these via closure)
                        broker.publish(meta.engine, engine)
                        lent_id, lent_engine = meta.engine, engine
                    text = normalize_text(
                        seg.text, profile=profile_for(meta.engine), lexicon=lexicon
                    )
                    engine_voice, settings = render_voice_args(meta)
                    key = SegmentKey.build(
                        engine=meta.engine,
                        engine_model_version=engine.model_version,
                        voice_id=voice_id,
                        settings=settings,
                        seed=meta.seed,
                        normalized_text=text,
                    )
                    wav_path = cache.get(key)
                    if wav_path is not None:
                        cache_hits += 1
                        duration = sf.info(str(wav_path)).duration
                        validation = cache.get_validation(key)
                        attempts = 1
                        if (
                            validation is None
                            and engine.requires_validation
                            and validator is not None
                        ):
                            # A cache hit with no stored verdict (crash between the wav write
                            # and the verdict write, or a pre-M4 segment) must not ship
                            # unvalidated: re-validate the cached wav and persist the verdict
                            # so it is counted and flagged exactly like a fresh render.
                            validation, _words = _validate_capturing_words(
                                validator, wav_path, text
                            )
                            cache.put_validation(key, validation)
                    else:
                        paid_spent = _gate_paid(
                            engine, text, allow_paid, paid_spent, max_paid_usd,
                            book_id=book.book_meta.book_id, block_id=block.id, voice=voice_id,
                        )  # fmt: skip
                        try:
                            synth_voice = engine_voice
                            if meta.engine == "elevenlabs":  # resolve/create the cloud voice
                                synth_voice = ensure_cloud_voice(
                                    meta, engine.client, library, max_slots=cloud_max_slots
                                )
                            used_gpu = used_gpu or engine.uses_gpu
                            ctx = (
                                gpu.acquire(engine, engine.engine_id)
                                if engine.uses_gpu
                                else nullcontext()
                            )
                            with ctx:
                                audio, validation, attempts, words = _synthesize_validated(
                                    engine, text, synth_voice, settings, meta.seed,
                                    validator=validator, max_retries=validation_max_retries,
                                    cache_dir=cache.cache_dir,
                                )  # fmt: skip
                        except RenderError:
                            raise
                        except Exception as exc:
                            raise RenderError(
                                f"synthesis failed: book={book.book_meta.book_id} chapter={ci} "
                                f"block={block.id} voice={voice_id} engine={meta.engine}: {exc}"
                            ) from exc
                        wav_path = cache.put(key, audio)
                        if validation is not None:
                            cache.put_validation(key, validation)
                        duration = audio.duration_seconds
                        if words is not None:  # F2 piggyback: validated engines cache words inline
                            cache.put_words(key, SegmentWords(words=words, audio_duration=duration))
                        synthesized += 1
                    if validation is not None and not validation.ok:
                        validation_failures += 1
                        say(
                            f"  ! validation failed (score {validation.score}) block={block.id} "
                            f"voice={voice_id} after {attempts} attempt(s) — flagged for review"
                        )
                    voices_used.setdefault(
                        voice_id,
                        VoiceUse(
                            engine=meta.engine,
                            engine_model_version=engine.model_version,
                            kind=meta.kind.value,
                        ),
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
                            synth_attempts=attempts,
                        )
                    )
            rendered_chapters.append(
                RenderedChapter(index=ci, title=chapter.title, segments=rendered)
            )
    finally:
        # Stop lending BEFORE the engine is unloaded: an in-flight request gets an
        # immediate None (soft retry), never an about-to-be-freed instance.
        if broker is not None:
            broker.close()
        # free the GPU for the next stage/process — but only if this render acquired it;
        # a cloud-only render must not evict another consumer's resident model
        if used_gpu:
            gpu.free_all()

    manifest = RenderManifest(
        book_id=book.book_meta.book_id,
        book_title=book.book_meta.title,
        chapters=rendered_chapters,
        voices_used=voices_used,
        assignment=assignment.model_dump(mode="json"),
        validation_failures=validation_failures,
    )
    manifest_path = book_output_dir / MANIFEST_NAME
    atomic_write_text(manifest_path, manifest.model_dump_json(indent=2))
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
            extra = {"voices_dir": library.voices_dir} if engine_id == "chatterbox" else {}
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
        by_block: dict[str, list] = {}
        for seg in attributed[ci].segments:
            by_block.setdefault(seg.block_id, []).append(seg)
        for block in chapter.blocks:
            if block.type is BlockType.SCENE_BREAK:
                continue
            for seg in by_block.get(block.id, []):
                # resolved id, NOT meta.voice_id: must build the EXACT SegmentKey the render
                # loop will use, or the gate authorizes a different bill than render runs up
                voice_id = resolve_voice(seg, assignment)
                meta = meta_for(voice_id)
                engine = engine_for(meta.engine)
                text = normalize_text(seg.text, profile=profile_for(meta.engine), lexicon=lexicon)
                _, settings = render_voice_args(meta)
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
                if cache.get(key) is not None:
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
            if cache.get(key) is not None:
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
