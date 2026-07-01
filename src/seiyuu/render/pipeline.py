"""Single-voice render: normalized JSON → cached canonical segment WAVs + manifest.

Only speakable blocks (paragraph, heading) become synthesis segments; scene
breaks pass through to the manifest as pause markers. Every synthesis call
goes through the segment cache.
"""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from seiyuu.attribute.models import AttributionReport
from seiyuu.engines import TTSEngine, get_engine
from seiyuu.gpu import get_gpu_manager
from seiyuu.ingest.models import BlockType, NormalizedBook
from seiyuu.normalize import normalize_text, profile_for
from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest, VoiceUse
from seiyuu.repository import atomic_write_text
from seiyuu.validate import ValidationResult, Validator
from seiyuu.voices import (
    VoiceAssignment,
    VoiceKind,
    VoiceLibrary,
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


def _gate_paid(engine: TTSEngine, text: str, allow_paid: bool, *, book_id, block_id, voice) -> None:
    """Refuse a paid synthesis unless explicitly authorized — no automatic code path may bill."""
    cost = engine.cost_estimate(text)
    if cost > 0 and not allow_paid:
        raise RenderError(
            f"refusing paid synthesis (~${cost:.4f}) book={book_id} block={block_id} "
            f"voice={voice} engine={engine.engine_id} without cost confirmation; confirm the "
            f"estimate first (CLI: `seiyuu estimate-cost`, then render with --confirm-cost)"
        )


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
) -> tuple[Any, ValidationResult | None, int]:
    """Synthesize one segment, returning (audio, validation, attempts).

    Deterministic engines (or when no validator is supplied) synthesize once and skip
    validation. LLM-style engines (`requires_validation`) transcribe each attempt and, on a
    failure, retry with a fresh seed up to `max_retries` more times, keeping the best-scoring
    attempt. A persistent failure is returned (not raised) so the caller can flag it for review
    rather than silently ship — or drop — the segment.
    """
    base = dict(settings)
    if not engine.requires_validation or validator is None:
        synth = {**base, **({"seed": seed} if seed is not None else {})}
        return engine.synthesize(text, voice_arg, synth), None, 1

    tmp = Path(cache_dir) / "_validate.tmp.wav"
    best_audio: Any = None
    best_result: ValidationResult | None = None
    attempts = 0
    try:
        for i in range(max_retries + 1):
            attempts = i + 1
            attempt_seed = seed if (i == 0 or seed is None) else seed + i
            synth = {**base, **({"seed": attempt_seed} if attempt_seed is not None else {})}
            audio = engine.synthesize(text, voice_arg, synth)
            audio.save(tmp)
            result = validator.validate(tmp, text)
            if best_result is None or result.score > best_result.score:
                best_audio, best_result = audio, result
            if result.ok:
                break
    finally:
        tmp.unlink(missing_ok=True)
    return best_audio, best_result, attempts


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
    validator: Validator | None = None,
    validation_max_retries: int = 2,
    allow_paid: bool = False,
) -> RenderResult:
    """Render a book (or a 1-based subset of `chapters`) with one voice."""
    settings = settings or {}
    book_output_dir = Path(book_output_dir)
    cache = SegmentCache(book_output_dir / "cache")
    say = progress or (lambda _msg: None)

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
    for ci, chapter in enumerate(book.chapters, start=1):
        if wanted and ci not in wanted:
            continue
        say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}")
        segments: list[RenderedSegment] = []
        for block in chapter.blocks:
            if block.type is BlockType.SCENE_BREAK:
                segments.append(RenderedSegment(block_id=block.id, type=block.type))
                continue
            text = normalize_text(block.text, profile=profile)
            key = SegmentKey.build(
                engine=engine.engine_id,
                engine_model_version=engine.model_version,
                voice_id=voice_id,
                settings=settings,
                seed=seed,
                normalized_text=text,
            )
            wav_path = cache.get(key)
            if wav_path is not None:
                cache_hits += 1
                duration = sf.info(str(wav_path)).duration
                validation = cache.get_validation(key)
                attempts = 1
            else:
                _gate_paid(
                    engine, text, allow_paid,
                    book_id=book.book_meta.book_id, block_id=block.id, voice=voice_id,
                )  # fmt: skip
                try:
                    audio, validation, attempts = _synthesize_validated(
                        engine, text, voice_id, settings, seed,
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
                synthesized += 1
                duration = audio.duration_seconds
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
                    seed=seed,
                    settings_hash=key.settings_hash,
                    validation=validation,
                    synth_attempts=attempts,
                )
            )
        rendered_chapters.append(RenderedChapter(index=ci, title=chapter.title, segments=segments))

    manifest = RenderManifest(
        book_id=book.book_meta.book_id,
        book_title=book.book_meta.title,
        engine=engine.engine_id,
        engine_model_version=engine.model_version,
        voice_id=voice_id,
        settings=settings,
        seed=seed,
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
    cloud_max_slots: int = 10,
) -> RenderResult:
    """Multi-voice render: attribution segments + per-character voices → cached WAVs + manifest.

    Reads the attribution report (segments + resolved speaker ids), the normalized book
    (scene-break pause markers + reading order), the voice library, and the assignment. Each
    segment's voice is resolved, its engine acquired through the GPU manager (one heavy model
    resident at a time), its text normalized per the engine profile, and synthesized through
    the FROZEN SegmentKey. Segments are emitted in reading order; the per-segment cache key
    makes that order-independent for caching.
    """
    book_output_dir = Path(book_output_dir)
    cache = SegmentCache(book_output_dir / "cache")
    say = progress or (lambda _msg: None)
    gpu = gpu or get_gpu_manager()

    wanted = set(chapters)
    attributed = {ch.index: ch for ch in report.chapters}
    engines: dict[str, TTSEngine] = {}
    metas: dict[str, VoiceMeta] = {}
    voices_used: dict[str, VoiceUse] = {}

    def engine_for(engine_id: str) -> TTSEngine:
        if engine_id not in engines:
            extra = {"voices_dir": library.voices_dir} if engine_id == "chatterbox" else {}
            engines[engine_id] = get_engine(engine_id, **extra)
        return engines[engine_id]

    def meta_for(voice_id: str) -> VoiceMeta:
        if voice_id not in metas:
            metas[voice_id] = library.load(voice_id)
        return metas[voice_id]

    rendered_chapters: list[RenderedChapter] = []
    synthesized = cache_hits = validation_failures = 0
    try:
        for ci, chapter in enumerate(book.chapters, start=1):
            if (wanted and ci not in wanted) or ci not in attributed:
                continue
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
                    voice_id = resolve_voice(seg, assignment)
                    meta = meta_for(voice_id)
                    if meta.kind is VoiceKind.CLONED and not meta.consent_attested:
                        raise RenderError(
                            f"voice {voice_id!r} (cloned) has no consent attestation; "
                            f"refusing to render"
                        )
                    engine = engine_for(meta.engine)
                    text = normalize_text(seg.text, profile=profile_for(meta.engine))
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
                    else:
                        _gate_paid(
                            engine, text, allow_paid,
                            book_id=book.book_meta.book_id, block_id=block.id, voice=voice_id,
                        )  # fmt: skip
                        try:
                            synth_voice = engine_voice
                            if meta.engine == "elevenlabs":  # resolve/create the cloud voice
                                synth_voice = ensure_cloud_voice(
                                    meta, engine.client, library, max_slots=cloud_max_slots
                                )
                            ctx = (
                                gpu.acquire(engine, engine.engine_id)
                                if engine.uses_gpu
                                else nullcontext()
                            )
                            with ctx:
                                audio, validation, attempts = _synthesize_validated(
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
                        synthesized += 1
                        duration = audio.duration_seconds
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
        gpu.free_all()  # free the GPU for the next stage/process

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
                meta = meta_for(resolve_voice(seg, assignment))
                engine = engine_for(meta.engine)
                text = normalize_text(seg.text, profile=profile_for(meta.engine))
                _, settings = render_voice_args(meta)
                key = SegmentKey.build(
                    engine=meta.engine,
                    engine_model_version=engine.model_version,
                    voice_id=meta.voice_id,
                    settings=settings,
                    seed=meta.seed,
                    normalized_text=text,
                )
                if cache.get(key) is not None:
                    cached += 1
                    continue
                cost = engine.cost_estimate(text)
                if cost > 0:
                    total += cost
                    paid += 1
                else:
                    free += 1
    return CostEstimate(
        total_usd=round(total, 4), paid_segments=paid, cached_segments=cached, free_segments=free
    )
