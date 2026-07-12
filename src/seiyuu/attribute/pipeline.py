"""Attribution stage: normalized JSON + provider → attributed segments + registry.

Per chapter: paragraphs are chunked (context overlap, exclusive ownership) and sent to the
provider; headings become narration directly; scene breaks carry no segments. Each chunk
is validated against the reconstruction invariant and retried locally; a chunk that still
fails is escalated (hybrid mode) or flagged, with a verbatim narration fallback so the
pipeline always yields renderable output. Successful chunks are cached; flagged chunks are
not (so a better model can re-attribute later). The running registry threads across
chapters in order, which keeps the SPEC cache key (no registry component) reproducible.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from seiyuu.attribute.aliases import AliasResolver, resolve_registry_aliases
from seiyuu.attribute.cache import AttributionCache, ChunkCacheKey
from seiyuu.attribute.chunking import Chunk, chunk_blocks
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    CharacterRegistry,
    ChunkAttribution,
    EmotionVerdict,
    FlaggedBlock,
    Segment,
    SegmentType,
)
from seiyuu.attribute.providers.base import (
    SINGLE_QUOTE_KEY_SUFFIX,
    AttributionError,
    AttributionLLM,
    MalformedOutputError,
)
from seiyuu.attribute.registry import resolve_chunk
from seiyuu.attribute.spans import (
    ConventionDetection,
    DialogueConvention,
    detect_dialogue_convention,
)
from seiyuu.attribute.validate import ReconstructionFailure, find_reconstruction_failures
from seiyuu.ingest.models import Block, BlockType, NormalizedBook
from seiyuu.repository import atomic_write_text

ATTRIBUTION_NAME = "attribution.json"

# AttributionError is defined once in providers.base and re-exported here so that errors
# raised by a provider and by the pipeline are the same type (the CLI catches one).


@dataclass
class _ChunkOutcome:
    attribution: ChunkAttribution | None  # validated, owned-only; None if flagged
    attempts: int
    failures: list = field(default_factory=list)


def _attribute_chunk_validated(
    provider: AttributionLLM,
    chunk: Chunk,
    registry: CharacterRegistry,
    max_retries: int,
) -> _ChunkOutcome:
    """Call the provider, drop non-owned segments, check reconstruction, retry on failure.

    A per-attempt bad-output error (invalid JSON, schema violation) is treated like a
    reconstruction failure: retry, then flag. Fatal errors (unreachable backend,
    truncation/config) are NOT caught here — they abort the run with guidance.
    """
    last_failures: list = []
    for attempt in range(max_retries + 1):
        try:
            attribution = provider.attribute_chunk(chunk, registry, attempt)
        except MalformedOutputError as exc:
            last_failures = [
                ReconstructionFailure(b.id, f"unusable model output: {exc}")
                for b in chunk.owned_blocks
            ]
            continue
        owned = [s for s in attribution.segments if s.block_id in chunk.owned_ids]
        last_failures = find_reconstruction_failures(chunk.owned_blocks, owned)
        if not last_failures:
            return _ChunkOutcome(
                ChunkAttribution(segments=owned, characters=attribution.characters),
                attempts=attempt + 1,
            )
    return _ChunkOutcome(None, attempts=max_retries + 1, failures=last_failures)


_SUPERSEDABLE_PREFIXES = ("not merging", "alias: ambiguous", "alias: low-evidence")


def _drop_superseded_notes(notes: list[str], merged_names: set[str]) -> list[str]:
    """Drop review notes the alias post-pass later resolved by merging one of their names.

    Covers the incremental 'not merging X' flags AND the deterministic 'alias: ambiguous'/
    'alias: low-evidence' flags: once adjudication merges a character named X, any note that
    named X as un-mergeable is stale and would sit misleadingly beside the merged record.
    """
    return [
        n
        for n in notes
        if not (n.startswith(_SUPERSEDABLE_PREFIXES) and any(f"'{m}'" in n for m in merged_names))
    ]


def _convention_note(detection: ConventionDetection) -> str:
    """The registry_notes entry surfacing a non-DOUBLE dialogue convention.

    The stable "dialogue convention:" prefix is what the CLI attribute command filters on to
    echo it; `seiyuu characters` and the API already print every registry note verbatim.
    """
    counts = (
        f"{detection.single_curly_runs} curly / {detection.single_straight_runs} straight "
        f"single-quote runs vs {detection.double_runs} double"
    )
    if detection.convention is DialogueConvention.SINGLE_CURLY:
        return (
            f"dialogue convention: single curly quotes (‘…’) detected ({counts}); "
            "splitting dialogue on single-quote boundaries"
        )
    if detection.convention is DialogueConvention.SINGLE_STRAIGHT:
        return (
            f"dialogue convention: straight single quotes suspected ({counts}); straight "
            "singles are indistinguishable from apostrophes so the splitter stays on double "
            "quotes — dialogue may be missed, review recommended"
        )
    return (
        f"dialogue convention: unclear ({counts}); splitter stays on double quotes — "
        "dialogue may be missed, review recommended"
    )


def _fallback_segments(blocks: list[Block]) -> list[Segment]:
    """Whole-block narration at confidence 0.0 — used when a chunk can't be attributed."""
    return [
        Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text, confidence=0.0)
        for b in blocks
    ]


def _aligned_emotions(attribution: ChunkAttribution, count: int) -> list[EmotionVerdict | None]:
    """The chunk's per-segment emotions, normalized to ``count`` entries (F2).

    A v5/v6 attribution carries one emotion per segment; a v3/v4-shaped cached row carries an
    empty list. Either way we return exactly ``count`` entries so it stays index-aligned to the
    resolved segments as they thread into the chapter.
    """
    emotions = attribution.segment_emotions
    if len(emotions) == count:
        return list(emotions)
    return [None] * count


def attribute_book(
    book: NormalizedBook,
    provider: AttributionLLM,
    *,
    cache: AttributionCache,
    budget_tokens: int = 3000,
    overlap_blocks: int = 2,
    max_local_retries: int = 2,
    escalation_provider: AttributionLLM | None = None,
    chapters: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
    resolver: AliasResolver | None = None,
    adjudication_confidence_threshold: float = 0.85,
    adjudication_candidate_cap: int = 40,
    adjudication_use_nicknames: bool = True,
) -> AttributionReport:
    """Attribute a book (or a 1-based subset of ``chapters``) into segments + a registry.

    ``check_cancel`` (when given) is called between chapters and between chunks and may
    raise to abort cooperatively; completed chunks are already cached and no report is
    written, so a re-run resumes from the cache.
    """
    say = progress or (lambda _msg: None)
    check = check_cancel or (lambda: None)
    wanted = set(chapters)
    unknown = wanted - set(range(1, len(book.chapters) + 1))
    if unknown:
        raise AttributionError(
            f"{book.book_meta.book_id}: no such chapter(s) {sorted(unknown)} "
            f"(book has {len(book.chapters)})"
        )

    # Book-level dialogue convention, detected ONCE over the FULL book (never the --chapter
    # subset, so partial runs classify — and key — identically). Only the unambiguous curly
    # form switches the splitter; the convention rides on the provider(s) because the span
    # split happens inside attribute_chunk.
    detection = detect_dialogue_convention(
        "\n".join(
            b.text for ch in book.chapters for b in ch.blocks if b.type is BlockType.PARAGRAPH
        )
    )
    single_curly = detection.convention is DialogueConvention.SINGLE_CURLY
    splitter_convention = (
        DialogueConvention.SINGLE_CURLY if single_curly else DialogueConvention.DOUBLE
    )
    provider.dialogue_convention = splitter_convention
    if escalation_provider is not None:
        escalation_provider.dialogue_convention = splitter_convention
    # A single-quote book attributed BEFORE convention detection existed left all-narration
    # rows under the same chunk hashes, so single-curly mode records the prompt_version KEY
    # COMPONENT with the "-sq" suffix. The key FORMAT is unchanged and double-convention
    # books keep byte-identical keys (their caches hit untouched).
    key_prompt_version = provider.prompt_version
    if single_curly and not key_prompt_version.endswith(SINGLE_QUOTE_KEY_SUFFIX):
        key_prompt_version += SINGLE_QUOTE_KEY_SUFFIX

    registry = CharacterRegistry()
    notes: list[str] = []
    flagged: list[FlaggedBlock] = []
    out_chapters: list[AttributedChapter] = []

    if detection.convention is not DialogueConvention.DOUBLE:
        note = _convention_note(detection)
        notes.append(note)
        say(note)

    for ci, chapter in enumerate(book.chapters, start=1):
        if wanted and ci not in wanted:
            continue
        check()
        say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}")

        paragraphs = [b for b in chapter.blocks if b.type is BlockType.PARAGRAPH]
        chunks = chunk_blocks(
            paragraphs, budget_tokens=budget_tokens, overlap_blocks=overlap_blocks
        )
        # Each entry is (segment, emotion) so the F2 emotion rides alongside its segment through
        # resolution and block regrouping and can never desync from segment order.
        by_block: dict[str, list[tuple[Segment, EmotionVerdict | None]]] = {}

        for chunk in chunks:
            check()
            key = ChunkCacheKey(
                book_id=book.book_meta.book_id,
                chapter_index=ci,
                chunk_hash=chunk.content_hash,
                provider_id=provider.provider_id,
                model_id=provider.model_id,
                prompt_version=key_prompt_version,
            )
            label = f"  chunk {chunk.index + 1}/{len(chunks)} ({len(chunk.owned_ids)} blocks)"
            attribution = cache.get(key)
            if attribution is not None:
                say(f"{label}: cached")
            else:
                outcome = _attribute_chunk_validated(provider, chunk, registry, max_local_retries)
                if outcome.attribution is None and escalation_provider is not None:
                    say(f"{label}: escalating to {escalation_provider.provider_id}")
                    outcome = _attribute_chunk_validated(
                        escalation_provider, chunk, registry, max_local_retries
                    )
                if outcome.attribution is None:
                    why = outcome.failures[0].reason[:80] if outcome.failures else "unknown"
                    say(f"{label}: FLAGGED after {outcome.attempts} attempt(s) — {why}")
                    for failure in outcome.failures:
                        flagged.append(
                            FlaggedBlock(
                                block_id=failure.block_id, chapter_index=ci, reason=failure.reason
                            )
                        )
                    for block in chunk.owned_blocks:
                        by_block[block.id] = [(seg, None) for seg in _fallback_segments([block])]
                    continue
                attribution = outcome.attribution
                cache.put(key, attribution)
                say(f"{label}: {len(attribution.segments)} segments, {outcome.attempts} attempt(s)")

            resolved, chunk_notes = resolve_chunk(
                registry, attribution.segments, attribution.characters
            )
            notes.extend(chunk_notes)
            # resolve_chunk preserves order + count, so the aligned emotions zip 1:1.
            chunk_emotions = _aligned_emotions(attribution, len(resolved))
            for seg, emotion in zip(resolved, chunk_emotions, strict=True):
                by_block.setdefault(seg.block_id, []).append((seg, emotion))

        segments: list[Segment] = []
        segment_emotions: list[EmotionVerdict | None] = []
        for block in chapter.blocks:
            if block.type is BlockType.HEADING:
                segments.append(
                    Segment(block_id=block.id, type=SegmentType.NARRATION, text=block.text)
                )
                segment_emotions.append(None)
            elif block.type is BlockType.PARAGRAPH:
                for seg, emotion in by_block.get(block.id, []):
                    segments.append(seg)
                    segment_emotions.append(emotion)
        out_chapters.append(
            AttributedChapter(
                index=ci,
                title=chapter.title,
                segments=segments,
                segment_emotions=segment_emotions,
            )
        )

    # Once the whole registry exists, merge provably-same characters (honorific variants,
    # subsumed aliases) and flag the ambiguous. Then remap any absorbed speaker ids on the
    # already-resolved segments. This touches only the in-memory report, never the cache.
    pre_names = {c.id: c.canonical_name for c in registry.characters}
    id_remap, alias_notes = resolve_registry_aliases(
        registry,
        out_chapters,
        resolver=resolver,
        confidence_threshold=adjudication_confidence_threshold,
        candidate_cap=adjudication_candidate_cap,
        use_nicknames=adjudication_use_nicknames,
    )
    notes = _drop_superseded_notes(notes, {pre_names[loser] for loser in id_remap})
    notes.extend(alias_notes)
    if id_remap:
        for chapter_out in out_chapters:
            chapter_out.segments = [
                seg.model_copy(update={"speaker": id_remap[seg.speaker]})
                if seg.speaker in id_remap
                else seg
                for seg in chapter_out.segments
            ]

    return AttributionReport(
        book_id=book.book_meta.book_id,
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        prompt_version=key_prompt_version,
        registry=registry,
        chapters=out_chapters,
        flagged=flagged,
        registry_notes=notes,
    )


def write_attribution(report: AttributionReport, book_dir: Path) -> Path:
    path = Path(book_dir) / ATTRIBUTION_NAME
    return atomic_write_text(path, report.model_dump_json(indent=2))
