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

from seiyuu.attribute.aliases import resolve_registry_aliases
from seiyuu.attribute.cache import AttributionCache, ChunkCacheKey
from seiyuu.attribute.chunking import Chunk, chunk_blocks
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    CharacterRegistry,
    ChunkAttribution,
    FlaggedBlock,
    Segment,
    SegmentType,
)
from seiyuu.attribute.providers.base import (
    AttributionError,
    AttributionLLM,
    MalformedOutputError,
)
from seiyuu.attribute.registry import resolve_chunk
from seiyuu.attribute.validate import ReconstructionFailure, find_reconstruction_failures
from seiyuu.ingest.models import Block, BlockType, NormalizedBook

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


def _drop_superseded_notes(notes: list[str], merged_names: set[str]) -> list[str]:
    """Drop incremental 'not merging X' notes the alias post-pass later resolved by merging X."""
    return [
        n
        for n in notes
        if not (n.startswith("not merging") and any(f"'{m}'" in n for m in merged_names))
    ]


def _fallback_segments(blocks: list[Block]) -> list[Segment]:
    """Whole-block narration at confidence 0.0 — used when a chunk can't be attributed."""
    return [
        Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text, confidence=0.0)
        for b in blocks
    ]


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
) -> AttributionReport:
    """Attribute a book (or a 1-based subset of ``chapters``) into segments + a registry."""
    say = progress or (lambda _msg: None)
    wanted = set(chapters)
    unknown = wanted - set(range(1, len(book.chapters) + 1))
    if unknown:
        raise AttributionError(
            f"{book.book_meta.book_id}: no such chapter(s) {sorted(unknown)} "
            f"(book has {len(book.chapters)})"
        )

    registry = CharacterRegistry()
    notes: list[str] = []
    flagged: list[FlaggedBlock] = []
    out_chapters: list[AttributedChapter] = []

    for ci, chapter in enumerate(book.chapters, start=1):
        if wanted and ci not in wanted:
            continue
        say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}")

        paragraphs = [b for b in chapter.blocks if b.type is BlockType.PARAGRAPH]
        chunks = chunk_blocks(
            paragraphs, budget_tokens=budget_tokens, overlap_blocks=overlap_blocks
        )
        by_block: dict[str, list[Segment]] = {}

        for chunk in chunks:
            key = ChunkCacheKey(
                book_id=book.book_meta.book_id,
                chapter_index=ci,
                chunk_hash=chunk.content_hash,
                provider_id=provider.provider_id,
                model_id=provider.model_id,
                prompt_version=provider.prompt_version,
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
                        by_block[block.id] = _fallback_segments([block])
                    continue
                attribution = outcome.attribution
                cache.put(key, attribution)
                say(f"{label}: {len(attribution.segments)} segments, {outcome.attempts} attempt(s)")

            resolved, chunk_notes = resolve_chunk(
                registry, attribution.segments, attribution.characters
            )
            notes.extend(chunk_notes)
            for seg in resolved:
                by_block.setdefault(seg.block_id, []).append(seg)

        segments: list[Segment] = []
        for block in chapter.blocks:
            if block.type is BlockType.HEADING:
                segments.append(
                    Segment(block_id=block.id, type=SegmentType.NARRATION, text=block.text)
                )
            elif block.type is BlockType.PARAGRAPH:
                segments.extend(by_block.get(block.id, []))
        out_chapters.append(AttributedChapter(index=ci, title=chapter.title, segments=segments))

    # Once the whole registry exists, merge provably-same characters (honorific variants,
    # subsumed aliases) and flag the ambiguous. Then remap any absorbed speaker ids on the
    # already-resolved segments. This touches only the in-memory report, never the cache.
    pre_names = {c.id: c.canonical_name for c in registry.characters}
    id_remap, alias_notes = resolve_registry_aliases(registry, out_chapters)
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
        prompt_version=provider.prompt_version,
        registry=registry,
        chapters=out_chapters,
        flagged=flagged,
        registry_notes=notes,
    )


def write_attribution(report: AttributionReport, book_dir: Path) -> Path:
    path = Path(book_dir) / ATTRIBUTION_NAME
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
