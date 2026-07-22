"""AttributionLLM interface (SPEC provider lineup).

LLM SDKs live ONLY inside ``attribute/providers/`` behind this interface; pipeline code
never imports an LLM SDK directly. The public ``attribute_chunk()`` is a template method:
it renders the versioned prompt, builds the schema from the pydantic models, calls the
backend's ``_complete_json()``, and validates the result into a :class:`ChunkAttribution`.
Subclasses implement only the transport. Schema-enforced output (Ollama structured
outputs, Anthropic tool schema) makes malformed JSON impossible rather than something to
parse-and-retry.
"""

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

from seiyuu.attribute.chunking import Chunk
from seiyuu.attribute.models import (
    BlockSpeaker,
    CharacterMention,
    CharacterRegistry,
    ChunkAttribution,
    ChunkLabels,
    EmotionVerdict,
    QuoteSpeaker,
    Segment,
    SegmentType,
    ThoughtVerdict,
)
from seiyuu.attribute.spans import (
    DialogueConvention,
    Span,
    quoted_ordinals,
    thought_candidate_spans,
)

# Prompt versions that use the F1 per-quote (hybrid) + F2 emotion contract: owned blocks with
# >1 quote are indexed with ⟦Q{ordinal}⟧ markers and the model labels quotes BY INDEX. v6 is
# v5 plus the thought-candidates section. v3/v4 stay on the whole-block-speaker contract.
_PER_QUOTE_VERSIONS = frozenset({"v5", "v6"})

# Cache-key marker for single-curly (UK) splitter mode: the pipeline suffixes the
# prompt_version KEY COMPONENT with "-sq" (e.g. "v5-sq") so a single-quote book attributed
# before convention detection existed (all-narration rows under the same chunk hashes) can
# never be replayed. The suffix can therefore reach a provider (a user pinning
# --prompt-version v5-sq), so every EXACT match on a version string — the prompt-file lookup
# and the per-quote contract check — must compare the base form via base_prompt_version().
SINGLE_QUOTE_KEY_SUFFIX = "-sq"


def base_prompt_version(version: str) -> str:
    """The version without the single-quote cache suffix ("v5-sq" -> "v5"; "v5" unchanged)."""
    return version.removesuffix(SINGLE_QUOTE_KEY_SUFFIX)


# ⟦ ⟧ (U+27E6/U+27E7) never occur in ordinary prose, so a marker can't collide with real text.
# Markers live ONLY in the rendered prompt; assembly re-splits the RAW source (never the marked
# string), so they can never reach a Span, a Segment.text, or a CharacterMention.
_QUOTE_MARKER = "⟦Q{index}⟧"


class AttributionError(Exception):
    """Fatal attribution failure (unreachable backend, truncation/config, auth).

    The pipeline does NOT retry these — they abort the run with actionable guidance.
    """


class MalformedOutputError(AttributionError):
    """One attempt's output was unusable (invalid JSON, schema violation).

    A per-attempt failure: the pipeline retries, then flags the chunk's blocks for review
    with a verbatim-narration fallback. Subclasses AttributionError so the CLI still
    catches it if it ever escapes.
    """


@lru_cache
def _prompt_template(prompts_dir: Path, version: str) -> str:
    path = prompts_dir / "attribution" / f"{version}.md"
    if not path.is_file():
        raise AttributionError(f"attribution prompt not found: {path}")
    return path.read_text(encoding="utf-8")


@lru_cache
def adjudication_template(prompts_dir: Path, version: str) -> str:
    """Load the versioned alias-adjudication prompt (mirrors :func:`_prompt_template`)."""
    path = prompts_dir / "adjudication" / f"{version}.md"
    if not path.is_file():
        raise AttributionError(f"adjudication prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def chunk_label_schema() -> dict[str, Any]:
    """JSON schema for the model's RAW output (per-block span labels + character mentions)."""
    return ChunkLabels.model_json_schema()


def _render_blocks(blocks: list) -> str:
    return "\n\n".join(f"[{b.id}]\n{b.text}" for b in blocks) or "(none)"


def _render_owned_blocks_indexed(chunk: Chunk, spans_by_block: dict[str, list[Span]]) -> str:
    """Render owned blocks for the per-quote prompt (F1, v5/v6), HYBRID by design.

    A block with more than one quoted span gets a ``⟦Q{ordinal}⟧`` marker before each quoted
    run, so the model can label quotes by index without ever counting. A single-quote block
    stays plain (byte-identical to the v3 render), keeping the dominant path on the proven
    whole-block contract. Ordinals come from :func:`quoted_ordinals` — the SAME helper
    ``_assemble_segments`` keys on — so the two sides can never disagree. Markers are inserted
    only into this display string; the block's real text is untouched.
    """
    rendered: list[str] = []
    for block in chunk.owned_blocks:
        spans = spans_by_block[block.id]
        ordinals = quoted_ordinals(spans)
        if len(ordinals) <= 1:
            rendered.append(f"[{block.id}]\n{block.text}")
            continue
        marker_at = {span: ordinal for ordinal, span in ordinals}
        parts: list[str] = []
        for span in spans:
            if span in marker_at:
                parts.append(_QUOTE_MARKER.format(index=marker_at[span]))
            parts.append(span.text)
        rendered.append(f"[{block.id}]\n{''.join(parts)}")
    return "\n\n".join(rendered) or "(none)"


def _render_thought_candidates(spans_by_block: dict[str, list[Span]]) -> str:
    """The deterministic thought candidates (candidate_id + its exact text) for the prompt."""
    lines = [
        f"- candidate_id: {span.candidate_id}\n  text: {span.text.strip()!r}"
        for spans in spans_by_block.values()
        for span in spans
        if span.candidate_id
    ]
    return "\n".join(lines) or "(none)"


def render_prompt(
    template: str,
    registry: CharacterRegistry,
    chunk: Chunk,
    thought_candidates: str = "(none)",
    owned_render: str | None = None,
) -> str:
    owned_ids = chunk.owned_ids
    context_blocks = [b for b in chunk.blocks if b.id not in owned_ids]
    # Render the registry in the SAME shape the model must emit (CharacterMention: a `name`
    # field, no `id`/`canonical_name`). Showing it the internal Character shape made the
    # model copy `canonical_name`/`id` into its output and drop the required `name`.
    registry_json = json.dumps(
        [
            {
                "name": c.canonical_name,
                "aliases": c.aliases,
                "gender": c.gender,
                "age_hint": c.age_hint,
                "description": c.description,
            }
            for c in registry.characters
        ],
        indent=2,
        ensure_ascii=False,
    )
    # Literal replacement, not str.format — the prompt contains JSON examples with braces.
    # The {thought_candidates} placeholder only exists in the thought-aware (v4/v6) prompt; on
    # v3/v5 the replace is a no-op. ``owned_render`` is the per-quote-indexed block render for
    # v5/v6; when None (v3/v4) we fall back to the plain whole-block render, byte-identical.
    owned = owned_render if owned_render is not None else _render_blocks(chunk.owned_blocks)
    return (
        template.replace("{registry_json}", registry_json)
        .replace("{context_blocks}", _render_blocks(context_blocks))
        .replace("{blocks}", owned)
        .replace("{thought_candidates}", thought_candidates)
    )


class AttributionLLM(ABC):
    provider_id: str
    # Local providers hold model weights in VRAM (via Ollama) and must go through the GPU
    # resource manager; cloud providers (anthropic) set False so a network-only attribution
    # run never evicts a resident TTS model or serializes the manager. Default True: an
    # unknown future provider is presumed heavy until it says otherwise.
    uses_gpu: bool = True
    # A confirmed thought below this confidence degrades to narration (never a low-confidence
    # or guessed thought voice). Precision over recall, matching the alias adjudicator.
    thought_confidence_floor: float = 0.5
    # Book-level dialogue convention, set per run by the pipeline (attribute_book) before any
    # chunk call; only SINGLE_CURLY changes the span split. The class default keeps every
    # direct caller (tests, one-off structured calls) on the proven double-quote path.
    dialogue_convention: DialogueConvention = DialogueConvention.DOUBLE
    # Class-level default so scripted test providers that bypass __init__ still resolve it
    # (instances set it in __init__ as before).
    emit_thoughts: bool = False

    def __init__(
        self,
        *,
        model: str,
        prompts_dir: Path,
        prompt_version: str = "v1",
        emit_thoughts: bool = False,
    ) -> None:
        self.model_id = model
        self.prompts_dir = Path(prompts_dir)
        self.prompt_version = prompt_version
        # Opt-in (default OFF): when False the provider is byte-identical to the v3 path — no
        # prose sub-split, no candidates, no thought verdicts. The caller pairs it with the v4
        # prompt so the cache key (which includes prompt_version) keeps the two runs apart.
        self.emit_thoughts = emit_thoughts

    def unload(self) -> None:  # noqa: B027 — intentional no-op default; cloud providers override nothing
        """Free GPU memory (GpuConsumer protocol). No-op default; cloud providers need nothing."""

    def _spans_for(self, chunk: Chunk) -> dict[str, list[Span]]:
        """Each owned block's deterministic spans under this provider's convention.

        Thought-off (default): no italic offsets are passed, so the sub-split reproduces
        the quote-only split exactly and no candidates are generated — byte-identical to
        v3. Shared by ``attribute_chunk`` and ``narration_only_attribution`` so the two
        paths can never disagree about what needs the model."""
        return {
            b.id: thought_candidate_spans(
                b.id,
                b.text,
                b.italic_spans if self.emit_thoughts else [],
                convention=self.dialogue_convention,
            )
            for b in chunk.owned_blocks
        }

    def narration_only_attribution(self, chunk: Chunk) -> ChunkAttribution | None:
        """The deterministic result for a chunk the model could not influence, else None.

        When no owned block has a quoted span (under the active dialogue convention) or a
        thought candidate, ``_assemble_segments`` emits DIALOGUE only for quoted spans and
        THOUGHT only for generated candidate ids — so every span degrades to narration
        regardless of what the model would have returned, and the round trip buys nothing.
        Synthesizing with an EMPTY model output through the same assembler guarantees
        byte-equivalence (narration emotions are always None either way). The only loss is
        CharacterMention enrichment, and the prompt asks only for people who SPEAK in the
        slice, so a compliant model returns none on such chunks anyway.
        """
        spans_by_block = self._spans_for(chunk)
        if any(
            span.quoted or span.candidate_id for spans in spans_by_block.values() for span in spans
        ):
            return None
        segments, segment_emotions = self._assemble_segments([], spans_by_block, [])
        return ChunkAttribution(segments=segments, characters=[], segment_emotions=segment_emotions)

    def attribute_chunk(
        self, chunk: Chunk, registry: CharacterRegistry, attempt: int = 0
    ) -> ChunkAttribution:
        """Label one chunk's spans and assemble Segments from the SOURCE text.

        We split each owned block into spans here; the model only labels them. Segment text
        is sliced from the source, so reconstruction cannot be violated by the model.
        ``attempt`` is the 0-based retry index; a retry adds a corrective reminder.
        """
        spans_by_block = self._spans_for(chunk)
        template = _prompt_template(self.prompts_dir, base_prompt_version(self.prompt_version))
        candidates = _render_thought_candidates(spans_by_block) if self.emit_thoughts else "(none)"
        # v5/v6 index multi-quote blocks so the model labels quotes BY INDEX (F1); v3/v4 keep
        # the plain whole-block render. Single-quote blocks stay plain on every version.
        per_quote = base_prompt_version(self.prompt_version) in _PER_QUOTE_VERSIONS
        owned_render = _render_owned_blocks_indexed(chunk, spans_by_block) if per_quote else None
        prompt = render_prompt(template, registry, chunk, candidates, owned_render)
        if attempt > 0:
            if per_quote:
                prompt += (
                    "\n\n## Reminder\n\nFor a block with ⟦Q0⟧/⟦Q1⟧ markers, return a `quotes` "
                    "entry per marker with its `index` and the speaker of THAT quote (and its "
                    "`emotion`); for a block with a single quote, return the block `speaker`. "
                    "Use null for a quote you cannot attribute."
                )
            else:
                prompt += (
                    "\n\n## Reminder\n\nReturn one entry per block with the `block_id` and the "
                    "speaker of that block's dialogue (null if the block is pure narration)."
                )
        raw = self._complete_json(prompt, chunk_label_schema(), attempt)
        if not isinstance(raw, dict):
            raise MalformedOutputError(
                f"{self.provider_id}/{self.model_id} returned a non-object for chunk {chunk.index}"
            )
        segments, segment_emotions = self._assemble_segments(
            raw.get("blocks") or [], spans_by_block, raw.get("thoughts") or []
        )
        # Character mentions are auxiliary metadata; a malformed one must not sink the whole
        # chunk. Drop entries that don't validate (e.g. the model echoed the registry shape).
        characters = []
        for entry in raw.get("characters") or []:
            try:
                characters.append(CharacterMention.model_validate(entry))
            except Exception:
                continue
        return ChunkAttribution(
            segments=segments, characters=characters, segment_emotions=segment_emotions
        )

    def _assemble_segments(
        self,
        raw_blocks: list,
        spans_by_block: dict[str, list[Span]],
        raw_thoughts: list,
    ) -> tuple[list[Segment], list[EmotionVerdict | None]]:
        """Build segments from source spans and a parallel per-segment emotion list.

        Quoted span -> dialogue; a confirmed thought candidate -> thought; prose -> narration.
        The model never counts spans or echoes text (F1 markers live only in the prompt), so
        this can't fail on alignment. Per-quote (F1): when a block emitted ``quotes``, each
        quoted span is labeled by its quoted-span ORDINAL (via the shared
        :func:`quoted_ordinals`); an unlabeled/null/out-of-range quote degrades to narration
        (never a guess-merge) at the model's reported confidence — 0.0 when it gave no usable
        verdict — so unattributed quotes stay visible to the review surfaces. Genuine prose
        narration keeps the default confidence 1.0. When a block emitted no ``quotes``
        (single-quote block or a v3/v4-shaped row) the whole-block ``speaker`` labels every
        quote — byte-identical to
        today. The returned ``emotions`` list is index-aligned to ``segments`` (F2): dialogue
        carries its emotion tag (or None); narration and thought carry None.
        """
        blocks: dict[str, BlockSpeaker] = {}
        for raw_block in raw_blocks:
            try:
                block = BlockSpeaker.model_validate(raw_block)
            except Exception:
                continue  # drop a malformed entry; its block just gets no attributed speaker
            blocks[block.block_id] = block

        # Only verdicts for candidate_ids we deterministically generated are honored, so a
        # stray/hallucinated id can never mint a thought (mirrors the alias adjudicator).
        known_ids = {
            s.candidate_id for spans in spans_by_block.values() for s in spans if s.candidate_id
        }
        verdicts: dict[str, ThoughtVerdict] = {}
        for raw_thought in raw_thoughts:
            try:
                verdict = ThoughtVerdict.model_validate(raw_thought)
            except Exception:
                continue
            if verdict.candidate_id in known_ids:
                verdicts[verdict.candidate_id] = verdict

        segments: list[Segment] = []
        emotions: list[EmotionVerdict | None] = []
        for block_id, spans in spans_by_block.items():
            block = blocks.get(block_id)
            ordinal_of = {span: ordinal for ordinal, span in quoted_ordinals(spans)}
            valid_ordinals = set(ordinal_of.values())
            # Per-quote labels keyed by quoted-span ordinal, dropping any index that isn't an
            # actual quoted span in this block (mirrors the adjudicator's unknown-id discard).
            quote_labels: dict[int, QuoteSpeaker] = {}
            if block is not None:
                for quote in block.quotes:
                    if quote.index in valid_ordinals:
                        quote_labels[quote.index] = quote
            # A block that emitted per-quote labels is in per-quote mode: each quote must be
            # labeled individually and an unlabeled one is narration (no whole-block fallback,
            # or two speakers trading lines would collapse to one).
            per_quote_mode = bool(quote_labels)
            block_speaker = block.speaker if block is not None else None
            block_conf = block.confidence if block is not None else 1.0
            block_emotion = block.emotion if block is not None else None

            for span in spans:
                if not span.text.strip():
                    continue  # whitespace-only span (e.g. a seam) carries no segment
                if span.quoted:
                    if per_quote_mode:
                        quote = quote_labels.get(ordinal_of[span])
                        if quote is not None and quote.speaker:
                            segments.append(
                                Segment(
                                    block_id=block_id,
                                    type=SegmentType.DIALOGUE,
                                    speaker=quote.speaker,
                                    text=span.text,
                                    confidence=quote.confidence,
                                )
                            )
                            emotions.append(quote.emotion)
                            continue
                        # A labeled-but-null quote keeps the model's own confidence; an
                        # unlabeled one has no verdict at all -> 0.0.
                        degrade_conf = quote.confidence if quote is not None else 0.0
                    elif block_speaker:
                        segments.append(
                            Segment(
                                block_id=block_id,
                                type=SegmentType.DIALOGUE,
                                speaker=block_speaker,
                                text=span.text,
                                confidence=block_conf,
                            )
                        )
                        emotions.append(block_emotion)
                        continue
                    elif block is not None and not block.quotes:
                        # Whole-block shape, speaker null: the model's block confidence.
                        degrade_conf = block_conf
                    else:
                        # No row for the block, or per-quote-shaped output whose labels were
                        # all invalid (out-of-range indexes): no usable verdict for this quote.
                        degrade_conf = 0.0
                    # Unattributed / unlabeled quote -> narration (precision over recall),
                    # carrying the degraded confidence — NEVER the Segment default 1.0, which
                    # made these quotes invisible to every review surface.
                    segments.append(
                        Segment(
                            block_id=block_id,
                            type=SegmentType.NARRATION,
                            text=span.text,
                            confidence=degrade_conf,
                        )
                    )
                    emotions.append(None)
                    continue
                verdict = verdicts.get(span.candidate_id) if span.candidate_id else None
                if (
                    verdict is not None
                    and verdict.is_thought
                    and verdict.thinker
                    and verdict.confidence >= self.thought_confidence_floor
                ):
                    segments.append(
                        Segment(
                            block_id=block_id,
                            type=SegmentType.THOUGHT,
                            speaker=verdict.thinker,
                            text=span.text,
                            confidence=verdict.confidence,
                        )
                    )
                    emotions.append(None)  # dialogue-only emotion in v1
                else:
                    # Prose, an unattributed quote, or an unconfirmed candidate: narration.
                    segments.append(
                        Segment(block_id=block_id, type=SegmentType.NARRATION, text=span.text)
                    )
                    emotions.append(None)
        return segments, emotions

    def complete_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        tool_name: str = "emit_structured",
        tool_description: str = "Return the structured result matching the schema.",
    ) -> dict[str, Any]:
        """Public seam for one-off schema-enforced calls outside the attribution template.

        Reuses each backend's schema-enforced JSON path (Ollama structured outputs / the
        Anthropic forced tool) so callers such as the alias adjudicator never reach into the
        protected ``_complete_json``. ``tool_name``/``tool_description`` only matter to the
        Anthropic forced tool (it overrides this to honor them); backends that ignore tool
        metadata fall through to ``_complete_json`` unchanged.
        """
        return self._complete_json(prompt, schema)

    @abstractmethod
    def _complete_json(
        self, prompt: str, schema: dict[str, Any], attempt: int = 0
    ) -> dict[str, Any]:
        """Backend-specific: run the prompt under schema-enforced JSON output, return it."""
