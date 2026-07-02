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
    Segment,
    SegmentType,
)
from seiyuu.attribute.spans import is_quoted_span, split_block_spans


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


def chunk_label_schema() -> dict[str, Any]:
    """JSON schema for the model's RAW output (per-block span labels + character mentions)."""
    return ChunkLabels.model_json_schema()


def _render_blocks(blocks: list) -> str:
    return "\n\n".join(f"[{b.id}]\n{b.text}" for b in blocks) or "(none)"


def render_prompt(template: str, registry: CharacterRegistry, chunk: Chunk) -> str:
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
    return (
        template.replace("{registry_json}", registry_json)
        .replace("{context_blocks}", _render_blocks(context_blocks))
        .replace("{blocks}", _render_blocks(chunk.owned_blocks))
    )


class AttributionLLM(ABC):
    provider_id: str
    # Local providers hold model weights in VRAM (via Ollama) and must go through the GPU
    # resource manager; cloud providers (anthropic) set False so a network-only attribution
    # run never evicts a resident TTS model or serializes the manager. Default True: an
    # unknown future provider is presumed heavy until it says otherwise.
    uses_gpu: bool = True

    def __init__(self, *, model: str, prompts_dir: Path, prompt_version: str = "v1") -> None:
        self.model_id = model
        self.prompts_dir = Path(prompts_dir)
        self.prompt_version = prompt_version

    def unload(self) -> None:  # noqa: B027 — intentional no-op default; cloud providers override nothing
        """Free GPU memory (GpuConsumer protocol). No-op default; cloud providers need nothing."""

    def attribute_chunk(
        self, chunk: Chunk, registry: CharacterRegistry, attempt: int = 0
    ) -> ChunkAttribution:
        """Label one chunk's spans and assemble Segments from the SOURCE text.

        We split each owned block into spans here; the model only labels them. Segment text
        is sliced from the source, so reconstruction cannot be violated by the model.
        ``attempt`` is the 0-based retry index; a retry adds a corrective reminder.
        """
        spans_by_block = {b.id: split_block_spans(b.text) for b in chunk.owned_blocks}
        template = _prompt_template(self.prompts_dir, self.prompt_version)
        prompt = render_prompt(template, registry, chunk)
        if attempt > 0:
            prompt += (
                "\n\n## Reminder\n\nReturn one entry per block with the `block_id` and the "
                "speaker of that block's dialogue (null if the block is pure narration)."
            )
        raw = self._complete_json(prompt, chunk_label_schema(), attempt)
        if not isinstance(raw, dict):
            raise MalformedOutputError(
                f"{self.provider_id}/{self.model_id} returned a non-object for chunk {chunk.index}"
            )
        segments = self._assemble_segments(raw.get("blocks") or [], spans_by_block)
        # Character mentions are auxiliary metadata; a malformed one must not sink the whole
        # chunk. Drop entries that don't validate (e.g. the model echoed the registry shape).
        characters = []
        for entry in raw.get("characters") or []:
            try:
                characters.append(CharacterMention.model_validate(entry))
            except Exception:
                continue
        return ChunkAttribution(segments=segments, characters=characters)

    def _assemble_segments(
        self, raw_blocks: list, spans_by_block: dict[str, list[str]]
    ) -> list[Segment]:
        """Build segments from source spans: quoted span -> dialogue (model's per-block
        speaker), prose -> narration. The model never counts spans or echoes text, so this
        can't fail on alignment; an un-attributed quote degrades to narration, not an error.
        """
        speakers: dict[str, tuple[str | None, float]] = {}
        for raw_block in raw_blocks:
            try:
                block = BlockSpeaker.model_validate(raw_block)
            except Exception:
                continue  # drop a malformed entry; its block just gets no attributed speaker
            speakers[block.block_id] = (block.speaker, block.confidence)

        segments: list[Segment] = []
        for block_id, spans in spans_by_block.items():
            speaker, confidence = speakers.get(block_id, (None, 1.0))
            for span_text in spans:
                if not span_text.strip():
                    continue  # whitespace-only span (e.g. a seam) carries no segment
                if is_quoted_span(span_text) and speaker:
                    segments.append(
                        Segment(
                            block_id=block_id,
                            type=SegmentType.DIALOGUE,
                            speaker=speaker,
                            text=span_text,
                            confidence=confidence,
                        )
                    )
                else:
                    # Prose, or a quote the model could not attribute: narration.
                    segments.append(
                        Segment(block_id=block_id, type=SegmentType.NARRATION, text=span_text)
                    )
        return segments

    @abstractmethod
    def _complete_json(
        self, prompt: str, schema: dict[str, Any], attempt: int = 0
    ) -> dict[str, Any]:
        """Backend-specific: run the prompt under schema-enforced JSON output, return it."""
