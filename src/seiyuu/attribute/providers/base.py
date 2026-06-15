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
from seiyuu.attribute.models import CharacterRegistry, ChunkAttribution


class AttributionError(Exception):
    """Loud attribution failure; the pipeline adds book/chapter/chunk context."""


@lru_cache
def _prompt_template(prompts_dir: Path, version: str) -> str:
    path = prompts_dir / "attribution" / f"{version}.md"
    if not path.is_file():
        raise AttributionError(f"attribution prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def chunk_attribution_schema() -> dict[str, Any]:
    """JSON schema for one chunk's output — handed to the backend's structured-output mode."""
    return ChunkAttribution.model_json_schema()


def _render_blocks(blocks: list) -> str:
    return "\n\n".join(f"[{b.id}]\n{b.text}" for b in blocks) or "(none)"


def render_prompt(template: str, registry: CharacterRegistry, chunk: Chunk) -> str:
    owned_ids = chunk.owned_ids
    context_blocks = [b for b in chunk.blocks if b.id not in owned_ids]
    registry_json = json.dumps(
        [c.model_dump() for c in registry.characters], indent=2, ensure_ascii=False
    )
    return template.format(
        registry_json=registry_json,
        context_blocks=_render_blocks(context_blocks),
        blocks=_render_blocks(chunk.owned_blocks),
    )


class AttributionLLM(ABC):
    provider_id: str

    def __init__(self, *, model: str, prompts_dir: Path, prompt_version: str = "v1") -> None:
        self.model_id = model
        self.prompts_dir = Path(prompts_dir)
        self.prompt_version = prompt_version

    def attribute_chunk(
        self, chunk: Chunk, registry: CharacterRegistry, attempt: int = 0
    ) -> ChunkAttribution:
        """Attribute one chunk's owned blocks; returns name-based speakers (pre-resolution).

        ``attempt`` is the 0-based retry index. The pipeline retries chunks that fail the
        reconstruction invariant; on a retry we add a corrective reminder and let the
        backend vary its sampling so the next answer differs from the rejected one.
        """
        template = _prompt_template(self.prompts_dir, self.prompt_version)
        prompt = render_prompt(template, registry, chunk)
        if attempt > 0:
            prompt += (
                "\n\n## Reminder\n\nA previous attempt changed the wording. Reproduce every "
                "block's text EXACTLY, character for character; split only where the speaker "
                "changes."
            )
        raw = self._complete_json(prompt, chunk_attribution_schema(), attempt)
        try:
            return ChunkAttribution.model_validate(raw)
        except Exception as exc:
            raise AttributionError(
                f"{self.provider_id}/{self.model_id} returned output failing the segment "
                f"schema for chunk {chunk.index}: {exc}"
            ) from exc

    @abstractmethod
    def _complete_json(
        self, prompt: str, schema: dict[str, Any], attempt: int = 0
    ) -> dict[str, Any]:
        """Backend-specific: run the prompt under schema-enforced JSON output, return it."""
