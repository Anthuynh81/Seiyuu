"""In-memory AttributionLLM for tests: returns scripted ChunkAttributions, tracks calls.

Lets pipeline tests drive honest, paraphrasing, and dropping providers without a live LLM.
"""

from collections.abc import Callable

from seiyuu.attribute.chunking import Chunk
from seiyuu.attribute.models import CharacterRegistry, ChunkAttribution
from seiyuu.attribute.providers.base import AttributionLLM


class FakeProvider(AttributionLLM):
    provider_id = "fake"

    def __init__(
        self,
        script: Callable[[Chunk, CharacterRegistry, int], ChunkAttribution],
        *,
        model: str = "fake-1.0",
    ) -> None:
        # Bypass the SDK template; we script attribute_chunk directly.
        self.model_id = model
        self.prompt_version = "v1"
        self.calls: list[tuple[int, int]] = []  # (chunk index, attempt)
        self._script = script

    def attribute_chunk(
        self, chunk: Chunk, registry: CharacterRegistry, attempt: int = 0
    ) -> ChunkAttribution:
        self.calls.append((chunk.index, attempt))
        return self._script(chunk, registry, attempt)

    def _complete_json(self, prompt, schema, attempt=0):  # pragma: no cover - never called
        raise NotImplementedError
