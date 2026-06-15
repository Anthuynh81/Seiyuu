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
        script: Callable[[Chunk, CharacterRegistry], ChunkAttribution],
        *,
        model: str = "fake-1.0",
    ) -> None:
        # Bypass the SDK template; we script attribute_chunk directly.
        self.model_id = model
        self.prompt_version = "v1"
        self.calls: list[int] = []
        self._script = script

    def attribute_chunk(self, chunk: Chunk, registry: CharacterRegistry) -> ChunkAttribution:
        self.calls.append(chunk.index)
        return self._script(chunk, registry)

    def _complete_json(self, prompt, schema):  # pragma: no cover - never called
        raise NotImplementedError
