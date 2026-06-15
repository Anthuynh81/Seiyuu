"""Local attribution provider: the OpenAI SDK pointed at Ollama's OpenAI-compatible API.

Uses Ollama structured outputs (``response_format`` json_schema) so output is schema-valid
by construction. Sets ``keep_alive: 0`` so Ollama unloads the model from VRAM right after
the call — the render stage must be able to load a TTS engine without contending for the
single GPU (SPEC GPU discipline). Ollama being down is a clear, actionable error.
"""

import json
import re
from typing import Any

from seiyuu.attribute.providers.base import AttributionError, AttributionLLM

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON in a ```json fence despite structured-output mode."""
    match = _FENCE.match(text.strip())
    return match.group(1) if match else text


class OllamaProvider(AttributionLLM):
    provider_id = "local"

    def __init__(
        self,
        *,
        model: str,
        prompts_dir,
        prompt_version: str = "v1",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        super().__init__(model=model, prompts_dir=prompts_dir, prompt_version=prompt_version)
        self.base_url = base_url
        self.temperature = temperature
        self._client = client  # injectable for tests; built lazily otherwise

    def _get_client(self) -> Any:
        if self._client is None:
            # Lazy import: importing the package must not require the OpenAI SDK.
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key="ollama")
        return self._client

    def _complete_json(
        self, prompt: str, schema: dict[str, Any], attempt: int = 0
    ) -> dict[str, Any]:
        from openai import APIConnectionError

        # Retries need variation or they just reproduce the rejected answer; nudge
        # temperature up per attempt while keeping the first pass deterministic.
        temperature = min(0.8, self.temperature + 0.2 * attempt)
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "attribution", "schema": schema},
                },
                # Ollama extension: free VRAM immediately so a TTS engine can load.
                extra_body={"keep_alive": 0},
            )
        except APIConnectionError as exc:
            raise AttributionError(
                f"cannot reach Ollama at {self.base_url}: is `ollama serve` running and is "
                f"model {self.model_id!r} pulled (`ollama pull {self.model_id}`)?"
            ) from exc

        choice = response.choices[0]
        # `length` means the model ran out of context before closing the JSON. With a
        # reasoning model (e.g. Qwen3), thinking tokens can consume the whole window and
        # leave empty content. The OpenAI-compatible endpoint cannot raise num_ctx or
        # disable thinking per-request — both are server-side (OLLAMA_CONTEXT_LENGTH, or a
        # non-thinking model). Fail with that guidance instead of an opaque parse error.
        if choice.finish_reason == "length":
            raise AttributionError(
                f"Ollama truncated output for model {self.model_id!r} (hit the context "
                f"window). Raise the server context (OLLAMA_CONTEXT_LENGTH), use a "
                f"non-thinking model, or lower attribution_chunk_tokens."
            )
        content = (choice.message.content or "").strip()
        if not content:
            raise AttributionError(f"Ollama returned empty content for model {self.model_id!r}")
        return json.loads(_strip_code_fence(content))
