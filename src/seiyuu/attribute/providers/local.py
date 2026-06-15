"""Local attribution provider: the OpenAI SDK pointed at Ollama's OpenAI-compatible API.

Uses Ollama structured outputs (``response_format`` json_schema) so output is schema-valid
by construction. Sets ``keep_alive: 0`` so Ollama unloads the model from VRAM right after
the call — the render stage must be able to load a TTS engine without contending for the
single GPU (SPEC GPU discipline). Ollama being down is a clear, actionable error.
"""

import json
from typing import Any

from seiyuu.attribute.providers.base import AttributionError, AttributionLLM


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

    def _complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        from openai import APIConnectionError

        try:
            response = self._get_client().chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
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

        content = response.choices[0].message.content
        if not content:
            raise AttributionError(f"Ollama returned empty content for model {self.model_id!r}")
        return json.loads(content)
