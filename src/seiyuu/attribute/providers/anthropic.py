"""Cloud attribution provider: the Anthropic SDK with schema-enforced tool use.

Forced ``tool_choice`` on a single tool whose ``input_schema`` is the ChunkAttribution
schema makes malformed JSON impossible by construction — the Anthropic analog of Ollama's
structured outputs (SPEC: "Anthropic tool-use schema"). This is a PAID provider: it is
constructed only when the anthropic provider or hybrid escalation is explicitly enabled,
and a missing API key is a clear, actionable error rather than a crash elsewhere.

Default model is ``claude-opus-4-8`` (settings.anthropic_model); a prompt tuned against the
local model transfers up to Claude trivially (SPEC prompt-development workflow).
"""

from typing import Any

from seiyuu.attribute.providers.base import AttributionError, AttributionLLM

_TOOL_NAME = "emit_attribution"
# A chunk is ~2-4k input tokens; its segmented output is comparable. 16k stays well under
# the SDK's non-streaming timeout while leaving headroom for dialogue-dense chapters.
_MAX_TOKENS = 16000


class AnthropicProvider(AttributionLLM):
    provider_id = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        prompts_dir,
        prompt_version: str = "v1",
        api_key: str | None = None,
        max_tokens: int = _MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        super().__init__(model=model, prompts_dir=prompts_dir, prompt_version=prompt_version)
        if client is None and not api_key:
            raise AttributionError(
                "anthropic attribution is enabled but ANTHROPIC_API_KEY is not set; "
                "set it in .env or disable the anthropic provider / hybrid mode"
            )
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client = client  # injectable for tests; built lazily otherwise

    def _get_client(self) -> Any:
        if self._client is None:
            # Lazy import: importing the package must not require the Anthropic SDK.
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def _complete_json(
        self, prompt: str, schema: dict[str, Any], attempt: int = 0
    ) -> dict[str, Any]:
        import anthropic

        try:
            response = self._get_client().messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                tools=[
                    {
                        "name": _TOOL_NAME,
                        "description": "Return the attributed segments and characters.",
                        "input_schema": schema,
                    }
                ],
                # Force the tool so output must match the schema — no free-text JSON.
                tool_choice={"type": "tool", "name": _TOOL_NAME},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError as exc:
            raise AttributionError(
                "Anthropic rejected the API key; check ANTHROPIC_API_KEY"
            ) from exc
        except anthropic.APIError as exc:
            raise AttributionError(
                f"Anthropic API error for model {self.model_id!r}: {exc}"
            ) from exc

        for block in response.content:
            if block.type == "tool_use":
                return block.input
        raise AttributionError(
            f"Anthropic returned no tool_use block (stop_reason={response.stop_reason!r})"
        )
