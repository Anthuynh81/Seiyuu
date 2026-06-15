"""Attribution providers: pipeline code gets a provider ONLY via get_provider()."""

import importlib
from pathlib import Path

from seiyuu.attribute.providers.base import (
    AttributionError,
    AttributionLLM,
    chunk_attribution_schema,
)

# Provider classes are referenced as strings and imported lazily so that importing
# seiyuu.attribute.providers never pulls in an LLM SDK (openai / anthropic).
_PROVIDERS = {
    "local": "seiyuu.attribute.providers.local:OllamaProvider",
    "anthropic": "seiyuu.attribute.providers.anthropic:AnthropicProvider",
}


def get_provider(provider_id: str, *, model: str, prompts_dir: Path, **kwargs) -> AttributionLLM:
    if provider_id not in _PROVIDERS:
        raise ValueError(
            f"unknown attribution provider {provider_id!r}; available: {sorted(_PROVIDERS)}"
        )
    module_name, class_name = _PROVIDERS[provider_id].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(model=model, prompts_dir=prompts_dir, **kwargs)


__all__ = [
    "AttributionError",
    "AttributionLLM",
    "chunk_attribution_schema",
    "get_provider",
]
