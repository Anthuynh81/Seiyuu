"""Service layer for the opt-in LLM advisory layers (F3 respell suggester, F4 Layer-2 caster).

Both features run an LLM ONLY on an explicit user action (never an automatic pipeline path) and
reuse the attribution provider seam via :func:`~seiyuu.services.attribution.build_provider`. Two
shared concerns live here so the CLI and the API stay thin:

- **GPU discipline.** A local (Ollama) provider is a heavy GPU consumer like any TTS model, so
  the call acquires the GPU through the resource manager and frees it in ``finally`` — exactly
  like ``run_adjudication``. A cloud (anthropic) provider is network-only (``uses_gpu`` False)
  and never touches the card.
- **Paid resolution.** :func:`resolve_advisory` reports the effective provider + model + paid-ness
  so each boundary can apply the confirm-paid gate before this service ever builds a paid client.
  The gate itself stays at the boundary (ApiError vs ClickException); this module only runs after
  it has passed.
"""

from contextlib import nullcontext
from typing import NamedTuple

from seiyuu.attribute.models import Character
from seiyuu.gpu import get_gpu_manager
from seiyuu.normalize.respell import RespellSuggestion, suggest_respellings
from seiyuu.services.attribution import build_provider
from seiyuu.voices.llm_caster import suggest_trait_hints


class ResolvedAdvisory(NamedTuple):
    provider_id: str
    model: str
    is_paid: bool  # anthropic -> the caller must have confirm_paid + the key


def _default_model(cfg, provider_id: str) -> str:
    """Per-provider default model, mirroring services.attribution._adjudication_model."""
    return cfg.anthropic_model if provider_id == "anthropic" else cfg.attribution_model


def resolve_advisory(cfg, configured_provider: str, configured_model, override) -> ResolvedAdvisory:
    """Resolve the effective (provider, model, paid) for an advisory call.

    ``configured_provider``/``configured_model`` are the feature's cfg defaults; ``override`` is an
    optional per-request provider id. Only ``anthropic`` is paid — the gate keys off ``is_paid``.
    """
    provider_id = (override or configured_provider or "").strip() or configured_provider
    model = configured_model or _default_model(cfg, provider_id)
    return ResolvedAdvisory(provider_id, model, provider_id == "anthropic")


def _run(
    cfg, resolved: ResolvedAdvisory, prompt_version: str, task: str, fn, gpu=None, provider=None
):
    """Build the provider (unless injected), acquire the GPU for a local provider, run ``fn``.

    ``fn`` receives the built provider. The GPU is held for the whole call and freed in
    ``finally`` (via the manager, keeping residency truthful) so a canceled call never leaves
    Ollama resident before a TTS engine loads.
    """
    provider = provider or build_provider(cfg, resolved.provider_id, resolved.model, prompt_version)
    gpu = gpu or get_gpu_manager()
    uses_gpu = getattr(provider, "uses_gpu", True)
    ctx = (
        gpu.acquire(provider, f"llm:{task}:{provider.provider_id}:{provider.model_id}")
        if uses_gpu
        else nullcontext()
    )
    try:
        with ctx:
            return fn(provider)
    finally:
        if uses_gpu:
            gpu.free_all()


def run_respell_suggestions(
    cfg,
    resolved: ResolvedAdvisory,
    terms: list[str],
    *,
    gpu=None,
    provider=None,
) -> list[RespellSuggestion]:
    """F3: LLM respelling proposals for ``terms``. Advisory — the caller persists nothing here."""
    return _run(
        cfg,
        resolved,
        cfg.respell_prompt_version,
        "respell",
        lambda p: suggest_respellings(
            p, terms, prompts_dir=cfg.prompts_dir, prompt_version=cfg.respell_prompt_version
        ),
        gpu=gpu,
        provider=provider,
    )


def run_cast_hints(
    cfg,
    resolved: ResolvedAdvisory,
    characters: list[Character],
    *,
    gpu=None,
    provider=None,
) -> dict[str, set[str]]:
    """F4: LLM per-character voice-trait preference. Feeds ``cast_book(trait_hints=...)`` as a
    tie-breaker only — the assignment itself stays deterministic and collision-free."""
    return _run(
        cfg,
        resolved,
        cfg.cast_prompt_version,
        "cast",
        lambda p: suggest_trait_hints(
            p, characters, prompts_dir=cfg.prompts_dir, prompt_version=cfg.cast_prompt_version
        ),
        gpu=gpu,
        provider=provider,
    )
