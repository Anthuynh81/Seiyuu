"""LLM alias adjudicator: the :class:`~seiyuu.attribute.aliases.AliasResolver` implementation.

Wraps an :class:`AttributionLLM` provider plus the per-book adjudication cache. Given the
deterministically-generated :class:`CandidatePair`s, it renders the versioned adjudication
prompt, calls the provider's schema-enforced ``complete_structured``, validates the reply
into an :class:`AdjudicationResult`, and returns one :class:`PairVerdict` per KNOWN pair
(verdicts for unrecognized pair_ids are dropped). It NEVER merges anything — survivor
selection and the ``_conflict`` hard veto live in ``aliases.resolve_registry_aliases``.

Determinism: ``candidates_digest`` is a stable hash over the sorted candidate pairs (each
carrying both characters' canonical name + sorted aliases + gender + age_hint + description).
Because the registry is rebuilt deterministically from the cached chunks, the digest is
identical across no-op reruns, so a cache hit replays the stored verdicts and the LLM fires
only when the candidate set genuinely changes.
"""

import hashlib
import json

from seiyuu.attribute.cache import AdjudicationCacheKey, AttributionCache
from seiyuu.attribute.models import (
    AdjudicationResult,
    CandidatePair,
    CharacterEvidence,
    CharacterRegistry,
    PairVerdict,
)
from seiyuu.attribute.providers.base import (
    AttributionLLM,
    MalformedOutputError,
    adjudication_template,
)

_ADJUDICATION_TOOL = "adjudicate_alias_pairs"
_ADJUDICATION_TOOL_DESC = (
    "Return one verdict per candidate pair: whether the two records are the same character."
)


def _evidence_row(ev: CharacterEvidence) -> list:
    # Stable, order-normalized projection for the digest (aliases sorted).
    return [ev.canonical_name, sorted(ev.aliases), ev.gender, ev.age_hint, ev.description]


def candidates_digest(candidates: list[CandidatePair]) -> str:
    """Deterministic SHA-256 over the candidate set — stable across reruns and input order."""
    rows = [
        [c.pair_id, c.generator, _evidence_row(c.a), _evidence_row(c.b)]
        for c in sorted(candidates, key=lambda p: p.pair_id)
    ]
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _render_evidence(ev: CharacterEvidence) -> str:
    parts = [f"canonical_name={ev.canonical_name!r}"]
    if ev.aliases:
        parts.append(f"aliases={ev.aliases}")
    fields = (("gender", ev.gender), ("age", ev.age_hint), ("description", ev.description))
    for label, value in fields:
        if value:
            parts.append(f"{label}={value!r}")
    return ", ".join(parts)


def render_adjudication_prompt(template: str, candidates: list[CandidatePair]) -> str:
    """Fill the versioned prompt with one bounded block per candidate pair."""
    blocks = []
    for c in candidates:
        blocks.append(
            f"- pair_id: {c.pair_id}\n  A: {_render_evidence(c.a)}\n  B: {_render_evidence(c.b)}"
        )
    rendered = "\n".join(blocks) or "(none)"
    # Literal replacement, not str.format — the prompt contains JSON examples with braces.
    return template.replace("{candidate_pairs}", rendered)


class LLMAdjudicator:
    """AliasResolver backed by an LLM provider + the per-book adjudication cache."""

    def __init__(
        self,
        provider: AttributionLLM,
        *,
        cache: AttributionCache,
        book_id: str,
        prompt_version: str,
        prompts_dir,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._book_id = book_id
        self._prompt_version = prompt_version
        self._prompts_dir = prompts_dir

    @property
    def provider(self) -> AttributionLLM:
        """The underlying LLM provider (the standalone command acquires the GPU on it)."""
        return self._provider

    @property
    def uses_gpu(self) -> bool:
        return getattr(self._provider, "uses_gpu", True)

    def resolve(
        self, candidates: list[CandidatePair], registry: CharacterRegistry
    ) -> list[PairVerdict]:
        if not candidates:
            return []
        key = AdjudicationCacheKey(
            book_id=self._book_id,
            provider_id=self._provider.provider_id,
            model_id=self._provider.model_id,
            adjudication_prompt_version=self._prompt_version,
            candidates_digest=candidates_digest(candidates),
        )
        cached = self._cache.get_adjudication(key)
        if cached is not None:
            return cached

        template = adjudication_template(self._prompts_dir, self._prompt_version)
        prompt = render_adjudication_prompt(template, candidates)
        raw = self._provider.complete_structured(
            prompt,
            AdjudicationResult.model_json_schema(),
            tool_name=_ADJUDICATION_TOOL,
            tool_description=_ADJUDICATION_TOOL_DESC,
        )
        if not isinstance(raw, dict):
            raise MalformedOutputError(
                f"{self._provider.provider_id}/{self._provider.model_id} returned a non-object "
                f"for alias adjudication"
            )
        result = AdjudicationResult.model_validate(raw)
        known = {c.pair_id for c in candidates}
        verdicts = [v for v in result.verdicts if v.pair_id in known]
        self._cache.put_adjudication(key, verdicts)
        return verdicts
