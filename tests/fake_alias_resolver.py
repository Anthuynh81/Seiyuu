"""Scripted test doubles for the opt-in alias adjudication — no live LLM.

``FakeAliasResolver`` is a scripted :class:`~seiyuu.attribute.aliases.AliasResolver`: it
returns whatever verdicts a decision callable dictates for the pre-generated candidate pairs,
so the adversarial over-merge suite can APPROVE a pair at high confidence and assert the
guards still refuse to merge it.

``ScriptedAdjudicatorProvider`` is a minimal LLM-provider stand-in exposing only
``complete_structured``; the cache/determinism test wraps it in a real ``LLMAdjudicator`` and
counts calls to prove the per-book cache fires the "LLM" exactly once across reruns.
"""

import re
from collections.abc import Callable

from seiyuu.attribute.models import CandidatePair, CharacterRegistry, PairVerdict

# A decision maps a candidate to (same_person, confidence), or None to omit a verdict entirely.
Decision = Callable[[CandidatePair], tuple[bool, float] | None]


class FakeAliasResolver:
    def __init__(self, decide: Decision) -> None:
        self._decide = decide
        self.calls: list[list[str]] = []  # the pair_ids seen on each resolve() call

    def resolve(
        self, candidates: list[CandidatePair], registry: CharacterRegistry
    ) -> list[PairVerdict]:
        self.calls.append([c.pair_id for c in candidates])
        verdicts: list[PairVerdict] = []
        for cand in candidates:
            decision = self._decide(cand)
            if decision is None:
                continue
            same, confidence = decision
            verdicts.append(
                PairVerdict(pair_id=cand.pair_id, same_person=same, confidence=confidence)
            )
        return verdicts


def approve_all(confidence: float = 1.0) -> Decision:
    """A decision callable that APPROVES every candidate at ``confidence``."""
    return lambda _cand: (True, confidence)


class ScriptedAdjudicatorProvider:
    """Minimal LLM provider: approves every pair named in the prompt, counting each call."""

    provider_id = "fake-adj"
    uses_gpu = False

    def __init__(
        self, *, model: str = "fake-adj-1.0", same_person: bool = True, confidence: float = 1.0
    ) -> None:
        self.model_id = model
        self.calls = 0
        self._same = same_person
        self._confidence = confidence

    def complete_structured(
        self, prompt: str, schema: dict, *, tool_name: str = "", tool_description: str = ""
    ) -> dict:
        self.calls += 1
        pair_ids = re.findall(r"pair_id:\s*(\S+)", prompt)
        return {
            "verdicts": [
                {"pair_id": pid, "same_person": self._same, "confidence": self._confidence}
                for pid in pair_ids
            ]
        }
