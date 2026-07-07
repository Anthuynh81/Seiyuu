"""F4: opt-in Layer-2 LLM caster — an ADVISORY voice-trait preference over the Phase-0 caster.

The LLM is asked for ONE thing per character: which voice trait tags (from the closed
:data:`~seiyuu.voices.casting.KNOWN_TRAITS` vocabulary) suit them. That preference is fed to
:func:`~seiyuu.voices.casting.cast_book` as ``trait_hints``, where it only reorders the greedy
tie-breaker among the already-DISTINCT candidates. The LLM NEVER names a preset, never picks the
final voice, and never sees the candidate pool — so a hallucinated, duplicated, or empty
preference can at most change WHICH distinct voice a character gets. Distinctness,
collision-freeness, and determinism stay a structural property of ``cast_book``.

The LLM call goes through the shared :class:`~seiyuu.attribute.providers.base.AttributionLLM`
seam (``complete_structured`` with a schema-enforced tool), never a raw SDK — the same discipline
as attribution and alias adjudication. This module is torch-free and imports no SDK.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from seiyuu.attribute.models import Character
from seiyuu.attribute.providers.base import AttributionLLM
from seiyuu.voices.casting import KNOWN_TRAITS

_CASTER_TOOL = "suggest_voice_traits"
_CASTER_TOOL_DESC = (
    "Return, per character, the voice trait tags that best fit them (a preference only)."
)


class CharacterVoicePreference(BaseModel):
    """One character's advisory trait preference. ``traits`` is filtered to the closed
    vocabulary downstream, so an out-of-vocabulary tag simply drops out."""

    character_id: str
    traits: list[str] = Field(default_factory=list)


class CastingPreferences(BaseModel):
    """The LLM's whole reply: a preference signal, never an assignment."""

    preferences: list[CharacterVoicePreference] = Field(default_factory=list)


@lru_cache
def caster_template(prompts_dir: Path, version: str) -> str:
    path = Path(prompts_dir) / "caster" / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"caster prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_character(char: Character) -> str:
    parts = [f"character_id: {char.id}", f"name: {char.canonical_name!r}"]
    if char.gender:
        parts.append(f"gender={char.gender!r}")
    if char.age_hint:
        parts.append(f"age={char.age_hint!r}")
    if char.description:
        parts.append(f"description={char.description!r}")
    return "- " + ", ".join(parts)


def render_caster_prompt(template: str, characters: list[Character]) -> str:
    rendered = "\n".join(_render_character(c) for c in characters) or "(none)"
    traits = ", ".join(sorted(KNOWN_TRAITS))
    # Literal replacement, not str.format — the prompt contains JSON examples with braces.
    return template.replace("{characters}", rendered).replace("{trait_vocabulary}", traits)


def suggest_trait_hints(
    provider: AttributionLLM,
    characters: list[Character],
    *,
    prompts_dir: Path,
    prompt_version: str = "v1",
) -> dict[str, set[str]]:
    """Ask the provider for per-character trait preferences; return ``{char_id -> trait tags}``.

    Only entries for KNOWN character ids are honored (a stray id is dropped, mirroring the
    adjudicator's unknown-pair discard) and each entry's tags are intersected with
    :data:`KNOWN_TRAITS`. The result is advisory: it is handed to ``cast_book(trait_hints=...)``
    which enforces distinctness deterministically regardless of what this returns.
    """
    if not characters:
        return {}
    template = caster_template(prompts_dir, prompt_version)
    prompt = render_caster_prompt(template, characters)
    raw = provider.complete_structured(
        prompt,
        CastingPreferences.model_json_schema(),
        tool_name=_CASTER_TOOL,
        tool_description=_CASTER_TOOL_DESC,
    )
    if not isinstance(raw, dict):
        return {}  # advisory: a malformed reply just means "no hints", never a hard failure
    try:
        result = CastingPreferences.model_validate(raw)
    except ValidationError:
        return {}  # advisory: degrade to "no hints" rather than sink the whole cast

    known = {c.id for c in characters}
    hints: dict[str, set[str]] = {}
    for pref in result.preferences:
        if pref.character_id not in known:
            continue
        hints[pref.character_id] = {t.strip().lower() for t in pref.traits} & KNOWN_TRAITS
    return hints
