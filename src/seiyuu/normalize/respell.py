"""F3: opt-in LLM grapheme-respelling suggester — an ADVISORY enrichment of the lexicon.

The deterministic hard-name surfacer (:func:`seiyuu.normalize.lexicon.suggest_terms`) stays the
free default: it says WHICH words are likely hard. This module is the opt-in second layer that,
on an explicit user action, asks the LLM to propose a grapheme RESPELLING for those terms
(schema-enforced: ``term -> respelling`` plus an optional ``note``). The output is advisory only —
the user accepts a suggestion into ``books/{id}/lexicon.json``, which stays the deterministic
source of truth. The LLM never writes the lexicon and never touches ``normalize_text``.

The call goes through the shared :class:`~seiyuu.attribute.providers.base.AttributionLLM` seam
(``complete_structured`` with a schema-enforced tool), never a raw SDK — the same discipline as
attribution and alias adjudication. Suggestions for terms that were not requested are dropped
(mirrors the adjudicator's unknown-id discard), so the model can never inject an unrelated word.
"""

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from seiyuu.attribute.providers.base import AttributionLLM

_RESPELL_TOOL = "suggest_respellings"
_RESPELL_TOOL_DESC = "Return a spoken grapheme respelling for each hard-to-pronounce term."


class RespellSuggestion(BaseModel):
    """One advisory respelling. Non-empty ``term``/``respelling`` is enforced by the filter in
    :func:`suggest_respellings`, not here, so a single blank entry can't sink the whole reply."""

    term: str
    respelling: str
    note: str | None = None


class RespellSuggestions(BaseModel):
    suggestions: list[RespellSuggestion] = Field(default_factory=list)


@lru_cache
def respell_template(prompts_dir: Path, version: str) -> str:
    path = Path(prompts_dir) / "respell" / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"respell prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render_respell_prompt(template: str, terms: Iterable[str]) -> str:
    rendered = "\n".join(f"- {t.strip()}" for t in terms if t.strip()) or "(none)"
    # Literal replacement, not str.format — the prompt contains JSON examples with braces.
    return template.replace("{terms}", rendered)


def suggest_respellings(
    provider: AttributionLLM,
    terms: list[str],
    *,
    prompts_dir: Path,
    prompt_version: str = "v1",
) -> list[RespellSuggestion]:
    """Ask the provider for a respelling of each requested term; return the cleaned suggestions.

    Advisory and defensive: entries are dropped when the term/respelling is blank or the term was
    not among ``terms`` (so a hallucinated word never appears). Terms echo back in the requested
    spelling; duplicates collapse to the first suggestion. Never writes anything.
    """
    requested = {t.strip().casefold(): t.strip() for t in terms if t.strip()}
    if not requested:
        return []
    template = respell_template(prompts_dir, prompt_version)
    prompt = render_respell_prompt(template, requested.values())
    raw = provider.complete_structured(
        prompt,
        RespellSuggestions.model_json_schema(),
        tool_name=_RESPELL_TOOL,
        tool_description=_RESPELL_TOOL_DESC,
    )
    if not isinstance(raw, dict):
        return []
    try:
        result = RespellSuggestions.model_validate(raw)
    except ValidationError:
        return []
    out: list[RespellSuggestion] = []
    seen: set[str] = set()
    for s in result.suggestions:
        term = s.term.strip()
        respelling = s.respelling.strip()
        key = term.casefold()
        if not term or not respelling or key not in requested or key in seen:
            continue
        seen.add(key)
        note = s.note.strip() if s.note else None
        out.append(RespellSuggestion(term=requested[key], respelling=respelling, note=note or None))
    return out
