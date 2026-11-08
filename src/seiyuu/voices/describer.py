"""Qwen-first casting: an LLM sketches how ONE character should sound (design doc 2026-11-08).

The model gets the character's evidence (name, what the registry knows, some of their lines,
some narration sentences that mention them) and returns a :class:`VoiceSketch`: gender, an age
band, a voice-only description that becomes the VoiceDesign prompt, the NUMBER of one of the
given lines to use as the sample line, and optionally ``same_as``, another character id this one
is the same person as.

Hallucination can't leak into the audio: the sample line is an index into lines the book
actually contains (an out-of-range number falls back to line 1), ``same_as`` must name another
listed character id (anything else is dropped), and a gender the registry already knows is
never overridden by the caller. Schema-enforced through the shared ``AttributionLLM`` seam,
like the caster and the alias adjudicator; torch-free, no SDK import.
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from seiyuu.attribute.providers.base import AttributionLLM

AGES = ("child", "teen", "young adult", "adult", "older adult", "elderly")
MAX_DESCRIPTION_WORDS = 30
_TOOL = "describe_voice"
_TOOL_DESC = "Return how this one character's voice should sound."


class VoiceSketch(BaseModel):
    gender: Literal["male", "female", "unknown"]
    age: Literal["child", "teen", "young adult", "adult", "older adult", "elderly"]
    voice_description: str = Field(min_length=1, max_length=400)
    sample_line: int
    same_as: str | None = None


@dataclass(frozen=True)
class CharacterEvidence:
    character_id: str
    name: str
    aliases: tuple[str, ...] = ()
    gender: str | None = None
    age_hint: str | None = None
    description: str | None = None
    line_count: int = 0
    lines: tuple[str, ...] = ()  # candidate sample lines, numbered 1.. in the prompt
    mentions: tuple[str, ...] = ()  # narration sentences that name them
    others: tuple[tuple[str, str], ...] = field(default=())  # (id, name) of everyone else


@lru_cache
def describer_template(prompts_dir: Path, version: str) -> str:
    path = Path(prompts_dir) / "describer" / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"describer prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _character_block(ev: CharacterEvidence) -> str:
    known = [f"- id: {ev.character_id}", f"- name: {ev.name}"]
    if ev.aliases:
        known.append(f"- also called: {', '.join(ev.aliases)}")
    if ev.gender:
        known.append(f"- gender: {ev.gender}")
    if ev.age_hint:
        known.append(f"- age: {ev.age_hint}")
    if ev.description:
        known.append(f"- the book says: {ev.description}")
    known.append(f"- spoken lines in the book: {ev.line_count}")
    lines = "\n".join(f"{i}. {line}" for i, line in enumerate(ev.lines, 1)) or "(none)"
    mentions = "\n".join(f"- {m}" for m in ev.mentions) or "(none)"
    return (
        "\n".join(known)
        + f"\n\n### Their lines\n\n{lines}\n\n### Narration that mentions them\n\n{mentions}"
    )


def render_describer_prompt(template: str, ev: CharacterEvidence, *, first_person: bool) -> str:
    pov = (
        'The book is told in the FIRST PERSON: the narrator\'s "I" is one of the characters.'
        if first_person
        else "The book is told in the third person."
    )
    others = "\n".join(f"- {cid}: {name}" for cid, name in ev.others) or "(none)"
    return (
        template.replace("{pov}", pov)
        .replace("{character}", _character_block(ev))
        .replace("{others}", others)
    )


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def clean_sketch(raw: dict, ev: CharacterEvidence) -> VoiceSketch:
    """Validate the model's reply and pin it to the evidence it was given (see module doc)."""
    sketch = VoiceSketch.model_validate(raw)
    words = sketch.voice_description.split()
    description = " ".join(words[:MAX_DESCRIPTION_WORDS]).strip()
    index = sketch.sample_line if 1 <= sketch.sample_line <= len(ev.lines) else 1
    other_ids = {cid for cid, _name in ev.others}
    same_as = sketch.same_as if sketch.same_as in other_ids else None
    return sketch.model_copy(
        update={"voice_description": description, "sample_line": index, "same_as": same_as}
    )


def sketch_character(
    provider: AttributionLLM,
    ev: CharacterEvidence,
    *,
    prompt: str,
) -> VoiceSketch:
    """One schema-enforced call for one character; raises ``ValueError`` on an unusable reply."""
    raw = provider.complete_structured(
        prompt, VoiceSketch.model_json_schema(), tool_name=_TOOL, tool_description=_TOOL_DESC
    )
    try:
        return clean_sketch(raw, ev)
    except ValidationError as exc:
        raise ValueError(f"describer: unusable reply for {ev.character_id!r}: {exc}") from exc
