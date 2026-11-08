"""Qwen-first casting, part 1: the cast plan (design/qwen-first-casting-2026-11-08.md).

For every character with lines, the local LLM drafts how they should sound: a voice-only
description, an age band, a sample line chosen from their own lines, and possibly "same person
as" another character. The user then edits it in the Review table. The plan lives in
``output/{book}/cast_plan.json``. It is a working file for that table and for the design-cast
job (part 2), never read by render, and in no cache key.

Rules that keep it honest:

- the narrator is its own row (id ``__narrator__``) with an editable default description; for a
  first-person book the plan also suggests the protagonist, whose voice the narrator can share;
- a row the user edited survives a re-draft untouched; design ids and voice ids always carry over;
- "same voice as" links are flattened to one hop and never form a cycle;
- characters with fewer than ``SMALL_PART_LINES`` lines get 1 take, everyone else 2.
"""

import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from seiyuu.attribute.cache import AttributionCache, SketchCacheKey
from seiyuu.attribute.genders import name_patterns, narration_texts, sentences
from seiyuu.attribute.models import AttributionReport, SegmentType
from seiyuu.repository import atomic_write_text
from seiyuu.services.common import ServiceError
from seiyuu.voices.describer import (
    AGES,
    CharacterEvidence,
    VoiceSketch,
    clean_sketch,
    describer_template,
    prompt_digest,
    render_describer_prompt,
    sketch_character,
)

CAST_PLAN_NAME = "cast_plan.json"
NARRATOR_ROW = "__narrator__"
SMALL_PART_LINES = 5
TAKES_FULL = 2
TAKES_SMALL = 1
MAX_LINES = 8
MAX_MENTIONS = 6
MAX_MENTION_CHARS = 300
DEFAULT_NARRATOR_DESCRIPTION = (
    "A warm, clear audiobook narrator's voice: steady and unhurried, natural and expressive "
    "without acting out the characters."
)
GENERIC_DESCRIPTION = "A natural, clear adult voice with a calm, conversational tone."

_FIRST = re.compile(r"\b(?:I|I'm|I’m|I'd|I’d|I've|I’ve|I'll|I’ll|me|my|myself)\b")
_THIRD = re.compile(r"\b(?:he|she|him|her|his|hers)\b", re.IGNORECASE)


class PlanRow(BaseModel):
    character_id: str
    name: str
    line_count: int = 0
    gender: str | None = None
    age: str | None = None
    description: str
    sample_lines: list[str] = []  # candidates from the book (their lines; narration for narrator)
    sample_index: int = 0  # 0-based into sample_lines
    same_as: str | None = None  # another row's character_id: share its voice, design nothing
    keep_current: bool = False  # skip designing; keep whatever voice is assigned today
    takes: int = Field(default=TAKES_FULL, ge=1, le=5)
    edited: bool = False  # the user changed it: a re-draft leaves it alone
    drafted_by: str | None = None  # "llm" | "default" | None (fallback after a bad reply)
    design_id: str | None = None  # set by the design-cast job (part 2)
    voice_id: str | None = None  # the voice it designed or kept


class CastPlan(BaseModel):
    book_id: str
    provider_id: str
    model_id: str
    prompt_version: str
    first_person: bool = False
    protagonist_suggestion: str | None = None  # row id whose voice could narrate (first person)
    narrator: PlanRow
    rows: list[PlanRow] = []  # busiest first
    notes: list[str] = []


class RowUpdate(BaseModel):
    """One user edit in the Review table. Omitted fields stay as they are."""

    model_config = {"extra": "forbid"}
    character_id: str
    description: str | None = Field(default=None, min_length=3, max_length=400)
    sample_index: int | None = Field(default=None, ge=0)
    same_as: str | None = None
    clear_same_as: bool = False
    keep_current: bool | None = None
    gender: str | None = None
    age: str | None = None


# -- evidence ------------------------------------------------------------------------------------


def _spread(items: list[str], k: int) -> list[str]:
    """Up to ``k`` items evenly spread over the list (deterministic), so lines span the book."""
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _character_lines(lines: list[str]) -> list[str]:
    """Up to MAX_LINES of a character's lines: mid-length ones spread over the book first (the
    best sample candidates), then short ones to fill, so a small part still has enough words
    to design a voice from (design-cast joins lines up to a usable length)."""
    unique = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    good = _spread([line for line in unique if 8 <= len(line.split()) <= 40], MAX_LINES)
    rest = _spread([line for line in unique if line not in good], MAX_LINES - len(good))
    return good + rest


def _prefer_mid_length(lines: list[str], lo: int = 8, hi: int = 40) -> list[str]:
    unique = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    good = [line for line in unique if lo <= len(line.split()) <= hi]
    return good or unique


def is_first_person(report: AttributionReport) -> bool:
    text = " ".join(narration_texts(report.chapters))
    words = max(len(text.split()), 1)
    first, third = len(_FIRST.findall(text)), len(_THIRD.findall(text))
    return first / words * 1000 >= 15 and first >= third


def gather_evidence(report: AttributionReport) -> list[CharacterEvidence]:
    """Evidence for every registry character with at least one line, busiest first."""
    lines: dict[str, list[str]] = {}
    for chapter in report.chapters:
        for seg in chapter.segments:
            if seg.speaker and seg.type in (SegmentType.DIALOGUE, SegmentType.THOUGHT):
                lines.setdefault(seg.speaker, []).append(seg.text)
    narration = narration_texts(report.chapters)
    patterns = name_patterns(report.registry, narration)
    mentions: dict[str, list[str]] = {cid: [] for cid in patterns}
    for text in narration:
        for sentence in sentences(text):
            for cid, pattern in patterns.items():
                if pattern.search(sentence):
                    mentions[cid].append(sentence.strip()[:MAX_MENTION_CHARS])

    speaking = [c for c in report.registry.characters if lines.get(c.id)]
    speaking.sort(key=lambda c: (-len(lines[c.id]), c.id))
    everyone = tuple((c.id, c.canonical_name) for c in speaking)
    return [
        CharacterEvidence(
            character_id=c.id,
            name=c.canonical_name,
            aliases=tuple(c.aliases),
            gender=c.gender,
            age_hint=c.age_hint,
            description=c.description,
            line_count=len(lines[c.id]),
            lines=tuple(_character_lines(lines[c.id])),
            mentions=tuple(_spread(mentions.get(c.id, []), MAX_MENTIONS)),
            others=tuple(o for o in everyone if o[0] != c.id),
        )
        for c in speaking
    ]


# Plain storytelling prose: starts with a capital letter, ends like a sentence, and carries
# none of the marks front matter, notices and system messages use ("※ Note : ...", "[Rule 3]").
_PROSE = re.compile(r"^[A-Z][^※\[\]{}<>#*|_:=~•]*[.!?…][\"'”’]?$")


def narrator_samples(report: AttributionReport) -> list[str]:
    candidates = [s.strip() for t in narration_texts(report.chapters) for s in sentences(t)]
    prose = [s for s in candidates if _PROSE.match(s)]
    return _spread(_prefer_mid_length(prose or candidates, 12, 40), MAX_LINES)


# -- drafting ------------------------------------------------------------------------------------


def _row_from(ev: CharacterEvidence, sketch: VoiceSketch | None) -> PlanRow:
    gender = ev.gender or (sketch.gender if sketch and sketch.gender != "unknown" else None)
    return PlanRow(
        character_id=ev.character_id,
        # the registry can hold a CHARACTER called "Narrator" (quoted lines attributed to the
        # narrator); label it apart from the narrator row (the prompt keeps the real name)
        name=ev.name if ev.name.casefold() != "narrator" else "Narrator (as a character)",
        line_count=ev.line_count,
        gender=gender,
        age=sketch.age if sketch else None,
        description=sketch.voice_description if sketch else (ev.description or GENERIC_DESCRIPTION),
        sample_lines=list(ev.lines),
        sample_index=(sketch.sample_line - 1) if sketch else 0,
        same_as=sketch.same_as if sketch else None,
        takes=TAKES_SMALL if ev.line_count < SMALL_PART_LINES else TAKES_FULL,
        drafted_by="llm" if sketch else None,
    )


def flatten_same_as(rows: list[PlanRow], narrator: PlanRow | None = None) -> None:
    """Make every ``same_as`` point at a row that designs its own voice: follow chains to their
    end, and drop a link that points at itself, at an unknown row, or round a cycle."""
    by_id = {r.character_id: r for r in rows}

    def target(start: PlanRow) -> str | None:
        seen = {start.character_id}
        current = start.same_as
        while current is not None:
            if current in seen or current not in by_id:
                return None
            seen.add(current)
            nxt = by_id[current].same_as
            if nxt is None:
                return current
            current = nxt
        return None

    resolved = {r.character_id: target(r) for r in rows}
    for r in rows:
        r.same_as = resolved[r.character_id]
    if narrator is not None and narrator.same_as is not None:
        narrator.same_as = (
            narrator.same_as
            if narrator.same_as in by_id and by_id[narrator.same_as].same_as is None
            else resolved.get(narrator.same_as)
        )


def draft_plan(
    report: AttributionReport,
    *,
    sketch: Callable[[CharacterEvidence], VoiceSketch | None],
    provider_id: str,
    model_id: str,
    prompt_version: str,
    existing: "CastPlan | None" = None,
    progress: Callable[[str], None] = lambda _m: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> CastPlan:
    """Build the plan; ``sketch`` returns one character's VoiceSketch (or None for a bad reply)."""
    evidence = gather_evidence(report)
    old = {r.character_id: r for r in existing.rows} if existing else {}
    rows: list[PlanRow] = []
    notes: list[str] = []
    for i, ev in enumerate(evidence, 1):
        check_cancel()
        prior = old.get(ev.character_id)
        if prior is not None and prior.edited:
            rows.append(prior.model_copy(update={"line_count": ev.line_count}))
            continue
        progress(f"describing {ev.name} ({i} of {len(evidence)})")
        drafted = sketch(ev)
        if drafted is None:
            notes.append(f"couldn't describe {ev.name!r}: the model's reply was unusable")
        row = _row_from(ev, drafted)
        if prior is not None and (
            prior.description,
            prior.sample_lines,
            prior.gender,
            prior.age,
        ) == (
            row.description,
            row.sample_lines,
            row.gender,
            row.age,
        ):
            # same voice brief as before: the voice designed from it still fits
            row.design_id, row.voice_id = prior.design_id, prior.voice_id
        rows.append(row)

    first_person = is_first_person(report)
    if existing is not None and existing.narrator.edited:
        narrator = existing.narrator
    else:
        narrator = PlanRow(
            character_id=NARRATOR_ROW,
            name="Narrator",
            description=DEFAULT_NARRATOR_DESCRIPTION,
            sample_lines=narrator_samples(report),
            drafted_by="default",
        )
        if existing is not None:
            narrator.design_id, narrator.voice_id = (
                existing.narrator.design_id,
                existing.narrator.voice_id,
            )
    flatten_same_as(rows, narrator)
    protagonist = None
    if first_person and rows:
        top = rows[0]
        protagonist = top.same_as or top.character_id
    return CastPlan(
        book_id=report.book_id,
        provider_id=provider_id,
        model_id=model_id,
        prompt_version=prompt_version,
        first_person=first_person,
        protagonist_suggestion=protagonist,
        narrator=narrator,
        rows=rows,
        notes=notes,
    )


def cached_sketcher(
    provider,
    cache: AttributionCache,
    *,
    book_id: str,
    prompts_dir: Path,
    prompt_version: str,
    first_person: bool,
) -> Callable[[CharacterEvidence], VoiceSketch | None]:
    """A ``sketch`` for :func:`draft_plan`: one cached, schema-enforced call per character."""
    template = describer_template(Path(prompts_dir), prompt_version)

    def sketch(ev: CharacterEvidence) -> VoiceSketch | None:
        prompt = render_describer_prompt(template, ev, first_person=first_person)
        key = SketchCacheKey(
            book_id=book_id,
            provider_id=provider.provider_id,
            model_id=provider.model_id,
            prompt_version=prompt_version,
            prompt_digest=prompt_digest(prompt),
        )
        hit = cache.get_sketch(key)
        if hit is not None:
            return clean_sketch(hit, ev)
        try:
            result = sketch_character(provider, ev, prompt=prompt)
        except ValueError:
            return None
        cache.put_sketch(key, result.model_dump())
        return result

    return sketch


# -- edits + persistence -------------------------------------------------------------------------


def apply_updates(plan: CastPlan, updates: list[RowUpdate]) -> CastPlan:
    """Apply Review-table edits; every touched row is marked edited (a re-draft keeps it)."""
    plan = plan.model_copy(deep=True)
    rows = {r.character_id: r for r in plan.rows}
    rows[NARRATOR_ROW] = plan.narrator
    for upd in updates:
        row = rows.get(upd.character_id)
        if row is None:
            raise ServiceError(f"cast plan: unknown character {upd.character_id!r}")
        if upd.description is not None:
            row.description = " ".join(upd.description.split())
        if upd.sample_index is not None:
            if upd.sample_index >= max(len(row.sample_lines), 1):
                raise ServiceError(f"cast plan: {row.name!r} has no sample line {upd.sample_index}")
            row.sample_index = upd.sample_index
        if upd.clear_same_as:
            row.same_as = None
        elif upd.same_as is not None:
            if upd.same_as == row.character_id or upd.same_as not in rows:
                raise ServiceError(f"cast plan: can't share a voice with {upd.same_as!r}")
            if upd.same_as == NARRATOR_ROW:
                raise ServiceError("cast plan: a character can't share the narrator's voice")
            row.same_as = upd.same_as
        if upd.keep_current is not None:
            row.keep_current = upd.keep_current
        if upd.gender is not None:
            if upd.gender not in {"male", "female", "unknown"}:
                raise ServiceError("cast plan: gender must be male, female or unknown")
            row.gender = None if upd.gender == "unknown" else upd.gender
        if upd.age is not None:
            if upd.age not in AGES:
                raise ServiceError(f"cast plan: age must be one of {', '.join(AGES)}")
            row.age = upd.age
        reshaped = any(
            v is not None for v in (upd.description, upd.sample_index, upd.gender, upd.age)
        )
        if reshaped:
            # the designed voice no longer matches the row: the next design-cast run redesigns
            # it (the old voice stays assigned and in the library until then)
            row.design_id = None
            row.voice_id = None
        row.edited = True
    flatten_same_as(plan.rows, plan.narrator)
    return plan


def plan_path(output_dir: Path, book_id: str) -> Path:
    return Path(output_dir) / book_id / CAST_PLAN_NAME


def load_plan(output_dir: Path, book_id: str) -> CastPlan | None:
    path = plan_path(output_dir, book_id)
    if not path.is_file():
        return None
    return CastPlan.model_validate_json(path.read_text(encoding="utf-8"))


def save_plan(output_dir: Path, plan: CastPlan) -> Path:
    path = plan_path(output_dir, plan.book_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(path, plan.model_dump_json(indent=2))


def run_describe_cast(
    cfg,
    book_id: str,
    *,
    progress: Callable[[str], None] = lambda _m: None,
    check_cancel: Callable[[], None] = lambda: None,
    provider=None,
    gpu=None,
) -> CastPlan:
    """Draft (or re-draft) the book's cast plan with the describer LLM and save it.

    Reads the EFFECTIVE report (the user's renames/merges applied). A local provider holds the
    GPU for the whole run and frees it at the end, so Ollama is never resident when the
    design-cast job loads Qwen. The paid gate for an anthropic provider stays at the boundary."""
    from seiyuu.services.attribution import load_report
    from seiyuu.services.llm_advisory import resolve_advisory, run_with_provider

    book_dir = Path(cfg.books_dir) / book_id
    report, _warnings = load_report(book_dir)
    existing = load_plan(cfg.output_dir, book_id)
    resolved = resolve_advisory(cfg, cfg.describe_provider, cfg.describe_model, None)
    version = cfg.describe_prompt_version
    first_person = is_first_person(report)

    def work(llm) -> CastPlan:
        with AttributionCache(book_dir / "attribution.db") as cache:
            sketcher = cached_sketcher(
                llm,
                cache,
                book_id=book_id,
                prompts_dir=cfg.prompts_dir,
                prompt_version=version,
                first_person=first_person,
            )
            return draft_plan(
                report,
                sketch=sketcher,
                provider_id=llm.provider_id,
                model_id=llm.model_id,
                prompt_version=version,
                existing=existing,
                progress=progress,
                check_cancel=check_cancel,
            )

    plan = run_with_provider(cfg, resolved, version, "describe", work, gpu=gpu, provider=provider)
    save_plan(cfg.output_dir, plan)
    return plan
