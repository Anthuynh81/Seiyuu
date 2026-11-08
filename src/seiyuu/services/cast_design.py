"""Qwen-first casting, part 2: design the cast from the plan.

Design doc: design/qwen-first-casting-2026-11-08.md.

One GPU job walks the cast plan and, for every row that designs its own voice (not "keep current
voice", not "same voice as"), runs the voice-design flow (``services/voice_design.py``): the
row's description, led by its known age and gender, and a design line long enough to clone from,
spoken in ``row.takes`` takes. The recommended take becomes a library voice (CLONED qwen3tts,
tagged with the book). Then the book's assignment is written in one go: designed rows get their
voice, "same voice as" rows get their target's voice, the narrator gets its designed voice (or
the protagonist's, when it shares one), and "keep current voice" rows keep what they had.

Resumable: the plan is saved after every row with its design id and voice id, a row whose voice
already exists is skipped, and a design interrupted mid-takes resumes with only the missing takes.
The VoiceDesign checkpoint stays resident across rows and is freed once at the end.

``keep_other_take`` swaps a row to another of its designed takes later (CPU only): it keeps
that take as a new voice and re-points the row, its "same voice as" followers and, when it
shares the voice, the narrator.
"""

from collections.abc import Callable
from typing import Any

from seiyuu.normalize import normalize_text
from seiyuu.services.assignment import draft_assignment, load_existing_assignment, save_assignment
from seiyuu.services.cast_plan import CastPlan, PlanRow, load_plan, save_plan
from seiyuu.services.common import ServiceError
from seiyuu.services.voice_design import (
    DESIGN_ENGINE,
    design_dir,
    keep_take,
    load_record,
    run_design,
    start_design,
)
from seiyuu.voices import VoiceAssignment, VoiceLibrary

MIN_DESIGN_WORDS = 15  # ~5 s of speech: enough for VoiceDesign to settle and to clone from
MAX_DESIGN_WORDS = 45  # ~15 s: a longer clip only eats Qwen's generation budget at render


def design_text(row: PlanRow) -> str:
    """The line the row's takes speak: its chosen sample, plus the lines after it (wrapping)
    until there are at least ``MIN_DESIGN_WORDS`` words, never past ``MAX_DESIGN_WORDS``."""
    lines = [line.strip() for line in row.sample_lines if line.strip()]
    if not lines:
        raise ServiceError(f"{row.name!r} has no line to design from")
    start = min(row.sample_index, len(lines) - 1)
    ordered = lines[start:] + lines[:start]
    picked: list[str] = []
    words = 0
    for line in ordered:
        n = len(line.split())
        if picked and words + n > MAX_DESIGN_WORDS:
            break
        picked.append(line)
        words += n
        if words >= MIN_DESIGN_WORDS:
            break
    return " ".join(picked)


_AGE_WORDS = {
    "child": "Child",
    "teen": "Teenage",
    "young adult": "Young adult",
    "adult": "Adult",
    "older adult": "Middle-aged",
    "elderly": "Elderly",
}


def design_prompt(row: PlanRow) -> str:
    """The VoiceDesign description: the row's known age and gender first, then its description,
    so a description that never says "male"/"female" can't come out as either at random."""
    age = _AGE_WORDS.get(row.age or "", "")
    gender = row.gender if row.gender in {"male", "female"} else ""
    lead = " ".join(w for w in (age, gender) if w)
    if not lead:
        return row.description
    lead = lead[0].upper() + lead[1:]
    return f"{lead} voice. {row.description}"


def designing_rows(plan: CastPlan) -> list[PlanRow]:
    """Rows that design their own voice, narrator first (it speaks the most)."""
    rows = [r for r in plan.rows if not r.keep_current and r.same_as is None]
    narrator = plan.narrator
    if not narrator.keep_current and narrator.same_as is None:
        rows.insert(0, narrator)
    return rows


def _design_row(
    row: PlanRow,
    *,
    book_id: str,
    data_dir,
    library: VoiceLibrary,
    engine: Any,
    validator: Any,
    gpu: Any,
    progress: Callable[[str], None],
    check_cancel: Callable[[], None],
) -> None:
    """Design (or finish designing) one row and keep its recommended take; mutates ``row``."""
    if row.design_id is None or not design_dir(data_dir, row.design_id).is_dir():
        text = normalize_text(design_text(row), profile=DESIGN_ENGINE)
        record = start_design(
            data_dir, name=row.name, description=design_prompt(row), text=text, takes=row.takes
        )
        row.design_id = record.design_id
    directory = design_dir(data_dir, row.design_id)
    record = run_design(
        directory,
        engine=engine,
        validator=validator,
        gpu=gpu,
        progress=lambda msg: progress(f"{row.name}: {msg}"),
        check_cancel=check_cancel,
        release=False,
    )
    if record.recommended is None:
        raise ServiceError(f"{row.name!r}: the design made no takes")
    meta = keep_take(
        directory,
        record.recommended,
        library=library,
        name=row.name,
        tags=["designed", book_id],
    )
    row.voice_id = meta.voice_id


def write_assignment(
    plan: CastPlan, base: VoiceAssignment, library: VoiceLibrary
) -> VoiceAssignment:
    """The book's assignment after designing: see the module docstring."""
    rows = {r.character_id: r for r in plan.rows}
    assignments = dict(base.assignments)

    def voice_of(row: PlanRow) -> str | None:
        target = rows.get(row.same_as) if row.same_as else row
        vid = target.voice_id if target is not None else None
        return vid if vid and library.meta_path(vid).is_file() else None

    for row in plan.rows:
        if row.keep_current:
            continue
        vid = voice_of(row)
        if vid:
            assignments[row.character_id] = vid
    narrator_vid = base.narrator_voice_id
    if not plan.narrator.keep_current:
        narrator_vid = voice_of(plan.narrator) or narrator_vid
    return base.model_copy(update={"assignments": assignments, "narrator_voice_id": narrator_vid})


def run_design_cast(
    cfg,
    book_id: str,
    *,
    engine: Any,
    validator: Any,
    gpu: Any,
    progress: Callable[[str], None] = lambda _m: None,
    check_cancel: Callable[[], None] = lambda: None,
) -> tuple[CastPlan, VoiceAssignment]:
    """Design every row that needs a voice, then write the book's assignment. Resumable."""
    from seiyuu.services.attribution import load_report

    plan = load_plan(cfg.output_dir, book_id)
    if plan is None:
        raise ServiceError(f"no cast plan for {book_id!r}; describe the cast first")
    library = VoiceLibrary(cfg.voices_dir)
    todo = [
        r
        for r in designing_rows(plan)
        if not (r.voice_id and library.meta_path(r.voice_id).is_file())
    ]
    try:
        for i, row in enumerate(todo, 1):
            check_cancel()
            progress(f"designing {row.name} ({i} of {len(todo)})")
            _design_row(
                row,
                book_id=book_id,
                data_dir=cfg.data_dir,
                library=library,
                engine=engine,
                validator=validator,
                gpu=gpu,
                progress=progress,
                check_cancel=check_cancel,
            )
            save_plan(cfg.output_dir, plan)  # resume point: this row is done
    finally:
        save_plan(cfg.output_dir, plan)  # keep a half-designed row's design id too
        if todo:
            gpu.free_all()  # VoiceDesign was held across rows; nothing renders with it

    base = load_existing_assignment(cfg.output_dir, book_id)
    if base is None:
        # nothing saved yet: "keep current voice" means the quick Kokoro draft
        report, _warnings = load_report(cfg.books_dir / book_id)
        base = draft_assignment(
            report, library, default_preset=cfg.kokoro_default_voice, strategy="smart"
        )
    assignment = write_assignment(plan, base, library)
    save_assignment(assignment, cfg.output_dir)
    progress(f"cast {sum(1 for r in plan.rows if not r.keep_current)} characters")
    return plan, assignment


def keep_other_take(
    cfg, book_id: str, character_id: str, take: int
) -> tuple[CastPlan, VoiceAssignment]:
    """Swap one designed row to another of its takes (no GPU) and re-point who uses it."""
    from seiyuu.services.cast_plan import NARRATOR_ROW

    plan = load_plan(cfg.output_dir, book_id)
    if plan is None:
        raise ServiceError(f"no cast plan for {book_id!r}")
    row = (
        plan.narrator
        if character_id == NARRATOR_ROW
        else next((r for r in plan.rows if r.character_id == character_id), None)
    )
    if row is None or row.design_id is None:
        raise ServiceError(f"{character_id!r} has no designed voice to change")
    directory = design_dir(cfg.data_dir, row.design_id)
    record = load_record(directory)
    library = VoiceLibrary(cfg.voices_dir)
    kept = record.kept.get(take)
    if kept and library.meta_path(kept).is_file():
        row.voice_id = kept  # this take is already a voice: reuse it
    else:
        meta = keep_take(
            directory, take, library=library, name=row.name, tags=["designed", book_id]
        )
        row.voice_id = meta.voice_id
    save_plan(cfg.output_dir, plan)
    base = load_existing_assignment(cfg.output_dir, book_id)
    if base is None:
        raise ServiceError(f"{book_id!r} has no saved cast yet; design the cast first")
    assignment = write_assignment(plan, base, library)
    save_assignment(assignment, cfg.output_dir)
    return plan, assignment
