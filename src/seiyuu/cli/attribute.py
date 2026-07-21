"""Attribution commands: attribute, adjudicate, characters, and the manual edit overlay."""

from pathlib import Path

import click

from seiyuu.attribute.models import EmotionLabel
from seiyuu.cli import main
from seiyuu.cli.common import _edit_books_dir_option, _resolve_book_dir


@main.command()
@click.argument("book_id")
@click.option(
    "--provider", "provider_id", default=None, help="Attribution provider (default from settings)."
)
@click.option("--model", default=None, help="LLM model id (default from settings).")
@click.option("--prompt-version", default=None, help="Prompt version (default from settings).")
@click.option(
    "--chapter",
    "chapter_indices",
    multiple=True,
    type=int,
    help="Attribute only these 1-based chapters (repeatable). Default: all.",
)
@click.option(
    "--hybrid/--no-hybrid",
    default=None,
    help="Escalate chunks that fail local retries to the anthropic provider (paid).",
)
@click.option(
    "--adjudicate/--no-adjudicate",
    default=None,
    help="Run the opt-in LLM alias adjudication after attribution (full-book runs only).",
)
@click.option(
    "--emit-thoughts/--no-emit-thoughts",
    default=None,
    help="Emit thought segments for italic interior monologue (uses the v4 prompt; opt-in).",
)
@click.option(
    "--books-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where normalized books live (default: settings.books_dir).",
)
def attribute(
    book_id: str,
    provider_id: str | None,
    model: str | None,
    prompt_version: str | None,
    chapter_indices: tuple[int, ...],
    hybrid: bool | None,
    adjudicate: bool | None,
    emit_thoughts: bool | None,
    books_dir: Path | None,
) -> None:
    """Attribute speakers with the local LLM: writes attribution.json + a cache DB."""
    from seiyuu.attribute import ATTRIBUTION_NAME, AttributionError
    from seiyuu.gpu import GpuBusyError
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, "normalized.json", "Run `seiyuu ingest` first."
    )
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )

    if chapter_indices and adjudicate:
        raise click.ClickException(
            "--adjudicate needs the full-book registry; drop --chapter or run "
            "`seiyuu adjudicate` after attributing all chapters."
        )

    from seiyuu.services import run_attribution

    try:
        report = run_attribution(
            book,
            book_dir,
            cfg=cfg,
            provider_id=provider_id,
            model=model,
            prompt_version=prompt_version,
            use_hybrid=hybrid,
            use_adjudicate=adjudicate,
            emit_thoughts=emit_thoughts,
            chapters=chapter_indices,
            progress=click.echo,
        )
    except (AttributionError, GpuBusyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    n_segments = sum(len(c.segments) for c in report.chapters)
    click.echo(
        f"done: {len(report.registry.characters)} characters, {n_segments} segments "
        f"({report.provider_id}/{report.model_id}, prompt {report.prompt_version})"
    )
    if report.flagged:
        click.echo(f"  {len(report.flagged)} blocks flagged for review — see `seiyuu characters`")
    # Surface a detected non-double dialogue convention (UK single-quote books) after the
    # run too — the note also lives in registry_notes for `seiyuu characters` and the API.
    for note in report.registry_notes:
        if note.startswith("dialogue convention:"):
            click.echo(f"  {note}")
    click.echo(f"wrote: {book_dir / ATTRIBUTION_NAME}")


@main.command()
@click.argument("book_id")
@click.option(
    "--books-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where normalized books live (default: settings.books_dir).",
)
def adjudicate(book_id: str, books_dir: Path | None) -> None:
    """Opt-in LLM alias adjudication over an attributed book: merges first-name/nickname aliases.

    Operates on the full attribution.json and rewrites it in place; cached per candidate set
    so re-runs are free and deterministic. Paid only if the adjudication provider is anthropic
    (needs ANTHROPIC_API_KEY); local is free and reuses the GPU.
    """
    from seiyuu.attribute import ATTRIBUTION_NAME, AttributionError
    from seiyuu.gpu import GpuBusyError
    from seiyuu.services import ServiceError, run_adjudication
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    try:
        report = run_adjudication(book_dir, cfg=cfg, progress=click.echo)
    except (ServiceError, AttributionError, GpuBusyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"done: {len(report.registry.characters)} characters "
        f"({report.provider_id}/{report.model_id})"
    )
    click.echo(f"wrote: {book_dir / ATTRIBUTION_NAME}")


@main.command()
@click.argument("book_id")
@click.option("--sample-lines", default=2, show_default=True, help="Dialogue lines per character.")
@click.option(
    "--books-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where normalized books live (default: settings.books_dir).",
)
def characters(book_id: str, sample_lines: int, books_dir: Path | None) -> None:
    """Report attributed characters, sample lines, and review flags (edits applied)."""
    import textwrap

    from seiyuu.attribute import ATTRIBUTION_NAME
    from seiyuu.services import ServiceError, characters_overview
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    try:
        overview = characters_overview(
            book_dir,
            confidence_threshold=cfg.attribution_confidence_threshold,
            sample_lines=sample_lines,
        )
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc

    provenance = f"{overview.provider_id}/{overview.model_id}, prompt {overview.prompt_version}"
    click.echo(f"{overview.book_id}  ({provenance})")
    click.echo(f"narration segments: {overview.narration_segments}")
    if overview.unattributed_quote_segments:
        click.echo(
            "unattributed quotes (narrator voice, no speaker): "
            f"{overview.unattributed_quote_segments}"
        )
    click.echo(f"characters: {len(overview.characters)}\n")

    for char in overview.characters:
        meta = ", ".join(filter(None, [char.gender, char.age_hint])) or "—"
        aliases = f"  aka {', '.join(char.aliases)}" if char.aliases else ""
        click.echo(f"  {char.name} [{char.id}] ({meta}) — {char.line_count} lines{aliases}")
        for line in char.sample_lines:
            click.echo(f"      “{textwrap.shorten(line, width=72)}”")

    if overview.low_confidence_segments:
        click.echo(
            f"\nlow-confidence speaker calls (< {overview.confidence_threshold}): "
            f"{overview.low_confidence_segments}"
        )
    if overview.flagged:
        click.echo(f"\nflagged for review: {len(overview.flagged)} blocks")
        for fb in overview.flagged[:10]:
            click.echo(f"  ch{fb.chapter_index} {fb.block_id}: {fb.reason}")
    for note in overview.notes:
        click.echo(f"note: {note}")
    for warning in overview.edit_warnings:
        click.echo(f"edit overlay: {warning}")


@main.group()
def edit() -> None:
    """Manual attribution edits — a durable overlay that survives re-attribution."""


def _edit_book_dir(book_id: str, books_dir: Path | None) -> Path:
    from seiyuu.settings import get_settings

    return _resolve_book_dir(
        books_dir or get_settings().books_dir,
        book_id,
        "attribution.json",
        "Run `seiyuu attribute` first.",
    )


@edit.command("rename")
@click.argument("book_id")
@click.argument("character_id")
@click.argument("new_name")
@_edit_books_dir_option
def edit_rename(book_id: str, character_id: str, new_name: str, books_dir: Path | None) -> None:
    """Rename a character (the old name is kept as an alias)."""
    from seiyuu.services import RenameCharacter, ServiceError, record_edit

    book_dir = _edit_book_dir(book_id, books_dir)
    try:
        record_edit(book_dir, RenameCharacter(character_id=character_id, new_name=new_name))
    except (ServiceError, ValueError) as exc:  # ValueError: pydantic op validation
        raise click.ClickException(str(exc)) from exc
    click.echo(f"renamed {character_id} -> {new_name!r} (durable; survives re-attribution)")


@edit.command("merge")
@click.argument("book_id")
@click.argument("loser_id")
@click.argument("winner_id")
@_edit_books_dir_option
def edit_merge(book_id: str, loser_id: str, winner_id: str, books_dir: Path | None) -> None:
    """Merge LOSER into WINNER: segments move over, names become aliases."""
    from seiyuu.services import MergeCharacters, ServiceError, record_edit

    book_dir = _edit_book_dir(book_id, books_dir)
    try:
        record_edit(book_dir, MergeCharacters(loser_id=loser_id, winner_id=winner_id))
    except (ServiceError, ValueError) as exc:  # ValueError: pydantic op validation
        raise click.ClickException(str(exc)) from exc
    click.echo(f"merged {loser_id} into {winner_id} (durable; survives re-attribution)")


@edit.command("reassign")
@click.argument("book_id")
@click.argument("block_id")
@click.argument("segment_index", type=int)
@click.option("--speaker", default=None, help="Character id to assign the segment to.")
@click.option("--narration", is_flag=True, help="Make the segment narration instead.")
@_edit_books_dir_option
def edit_reassign(
    book_id: str,
    block_id: str,
    segment_index: int,
    speaker: str | None,
    narration: bool,
    books_dir: Path | None,
) -> None:
    """Reassign one segment's speaker (SEGMENT_INDEX is within the block, 0-based)."""
    from seiyuu.services import ReassignSegment, ServiceError, record_edit

    if (speaker is None) == (not narration):  # exactly one of the two must be given
        raise click.ClickException("pass exactly one of --speaker CHARACTER_ID or --narration")
    book_dir = _edit_book_dir(book_id, books_dir)
    try:
        record_edit(
            book_dir,
            ReassignSegment(block_id=block_id, segment_index=segment_index, speaker=speaker),
        )
    except (ServiceError, ValueError) as exc:  # ValueError: pydantic op validation (index<0)
        raise click.ClickException(str(exc)) from exc
    target = speaker if speaker is not None else "narration"
    click.echo(f"reassigned {block_id}[{segment_index}] -> {target} (durable)")


@edit.command("set-emotion")
@click.argument("book_id")
@click.argument("block_id")
@click.argument("segment_index", type=int)
@click.option(
    "--label",
    type=click.Choice([e.value for e in EmotionLabel]),
    default=None,
    help="Emotion label to set on the segment.",
)
@click.option(
    "--intensity", type=click.IntRange(1, 3), default=2, help="Intensity 1..3 (default 2)."
)
@click.option("--clear", is_flag=True, help="Clear the segment's emotion instead of setting one.")
@_edit_books_dir_option
def edit_set_emotion(
    book_id: str,
    block_id: str,
    segment_index: int,
    label: str | None,
    intensity: int,
    clear: bool,
    books_dir: Path | None,
) -> None:
    """Set or clear one segment's emotion overlay (SEGMENT_INDEX is 0-based within the block)."""
    from seiyuu.attribute.models import EmotionVerdict
    from seiyuu.services import ServiceError, SetEmotion, record_edit

    if clear == (label is not None):  # exactly one of --label or --clear
        raise click.ClickException("pass exactly one of --label LABEL or --clear")
    emotion = None if clear else EmotionVerdict(label=EmotionLabel(label), intensity=intensity)
    book_dir = _edit_book_dir(book_id, books_dir)
    try:
        record_edit(
            book_dir,
            SetEmotion(block_id=block_id, segment_index=segment_index, emotion=emotion),
        )
    except (ServiceError, ValueError) as exc:  # ValueError: pydantic op validation
        raise click.ClickException(str(exc)) from exc
    target = "cleared" if clear else f"{label} (intensity {intensity})"
    click.echo(f"emotion for {block_id}[{segment_index}] -> {target} (durable)")


@edit.command("list")
@click.argument("book_id")
@_edit_books_dir_option
def edit_list(book_id: str, books_dir: Path | None) -> None:
    """Show the edit overlay ops in order."""
    from seiyuu.services import load_edits

    log = load_edits(_edit_book_dir(book_id, books_dir))
    if not log.ops:
        click.echo("no manual edits")
        return
    for i, op in enumerate(log.ops):
        click.echo(f"  {i}: {op.model_dump_json()}")


@edit.command("undo")
@click.argument("book_id")
@_edit_books_dir_option
def edit_undo(book_id: str, books_dir: Path | None) -> None:
    """Remove the most recent edit op."""
    from seiyuu.services import undo_edit

    op = undo_edit(_edit_book_dir(book_id, books_dir))
    if op is None:
        click.echo("no manual edits to undo")
        return
    click.echo(f"removed: {op.model_dump_json()}")
