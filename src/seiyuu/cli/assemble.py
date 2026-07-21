"""Assembly commands: per-chapter MP3 assembly and .m4b mastering."""

from pathlib import Path

import click

from seiyuu.cli import main
from seiyuu.cli.common import (
    _build_loudness,
    _build_pauses,
    _loudness_options,
    _pause_options,
    _resolve_book_dir,
)


@main.command()
@click.argument("book_id")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root (default: settings.output_dir).",
)
@_pause_options
@_loudness_options
def assemble(book_id: str, output_dir: Path | None, **overrides) -> None:
    """Assemble rendered segments into per-chapter MP3s (output/{book}/chapters/)."""
    from seiyuu.assemble import AssembleError, assemble_book
    from seiyuu.render import MANIFEST_NAME
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        output_dir or cfg.output_dir, book_id, MANIFEST_NAME, "Run `seiyuu render` first."
    )
    try:
        result = assemble_book(
            book_dir,
            pauses=_build_pauses(**overrides),
            loudness=_build_loudness(cfg, **overrides),
            progress=click.echo,
        )
    except AssembleError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"done: {len(result.mp3_paths)} chapter MP3s, "
        f"{result.total_seconds / 60:.1f} min total -> {book_dir / 'chapters'}"
    )


@main.command()
@click.argument("book_id")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root (default: settings.output_dir).",
)
@click.option(
    "--cover",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Cover image (jpg/png) embedded in the .m4b.",
)
@click.option("--bitrate", default="64k", show_default=True, help="AAC bitrate.")
@click.option(
    "--target-minutes",
    type=float,
    default=None,
    help="Nudge total runtime toward this many minutes (clamped tempo).",
)
@_pause_options
@_loudness_options
def master(
    book_id: str,
    output_dir: Path | None,
    cover: Path | None,
    bitrate: str,
    target_minutes: float | None,
    **overrides,
) -> None:
    """Build a chaptered .m4b audiobook (output/{book}/{book_id}.m4b) from the render manifest."""
    from seiyuu.assemble import AssembleError, master_book
    from seiyuu.render import MANIFEST_NAME
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        output_dir or cfg.output_dir, book_id, MANIFEST_NAME, "Run `seiyuu render` first."
    )
    try:
        result = master_book(
            book_dir,
            pauses=_build_pauses(**overrides),
            loudness=_build_loudness(cfg, **overrides),
            cover=cover,
            bitrate=bitrate,
            target_seconds=target_minutes * 60 if target_minutes else None,
            tempo_bounds=(cfg.tempo_min, cfg.tempo_max),
            progress=click.echo,
        )
    except AssembleError as exc:
        raise click.ClickException(str(exc)) from exc
    tempo_note = f" (atempo {result.tempo:.3f})" if abs(result.tempo - 1.0) > 1e-3 else ""
    click.echo(
        f"done: {result.m4b_path.name} — {result.chapters} chapters, "
        f"{result.total_seconds / 60:.1f} min{tempo_note} -> {result.m4b_path}"
    )
