"""Seiyuu CLI. Runnable as `seiyuu` or `python -m seiyuu.cli`."""

from pathlib import Path

import click

from seiyuu import __version__


@click.group()
@click.version_option(__version__, prog_name="seiyuu")
def main() -> None:
    """Seiyuu — multi-voice audiobook creator."""


@main.command()
@click.argument("epub_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--include-item",
    "include_items",
    multiple=True,
    help="Force-include a spine item (substring of its file name or id) that the "
    "front/back-matter heuristic would skip.",
)
@click.option(
    "--exclude-item",
    "exclude_items",
    multiple=True,
    help="Force-exclude a spine item (substring of its file name or id).",
)
@click.option(
    "--split-level",
    default=2,
    show_default=True,
    help="Maximum heading level (h1..hN) that starts a new chapter.",
)
@click.option(
    "--books-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output root directory (default: settings.books_dir).",
)
def ingest(
    epub_path: Path,
    include_items: tuple[str, ...],
    exclude_items: tuple[str, ...],
    split_level: int,
    books_dir: Path | None,
) -> None:
    """Ingest an EPUB into normalized JSON (books/{book_id}/normalized.json)."""
    from seiyuu.ingest import IngestError, parse_epub, write_normalized
    from seiyuu.settings import get_settings

    try:
        result = parse_epub(
            epub_path,
            include_items=include_items,
            exclude_items=exclude_items,
            split_level=split_level,
        )
    except IngestError as exc:
        raise click.ClickException(str(exc)) from exc

    out_path = write_normalized(result.book, books_dir or get_settings().books_dir)

    meta = result.book.book_meta
    n_blocks = sum(len(c.blocks) for c in result.book.chapters)
    click.echo(f"book_id:  {meta.book_id}")
    click.echo(f"title:    {meta.title} — {', '.join(meta.authors) or 'unknown author'}")
    click.echo(f"chapters: {len(result.book.chapters)} ({n_blocks} blocks)")
    for name in result.skipped_items:
        click.echo(f"skipped spine item: {name}")
    for section in result.dropped_sections:
        click.echo(f"dropped section:    {section}")
    click.echo(f"wrote: {out_path}")


def _resolve_book_dir(root: Path, book_id: str, marker: str, hint: str) -> Path:
    """Accept a full book_id or an unambiguous prefix; the dir must contain `marker`."""
    exact = root / book_id
    if (exact / marker).is_file():
        return exact
    if not root.is_dir():
        raise click.ClickException(f"directory not found: {root}. {hint}")
    matches = [
        d
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith(book_id) and (d / marker).is_file()
    ]
    if len(matches) == 1:
        return matches[0]
    known = ", ".join(sorted(d.name for d in root.iterdir() if (d / marker).is_file())) or "(none)"
    problem = "is ambiguous" if matches else "not found"
    raise click.ClickException(f"book {book_id!r} {problem}; candidates: {known}. {hint}")


@main.command()
@click.argument("book_id")
@click.option("--engine", "engine_id", default=None, help="TTS engine (default from settings).")
@click.option("--voice", default=None, help="Voice/preset id (default from settings).")
@click.option(
    "--chapter",
    "chapter_indices",
    multiple=True,
    type=int,
    help="Render only these 1-based chapters (repeatable). Default: all.",
)
@click.option("--speed", default=1.0, show_default=True, help="Speech speed multiplier.")
@click.option("--seed", default=41172, show_default=True, help="Synthesis seed.")
@click.option(
    "--books-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where normalized books live (default: settings.books_dir).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root (default: settings.output_dir).",
)
def render(
    book_id: str,
    engine_id: str | None,
    voice: str | None,
    chapter_indices: tuple[int, ...],
    speed: float,
    seed: int,
    books_dir: Path | None,
    output_dir: Path | None,
) -> None:
    """Render a book single-voice: cached segment WAVs + manifest.json."""
    from seiyuu.engines import get_engine
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.render import RenderError, render_book
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, "normalized.json", "Run `seiyuu ingest` first."
    )
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )

    engine_id = engine_id or cfg.tts_engine
    voice = voice or cfg.kokoro_default_voice
    try:
        engine = get_engine(engine_id)
        result = render_book(
            book,
            engine,
            voice,
            (output_dir or cfg.output_dir) / book.book_meta.book_id,
            settings={"speed": speed},
            seed=seed,
            chapters=chapter_indices,
            progress=click.echo,
        )
    except (RenderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    minutes = result.total_audio_seconds / 60
    click.echo(
        f"done: {result.synthesized} segments synthesized, "
        f"{result.cache_hits} from cache, {minutes:.1f} min of audio"
    )
    click.echo(f"manifest: {result.manifest_path}")


@main.command()
@click.argument("book_id")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root (default: settings.output_dir).",
)
def assemble(book_id: str, output_dir: Path | None) -> None:
    """Assemble rendered segments into per-chapter MP3s (output/{book}/chapters/)."""
    from seiyuu.assemble import AssembleError, assemble_book
    from seiyuu.render import MANIFEST_NAME
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        output_dir or cfg.output_dir, book_id, MANIFEST_NAME, "Run `seiyuu render` first."
    )
    try:
        result = assemble_book(book_dir, progress=click.echo)
    except AssembleError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"done: {len(result.mp3_paths)} chapter MP3s, "
        f"{result.total_seconds / 60:.1f} min total -> {book_dir / 'chapters'}"
    )


if __name__ == "__main__":
    main()
