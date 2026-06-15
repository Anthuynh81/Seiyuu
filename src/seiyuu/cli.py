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


def _build_provider(cfg, provider_id: str, model: str, prompt_version: str):
    """Construct an attribution provider, passing only the kwargs each backend needs."""
    from seiyuu.attribute.providers import get_provider

    kwargs = {"prompt_version": prompt_version}
    if provider_id == "local":
        kwargs["base_url"] = cfg.ollama_base_url
    return get_provider(provider_id, model=model, prompts_dir=cfg.prompts_dir, **kwargs)


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
    books_dir: Path | None,
) -> None:
    """Attribute speakers with the local LLM: writes attribution.json + a cache DB."""
    from seiyuu.attribute import (
        ATTRIBUTION_NAME,
        AttributionCache,
        AttributionError,
        attribute_book,
        write_attribution,
    )
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, "normalized.json", "Run `seiyuu ingest` first."
    )
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )

    provider_id = provider_id or cfg.attribution_provider
    model = model or cfg.attribution_model
    prompt_version = prompt_version or cfg.attribution_prompt_version
    use_hybrid = cfg.attribution_hybrid if hybrid is None else hybrid

    try:
        provider = _build_provider(cfg, provider_id, model, prompt_version)
        escalation = None
        if use_hybrid and provider_id != "anthropic":
            escalation = _build_provider(cfg, "anthropic", cfg.anthropic_model, prompt_version)
        with AttributionCache(book_dir / "attribution.db") as cache:
            report = attribute_book(
                book,
                provider,
                cache=cache,
                budget_tokens=cfg.attribution_chunk_tokens,
                overlap_blocks=cfg.attribution_chunk_overlap_blocks,
                max_local_retries=cfg.attribution_max_local_retries,
                escalation_provider=escalation,
                chapters=chapter_indices,
                progress=click.echo,
            )
    except (AttributionError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    write_attribution(report, book_dir)
    n_segments = sum(len(c.segments) for c in report.chapters)
    click.echo(
        f"done: {len(report.registry.characters)} characters, {n_segments} segments "
        f"({provider.provider_id}/{provider.model_id}, prompt {prompt_version})"
    )
    if report.flagged:
        click.echo(f"  {len(report.flagged)} blocks flagged for review — see `seiyuu characters`")
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
    """Report attributed characters, sample lines, and review flags (reads attribution.json)."""
    import textwrap
    from collections import Counter

    from seiyuu.attribute import ATTRIBUTION_NAME, AttributionReport, SegmentType
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    report = AttributionReport.model_validate_json(
        (book_dir / ATTRIBUTION_NAME).read_text(encoding="utf-8")
    )

    threshold = cfg.attribution_confidence_threshold
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    narration = low_confidence = 0
    for chapter in report.chapters:
        for seg in chapter.segments:
            if seg.speaker is None:
                narration += 1
                continue
            counts[seg.speaker] += 1
            if seg.confidence < threshold:
                low_confidence += 1
            if (
                seg.type is SegmentType.DIALOGUE
                and len(samples.setdefault(seg.speaker, [])) < sample_lines
            ):
                samples[seg.speaker].append(seg.text)

    provenance = f"{report.provider_id}/{report.model_id}, prompt {report.prompt_version}"
    click.echo(f"{report.book_id}  ({provenance})")
    click.echo(f"narration segments: {narration}")
    click.echo(f"characters: {len(report.registry.characters)}\n")

    for char in sorted(report.registry.characters, key=lambda c: counts[c.id], reverse=True):
        meta = ", ".join(filter(None, [char.gender, char.age_hint])) or "—"
        aliases = f"  aka {', '.join(char.aliases)}" if char.aliases else ""
        click.echo(
            f"  {char.canonical_name} [{char.id}] ({meta}) — {counts[char.id]} lines{aliases}"
        )
        for line in samples.get(char.id, []):
            click.echo(f"      “{textwrap.shorten(line, width=72)}”")

    if low_confidence:
        click.echo(f"\nlow-confidence speaker calls (< {threshold}): {low_confidence}")
    if report.flagged:
        click.echo(f"\nflagged for review: {len(report.flagged)} blocks")
        for fb in report.flagged[:10]:
            click.echo(f"  ch{fb.chapter_index} {fb.block_id}: {fb.reason}")
    for note in report.registry_notes:
        click.echo(f"note: {note}")


def _pause_options(fn):
    """Pause-tuning flags shared by `assemble` and `convert` (seconds)."""
    for name, help_text in reversed(
        [
            ("--pause-paragraph", "Silence between paragraphs."),
            ("--pause-after-heading", "Silence after a chapter heading."),
            ("--pause-scene-break", "Silence at a scene break (replaces the paragraph gap)."),
            ("--pause-lead-in", "Silence at the start of each chapter."),
            ("--pause-lead-out", "Silence at the end of each chapter."),
        ]
    ):
        fn = click.option(name, type=float, default=None, help=f"{help_text} [default: see SPEC]")(
            fn
        )
    return fn


def _build_pauses(**overrides):
    from seiyuu.assemble import PauseProfile

    defaults = PauseProfile()
    return PauseProfile(
        paragraph=overrides.get("pause_paragraph") or defaults.paragraph,
        after_heading=overrides.get("pause_after_heading") or defaults.after_heading,
        scene_break=overrides.get("pause_scene_break") or defaults.scene_break,
        chapter_lead_in=overrides.get("pause_lead_in") or defaults.chapter_lead_in,
        chapter_lead_out=overrides.get("pause_lead_out") or defaults.chapter_lead_out,
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
def assemble(book_id: str, output_dir: Path | None, **pause_overrides) -> None:
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
            book_dir, pauses=_build_pauses(**pause_overrides), progress=click.echo
        )
    except AssembleError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"done: {len(result.mp3_paths)} chapter MP3s, "
        f"{result.total_seconds / 60:.1f} min total -> {book_dir / 'chapters'}"
    )


# A full-book render is a long GPU job; above this many segments, confirm.
FULL_RENDER_CONFIRM_BLOCKS = 300


@main.command()
@click.argument("epub_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--engine", "engine_id", default=None, help="TTS engine (default from settings).")
@click.option("--voice", default=None, help="Voice/preset id (default from settings).")
@click.option(
    "--chapter",
    "chapter_indices",
    multiple=True,
    type=int,
    help="Convert only these 1-based chapters (repeatable). Default: all.",
)
@click.option("--speed", default=1.0, show_default=True, help="Speech speed multiplier.")
@click.option("--seed", default=41172, show_default=True, help="Synthesis seed.")
@click.option("--yes", "-y", is_flag=True, help="Skip the full-book render confirmation.")
@click.option(
    "--books-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Normalized books root (default: settings.books_dir).",
)  # fmt: skip
@click.option(
    "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Render output root (default: settings.output_dir).",
)  # fmt: skip
@_pause_options
def convert(
    epub_path: Path,
    engine_id: str | None,
    voice: str | None,
    chapter_indices: tuple[int, ...],
    speed: float,
    seed: int,
    yes: bool,
    books_dir: Path | None,
    output_dir: Path | None,
    **pause_overrides,
) -> None:
    """Full pipeline: EPUB -> normalized JSON -> single-voice render -> chapter MP3s."""
    from seiyuu.assemble import AssembleError, assemble_book
    from seiyuu.engines import get_engine
    from seiyuu.ingest import IngestError, parse_epub, write_normalized
    from seiyuu.render import RenderError, render_book
    from seiyuu.settings import get_settings

    cfg = get_settings()

    click.echo("== ingest ==")
    try:
        ingest_result = parse_epub(epub_path)
    except IngestError as exc:
        raise click.ClickException(str(exc)) from exc
    book = ingest_result.book
    write_normalized(book, books_dir or cfg.books_dir)
    click.echo(f"{book.book_meta.book_id}: {len(book.chapters)} chapters")

    wanted = set(chapter_indices)
    speakable = sum(
        1
        for ci, c in enumerate(book.chapters, start=1)
        for b in c.blocks
        if b.is_speakable and (not wanted or ci in wanted)
    )
    if not chapter_indices and speakable > FULL_RENDER_CONFIRM_BLOCKS and not yes:
        words = sum(len(b.text.split()) for c in book.chapters for b in c.blocks)
        click.confirm(
            f"Full-book render: {speakable} segments, roughly {words / 150 / 60:.1f} hours "
            f"of audio to synthesize. Continue?",
            abort=True,
        )

    click.echo("== render ==")
    book_dir = (output_dir or cfg.output_dir) / book.book_meta.book_id
    try:
        engine = get_engine(engine_id or cfg.tts_engine)
        render_result = render_book(
            book,
            engine,
            voice or cfg.kokoro_default_voice,
            book_dir,
            settings={"speed": speed},
            seed=seed,
            chapters=chapter_indices,
            progress=click.echo,
        )
    except (RenderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{render_result.synthesized} synthesized, {render_result.cache_hits} from cache")

    click.echo("== assemble ==")
    try:
        assemble_result = assemble_book(
            book_dir, pauses=_build_pauses(**pause_overrides), progress=click.echo
        )
    except AssembleError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"done: {len(assemble_result.mp3_paths)} chapter MP3s, "
        f"{assemble_result.total_seconds / 60:.1f} min total -> {book_dir / 'chapters'}"
    )


if __name__ == "__main__":
    main()
