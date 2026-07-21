"""Book lifecycle commands: ingest, convert (the full pipeline), and delete."""

from pathlib import Path

import click

from seiyuu.cli import main
from seiyuu.cli.common import (
    _build_loudness,
    _build_pauses,
    _build_validator,
    _loudness_options,
    _pass_cost_gate,
    _pause_options,
    _render_multivoice_cli,
    _voices_dir_option,
)


@main.command()
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--include-item",
    "include_items",
    multiple=True,
    help="Force-include a section the front/back-matter heuristic would skip "
    "(EPUB: substring of a spine item's file name or id; PDF: substring of a chapter title).",
)
@click.option(
    "--exclude-item",
    "exclude_items",
    multiple=True,
    help="Force-exclude a section (EPUB: substring of a spine item's file name or id; "
    "PDF: substring of a chapter title).",
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
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root where extracted cover art lands (default: settings.output_dir).",
)
def ingest(
    book_path: Path,
    include_items: tuple[str, ...],
    exclude_items: tuple[str, ...],
    split_level: int,
    books_dir: Path | None,
    output_dir: Path | None,
) -> None:
    """Ingest an EPUB or PDF into normalized JSON (books/{book_id}/normalized.json)."""
    from seiyuu.ingest import IngestError, extract_cover_art, parse_book, write_normalized
    from seiyuu.settings import get_settings

    try:
        result = parse_book(
            book_path,
            include_items=include_items,
            exclude_items=exclude_items,
            split_level=split_level,
        )
    except IngestError as exc:
        raise click.ClickException(str(exc)) from exc

    out_path = write_normalized(result.book, books_dir or get_settings().books_dir)
    cover_path = extract_cover_art(result, output_dir or get_settings().output_dir)

    meta = result.book.book_meta
    n_blocks = sum(len(c.blocks) for c in result.book.chapters)
    click.echo(f"book_id:  {meta.book_id}")
    click.echo(f"title:    {meta.title} — {', '.join(meta.authors) or 'unknown author'}")
    click.echo(f"chapters: {len(result.book.chapters)} ({n_blocks} blocks)")
    for name in result.skipped_items:
        click.echo(f"skipped spine item: {name}")
    for section in result.dropped_sections:
        click.echo(f"dropped section:    {section}")
    if cover_path is not None:
        click.echo(f"cover:    {cover_path} (extracted from the book)")
    click.echo(f"wrote: {out_path}")


def _convert_multivoice(
    cfg,
    book,
    attr_book_dir: Path,
    output_dir: Path | None,
    voices_dir: Path | None,
    chapter_indices: tuple[int, ...],
    *,
    narrator_voice_id: str | None,
    thought_voice_id: str | None,
    accent: str,
    hybrid: bool | None,
    confirm_cost: bool = False,
):
    """convert --multivoice: attribute -> auto-assign draft voices -> multi-voice render."""
    from seiyuu.attribute import AttributionError
    from seiyuu.gpu import GpuBusyError
    from seiyuu.services import ServiceError, draft_assignment, run_attribution, save_assignment
    from seiyuu.voices import VoiceLibrary

    click.echo("== attribute ==")
    try:
        report = run_attribution(
            book,
            attr_book_dir,
            cfg=cfg,
            use_hybrid=hybrid,
            chapters=chapter_indices,
            progress=click.echo,
        )  # Ollama VRAM freed inside, before the TTS engine loads (GPU discipline)
    except (AttributionError, GpuBusyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    flagged = f", {len(report.flagged)} flagged" if report.flagged else ""
    click.echo(f"{len(report.registry.characters)} characters{flagged}")

    click.echo("== assign ==")
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    try:
        assignment = draft_assignment(
            report,
            lib,
            narrator_voice_id=narrator_voice_id,
            thought_voice_id=thought_voice_id,
            accent=accent,
            default_preset=cfg.kokoro_default_voice,
        )
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    save_assignment(assignment, output_dir or cfg.output_dir)
    click.echo(
        f"narrator {assignment.narrator_voice_id}, {len(assignment.assignments)} character voices"
    )

    click.echo("== render (multi-voice) ==")
    _render_multivoice_cli(
        cfg, book, attr_book_dir, output_dir, voices_dir, chapter_indices,
        confirm_cost=confirm_cost,
    )  # fmt: skip


@main.command()
@click.argument("book_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
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
@click.option(
    "--multivoice",
    is_flag=True,
    help="Multi-voice: attribute speakers (local LLM), auto-assign draft voices, then render "
    "per character (ignores --engine/--voice/--speed/--seed).",
)
@click.option(
    "--narrator", "narrator_voice_id", default=None, help="Multi-voice narration voice id."
)
@click.option(
    "--thought", "thought_voice_id", default=None, help="Multi-voice interior-thought voice id."
)
@click.option(
    "--accent", default="a", show_default=True, help="Multi-voice auto-draft accent (a/b)."
)
@click.option(
    "--hybrid/--no-hybrid",
    default=None,
    help="Multi-voice: escalate chunks that fail local attribution to anthropic (paid).",
)
@click.option("--m4b", is_flag=True, help="Also build the chaptered .m4b audiobook.")
@click.option(
    "--confirm-cost",
    is_flag=True,
    help="Authorize paid cloud (ElevenLabs) synthesis without the interactive prompt.",
)
@_voices_dir_option
@_pause_options
@_loudness_options
def convert(
    book_path: Path,
    engine_id: str | None,
    voice: str | None,
    chapter_indices: tuple[int, ...],
    speed: float,
    seed: int,
    yes: bool,
    books_dir: Path | None,
    output_dir: Path | None,
    multivoice: bool,
    narrator_voice_id: str | None,
    thought_voice_id: str | None,
    accent: str,
    hybrid: bool | None,
    m4b: bool,
    confirm_cost: bool,
    voices_dir: Path | None,
    **pause_overrides,
) -> None:
    """Full pipeline: EPUB/PDF -> normalized JSON -> render -> chapter MP3s."""
    from seiyuu.assemble import AssembleError, assemble_book, master_book
    from seiyuu.engines import SynthesisError, get_engine, voices_dir_kwargs
    from seiyuu.ingest import IngestError, parse_book, write_normalized
    from seiyuu.render import RenderError, render_book
    from seiyuu.render.gate import FULL_RENDER_CONFIRM_BLOCKS
    from seiyuu.settings import get_settings

    cfg = get_settings()

    click.echo("== ingest ==")
    try:
        ingest_result = parse_book(book_path)
    except IngestError as exc:
        raise click.ClickException(str(exc)) from exc
    book = ingest_result.book
    book_books_dir = books_dir or cfg.books_dir
    write_normalized(book, book_books_dir)
    click.echo(f"{book.book_meta.book_id}: {len(book.chapters)} chapters")

    wanted = set(chapter_indices)
    speakable = sum(
        1
        for ci, c in enumerate(book.chapters, start=1)
        for b in c.blocks
        if b.is_speakable and (not wanted or ci in wanted)
    )
    if not chapter_indices and speakable > FULL_RENDER_CONFIRM_BLOCKS and not yes:
        from seiyuu.duration import estimate_runtime_seconds, format_hms

        runtime = format_hms(estimate_runtime_seconds(book, wpm=cfg.narration_wpm))
        click.confirm(
            f"Full-book render: {speakable} segments, roughly {runtime} of audio to "
            f"synthesize. Continue?",
            abort=True,
        )

    book_dir = (output_dir or cfg.output_dir) / book.book_meta.book_id
    if multivoice:
        _convert_multivoice(
            cfg,
            book,
            book_books_dir / book.book_meta.book_id,
            output_dir,
            voices_dir,
            chapter_indices,
            narrator_voice_id=narrator_voice_id,
            thought_voice_id=thought_voice_id,
            accent=accent,
            hybrid=hybrid,
            confirm_cost=confirm_cost,
        )
    else:
        click.echo("== render ==")
        from seiyuu.normalize.lexicon import load_compiled_lexicon
        from seiyuu.render import estimate_render_cost_single
        from seiyuu.voices import VoiceLibrary, VoiceLibraryError

        lib = VoiceLibrary(voices_dir or cfg.voices_dir)
        lexicon = load_compiled_lexicon(book_books_dir / book.book_meta.book_id)
        try:
            single_engine_id = engine_id or cfg.tts_engine
            extra = voices_dir_kwargs(single_engine_id, lib.voices_dir)
            engine = get_engine(single_engine_id, **extra)
            single_voice = voice or cfg.kokoro_default_voice
            est = estimate_render_cost_single(
                book, engine, single_voice, book_dir,
                settings={"speed": speed}, seed=seed, chapters=chapter_indices, library=lib,
                lexicon=lexicon,
            )  # fmt: skip
            approved_usd = _pass_cost_gate(
                cfg, est,
                book_id=book.book_meta.book_id,
                chapters=chapter_indices,
                assignment_hash=None,
                confirm_cost=confirm_cost,
                cost_token=None,
            )  # fmt: skip
            render_result = render_book(
                book,
                engine,
                single_voice,
                book_dir,
                settings={"speed": speed},
                seed=seed,
                chapters=chapter_indices,
                progress=click.echo,
                library=lib,  # consent gate for cloned voices on the single-voice path
                validator=_build_validator(cfg),
                validation_max_retries=cfg.validation_max_retries,
                allow_paid=approved_usd is not None,
                max_paid_usd=approved_usd,
                lexicon=lexicon,
            )
        except (RenderError, SynthesisError, ValueError, VoiceLibraryError) as exc:
            raise click.ClickException(str(exc)) from exc
        msg = f"{render_result.synthesized} synthesized, {render_result.cache_hits} from cache"
        if render_result.validation_failures:
            msg += f", {render_result.validation_failures} failed validation"
        click.echo(msg)

    click.echo("== assemble ==")
    try:
        assemble_result = assemble_book(
            book_dir,
            pauses=_build_pauses(**pause_overrides),
            loudness=_build_loudness(cfg, **pause_overrides),
            progress=click.echo,
        )
    except AssembleError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"done: {len(assemble_result.mp3_paths)} chapter MP3s, "
        f"{assemble_result.total_seconds / 60:.1f} min total -> {book_dir / 'chapters'}"
    )

    if m4b:
        click.echo("== master ==")
        try:
            master_result = master_book(
                book_dir,
                pauses=_build_pauses(**pause_overrides),
                loudness=_build_loudness(cfg, **pause_overrides),
                progress=click.echo,
            )
        except AssembleError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(
            f"done: {master_result.m4b_path.name} — {master_result.chapters} chapters "
            f"-> {master_result.m4b_path}"
        )


@main.command()
@click.argument("book_id")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.option(
    "--confirm-paid",
    is_flag=True,
    help="Approve discarding paid cloud (ElevenLabs/Fish) renders that cost real money to "
    "reproduce; REQUIRED to delete a book that has any.",
)
@click.option(
    "--books-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Normalized books root (default: settings.books_dir).",
)  # fmt: skip
@click.option(
    "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Render output root (default: settings.output_dir).",
)  # fmt: skip
def delete(
    book_id: str, yes: bool, confirm_paid: bool, books_dir: Path | None, output_dir: Path | None
) -> None:
    """Delete a book: purge output/{id} and books/{id} on disk and reap its terminal job
    rows. Refused while any job for the book is queued or running. Paid cloud renders are
    never discarded without --confirm-paid. The shared voice library and the global jobs.db
    file are left untouched."""
    from seiyuu.repository import JobState, JobStore, resolve_book_id
    from seiyuu.repository.books import RepositoryError, delete_book_trees
    from seiyuu.repository.jobs import JOBS_DB_NAME
    from seiyuu.services.deletion import compute_purge_manifest
    from seiyuu.settings import get_settings

    cfg = get_settings()
    if books_dir is not None or output_dir is not None:
        cfg = cfg.model_copy(
            update={
                "books_dir": books_dir or cfg.books_dir,
                "output_dir": output_dir or cfg.output_dir,
            }
        )
    try:
        resolved = resolve_book_id(book_id, books_dir=cfg.books_dir, output_dir=cfg.output_dir)
    except RepositoryError as exc:
        raise click.ClickException(str(exc)) from exc

    store = JobStore(cfg.data_dir / JOBS_DB_NAME)
    live = store.list_jobs(book_id=resolved, states=[JobState.QUEUED, JobState.RUNNING])
    if live:
        raise click.ClickException(
            f"a {live[0].kind.value} job for {resolved!r} is {live[0].state.value}; cancel "
            "or wait for it before deleting the book"
        )

    summary = compute_purge_manifest(cfg, resolved)
    roots = [
        f"{name}/{resolved}"
        for name, exists in (("output", summary.output_exists), ("books", summary.books_exists))
        if exists
    ]
    click.echo(f"book:  {resolved}")
    click.echo(f"purge: {', '.join(roots) or '(no on-disk trees)'}")
    paid = summary.paid
    if paid.paid_segment_count > 0:
        click.echo(
            f"PAID:  {paid.paid_segment_count} cloud segment(s) via {', '.join(paid.engines)} "
            f"(voices: {', '.join(paid.paid_voice_ids)}) will be discarded and re-bill if "
            "re-rendered"
        )
        if not confirm_paid:
            raise click.ClickException(
                "this book has paid cloud renders; re-run with --confirm-paid to approve "
                "discarding them"
            )

    if not yes:
        click.confirm(f"Permanently delete book {resolved!r} and all its artifacts?", abort=True)

    result = delete_book_trees(resolved, books_dir=cfg.books_dir, output_dir=cfg.output_dir)
    if result.survivors:
        raise click.ClickException(
            f"book {resolved!r} was only partially deleted; could not remove "
            f"{', '.join(result.survivors)} (a file may be open) — retry after closing them"
        )
    jobs_deleted = store.delete_jobs_for_book(resolved)
    # Drop the deleted book from any series membership so it leaves no dangling id behind.
    from seiyuu.voices import drop_book_everywhere

    dropped_from = drop_book_everywhere(cfg.data_dir, resolved)
    click.echo(
        f"deleted {resolved}: "
        f"output={'removed' if result.output_removed else 'absent'}, "
        f"books={'removed' if result.books_removed else 'absent'}, "
        f"job rows removed={jobs_deleted}, paid segments discarded={paid.paid_segment_count}"
        + (f", series membership removed from {len(dropped_from)}" if dropped_from else "")
    )
