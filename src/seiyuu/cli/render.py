"""Render commands: render, render-mode, estimate, estimate-cost, and validation reporting."""

from pathlib import Path

import click

from seiyuu.cli import main
from seiyuu.cli.common import (
    _build_validator,
    _pass_cost_gate,
    _render_multivoice_cli,
    _resolve_book_dir,
    _voices_dir_option,
)


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
    "--multivoice",
    is_flag=True,
    help="Multi-voice: render per-character voices from attribution.json + assignments.json "
    "(ignores --engine/--voice/--speed/--seed; each voice carries its own).",
)
@click.option(
    "--confirm-cost",
    is_flag=True,
    help="Authorize paid cloud (ElevenLabs) synthesis without the interactive prompt.",
)
@click.option(
    "--apply-emotion/--no-apply-emotion",
    default=None,
    help="Apply per-segment emotion (F2) to render settings for supported engines "
    "(multivoice only; default from settings.apply_emotion). Off keeps renders byte-identical.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-render: bypass the segment cache for the in-scope chapters, re-synthesizing "
    "and overwriting even a cache HIT. Pair with --chapter to redo one chapter fresh.",
)
@click.option(
    "--cost-token",
    default=None,
    help="Signed cost token from `seiyuu estimate-cost --token`; refused if anything "
    "about the paid work drifted since it was issued.",
)
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
@_voices_dir_option
def render(
    book_id: str,
    engine_id: str | None,
    voice: str | None,
    chapter_indices: tuple[int, ...],
    speed: float,
    seed: int,
    multivoice: bool,
    confirm_cost: bool,
    apply_emotion: bool | None,
    force: bool,
    cost_token: str | None,
    books_dir: Path | None,
    output_dir: Path | None,
    voices_dir: Path | None,
) -> None:
    """Render a book single- or multi-voice: cached segment WAVs + manifest.json."""
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, "normalized.json", "Run `seiyuu ingest` first."
    )
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )

    if multivoice:
        _render_multivoice_cli(
            cfg, book, book_dir, output_dir, voices_dir, chapter_indices,
            confirm_cost=confirm_cost, cost_token=cost_token, apply_emotion=apply_emotion,
            force=force,
        )  # fmt: skip
        return

    from seiyuu.engines import SynthesisError, get_engine, voices_dir_kwargs
    from seiyuu.normalize.lexicon import load_compiled_lexicon
    from seiyuu.render import RenderError, estimate_render_cost_single, render_book
    from seiyuu.voices import VoiceLibrary, VoiceLibraryError

    engine_id = engine_id or cfg.tts_engine
    voice = voice or cfg.kokoro_default_voice
    out_book_dir = (output_dir or cfg.output_dir) / book.book_meta.book_id
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    lexicon = load_compiled_lexicon(book_dir)  # F3: same lexicon for estimate + render
    try:
        # cloning engines resolve clones from the library dir; the consent gate in render_book
        # needs the same library, or --voices-dir clones would render ungated
        engine = get_engine(engine_id, **voices_dir_kwargs(engine_id, lib.voices_dir))
        # pre-flight: single-voice paid renders get the same estimate + ceiling + approval
        # gate as multivoice (the M5 gap: --confirm-cost used to authorize blind)
        est = estimate_render_cost_single(
            book, engine, voice, out_book_dir,
            settings={"speed": speed}, seed=seed, chapters=chapter_indices, library=lib,
            lexicon=lexicon, force=force,
        )  # fmt: skip
        approved_usd = _pass_cost_gate(
            cfg, est,
            book_id=book.book_meta.book_id,
            chapters=chapter_indices,
            assignment_hash=None,
            confirm_cost=confirm_cost,
            cost_token=cost_token,
        )  # fmt: skip
        result = render_book(
            book,
            engine,
            voice,
            out_book_dir,
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
            force=force,
        )
    # SynthesisError covers indextts2's model_version raising on missing checkpoints (the only
    # engine whose model_version can fail) — surface it as a clean click error, not a traceback.
    except (RenderError, SynthesisError, ValueError, VoiceLibraryError) as exc:
        raise click.ClickException(str(exc)) from exc

    minutes = result.total_audio_seconds / 60
    click.echo(
        f"done: {result.synthesized} segments synthesized, "
        f"{result.cache_hits} from cache, {minutes:.1f} min of audio"
    )
    if result.validation_failures:
        click.echo(f"  {result.validation_failures} segment(s) failed whisper validation")
    click.echo(f"manifest: {result.manifest_path}")


@main.command("render-mode")
@click.argument("book_id")
@click.argument("mode", type=click.Choice(["single", "multi"]))
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root (default: settings.output_dir).",
)
def render_mode(book_id: str, mode: str, output_dir: Path | None) -> None:
    """Switch the ACTIVE render (manifest.json — what listen/assemble/master read) to the
    chosen mode's archived manifest. A pure file switch: no synthesis, no cache touch;
    refused while a render/assemble/master job for the book is live."""
    from seiyuu.repository import JobStore, RepositoryError, resolve_book_id
    from seiyuu.repository.jobs import JOBS_DB_NAME
    from seiyuu.services import ServiceError
    from seiyuu.services.render_mode import activate_render_mode
    from seiyuu.settings import get_settings

    cfg = get_settings()
    out_root = output_dir or cfg.output_dir
    try:
        resolved = resolve_book_id(book_id, books_dir=cfg.books_dir, output_dir=out_root)
    except RepositoryError as exc:
        raise click.ClickException(str(exc)) from exc
    store = JobStore(cfg.data_dir / JOBS_DB_NAME)
    try:
        result = activate_render_mode(out_root, resolved, mode, store=store)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    label = "single-voice" if result.mode == "single" else "multivoice"
    state = "now active" if result.changed else "already active"
    click.echo(f"{resolved}: {label} render {state} ({result.chapters} chapter(s))")


@main.command()
@click.argument("book_id")
@click.option("--wpm", type=float, default=None, help="Narration pace (default from settings).")
@click.option(
    "--chapter",
    "chapter_indices",
    multiple=True,
    type=int,
    help="Estimate only these 1-based chapters (repeatable). Default: all.",
)
@click.option(
    "--books-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where normalized books live (default: settings.books_dir).",
)
def estimate(
    book_id: str, wpm: float | None, chapter_indices: tuple[int, ...], books_dir: Path | None
) -> None:
    """Estimate audiobook runtime from word count (pre-render)."""
    from seiyuu.duration import estimate_runtime_seconds, format_hms
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, "normalized.json", "Run `seiyuu ingest` first."
    )
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )
    pace = wpm or cfg.narration_wpm
    seconds = estimate_runtime_seconds(book, wpm=pace, chapters=chapter_indices)
    scope = f"chapters {sorted(chapter_indices)}" if chapter_indices else "whole book"
    click.echo(f"estimated runtime ({scope}): {format_hms(seconds)} at {pace:.0f} wpm")


def _load_multivoice_inputs(cfg, book_id, books_dir, output_dir, voices_dir, chapter_indices):
    """Shared loader for the multi-voice cost/render path: returns
    (report, book, lib, assignment, out_book_dir, lexicon)."""
    from seiyuu.attribute import ATTRIBUTION_NAME
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.normalize.lexicon import load_compiled_lexicon
    from seiyuu.services import ServiceError, load_assignment, load_report
    from seiyuu.voices import VoiceLibrary

    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    # the EFFECTIVE report (manual edits applied): cost estimation and the render must
    # both see the same segments or the gate's fingerprint would never match
    try:
        report, edit_warnings = load_report(book_dir)
        assignment = load_assignment(output_dir or cfg.output_dir, report.book_id)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    for warning in edit_warnings:
        # a skipped edit means the paid work being estimated does NOT contain the
        # user's fix — say so before they approve a quote for it
        click.echo(f"edit overlay: {warning}")
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )
    out_book_dir = (output_dir or cfg.output_dir) / report.book_id
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    lexicon = load_compiled_lexicon(book_dir)  # F3: same lexicon for estimate + render
    return report, book, lib, assignment, out_book_dir, lexicon


@main.command("estimate-cost")
@click.argument("book_id")
@click.option(
    "--chapter", "chapter_indices", multiple=True, type=int,
    help="Estimate only these 1-based chapters (repeatable). Default: all.",
)  # fmt: skip
@click.option(
    "--books-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Where attributed books live (default: settings.books_dir).",
)  # fmt: skip
@click.option(
    "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Render output root (default: settings.output_dir).",
)  # fmt: skip
@click.option(
    "--token", "issue_token", is_flag=True,
    help="Also print a signed cost token for `render --cost-token` (short-lived, "
    "single-use; refused on any drift in the paid work).",
)  # fmt: skip
@click.option(
    "--voice", default=None,
    help="Single-voice mode: estimate `render --voice X` instead of --multivoice "
    "(pass the same --engine/--speed/--seed you will render with).",
)  # fmt: skip
@click.option("--engine", "engine_id", default=None, help="Single-voice TTS engine.")
@click.option("--speed", default=1.0, show_default=True, help="Single-voice speed multiplier.")
@click.option("--seed", default=41172, show_default=True, help="Single-voice synthesis seed.")
@_voices_dir_option
def estimate_cost(
    book_id: str,
    chapter_indices: tuple[int, ...],
    books_dir: Path | None,
    output_dir: Path | None,
    issue_token: bool,
    voice: str | None,
    engine_id: str | None,
    speed: float,
    seed: int,
    voices_dir: Path | None,
) -> None:
    """Estimate the USD cost of a render (only uncached, paid segments cost money).

    Multi-voice by default; pass --voice/--engine for the single-voice render's estimate.
    The estimate must be built with the same parameters the render will use, or a token
    issued from it will (correctly) refuse.
    """
    from seiyuu.engines import SynthesisError
    from seiyuu.settings import get_settings

    cfg = get_settings()
    if voice or engine_id:
        # single-voice mode: mirrors the render command's defaults exactly
        from seiyuu.engines import get_engine
        from seiyuu.ingest.models import NormalizedBook
        from seiyuu.normalize.lexicon import load_compiled_lexicon
        from seiyuu.render import estimate_render_cost_single
        from seiyuu.voices import VoiceLibrary

        book_dir = _resolve_book_dir(
            books_dir or cfg.books_dir, book_id, "normalized.json", "Run `seiyuu ingest` first."
        )
        book = NormalizedBook.model_validate_json(
            (book_dir / "normalized.json").read_text(encoding="utf-8")
        )
        try:
            est = estimate_render_cost_single(
                book,
                get_engine(engine_id or cfg.tts_engine),
                voice or cfg.kokoro_default_voice,
                (output_dir or cfg.output_dir) / book.book_meta.book_id,
                settings={"speed": speed},
                seed=seed,
                chapters=chapter_indices,
                library=VoiceLibrary(voices_dir or cfg.voices_dir),
                lexicon=load_compiled_lexicon(book_dir),
            )
        # indextts2's model_version raises SynthesisError on missing checkpoints — clean error.
        except SynthesisError as exc:
            raise click.ClickException(str(exc)) from exc
        assignment_hash = None
    else:
        from seiyuu.render import estimate_render_cost, hash_assignment

        report, book, lib, assignment, out_book_dir, lexicon = _load_multivoice_inputs(
            cfg, book_id, books_dir, output_dir, voices_dir, chapter_indices
        )
        try:
            est = estimate_render_cost(
                report, book, lib, assignment, out_book_dir,
                chapters=chapter_indices, lexicon=lexicon,
            )  # fmt: skip
        # a voice assigned to indextts2 without checkpoints raises here — clean error.
        except SynthesisError as exc:
            raise click.ClickException(str(exc)) from exc
        assignment_hash = hash_assignment(assignment)
    click.echo(
        f"estimated cost: ${est.total_usd:.2f} over {est.paid_segments} paid segment(s); "
        f"{est.cached_segments} cached, {est.free_segments} free (local)"
    )
    if issue_token:
        from seiyuu.render import CostGateError, issue_quote

        try:
            quote = issue_quote(
                est,
                book_id=book.book_meta.book_id,
                chapters=chapter_indices,
                assignment_hash=assignment_hash,
                max_usd=cfg.render_max_usd,
                ttl_seconds=cfg.cost_quote_ttl_seconds,
                data_dir=cfg.data_dir,
            )
        except CostGateError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"cost token (valid {cfg.cost_quote_ttl_seconds // 60} min, single-use):")
        click.echo(quote.encode())


@main.command()
@click.argument("book_id")
@click.option(
    "--all", "show_all", is_flag=True, help="List every validated segment, not just failures."
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root (default: settings.output_dir).",
)
def validate(book_id: str, show_all: bool, output_dir: Path | None) -> None:
    """Report whisper validation results recorded in a render (failures, scores, transcripts)."""
    import textwrap

    from seiyuu.render import MANIFEST_NAME, RenderManifest
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        output_dir or cfg.output_dir, book_id, MANIFEST_NAME, "Run `seiyuu render` first."
    )
    manifest = RenderManifest.model_validate_json(
        (book_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    validated = [
        (c.index, s) for c in manifest.chapters for s in c.segments if s.validation is not None
    ]
    if not validated:
        click.echo(
            "no validated segments — validation runs only for LLM-style engines (e.g. chatterbox)"
        )
        return
    failures = [(ci, s) for ci, s in validated if not s.validation.ok]
    click.echo(f"validated segments: {len(validated)}, failures: {len(failures)}")
    for ci, seg in validated if show_all else failures:
        v = seg.validation
        mark = "" if v.ok else "  FAIL"
        click.echo(f"  ch{ci} {seg.block_id} voice={seg.voice_id} score={v.score}{mark}")
        if not v.ok:
            click.echo(f"      expected: {textwrap.shorten(v.expected, 70)}")
            click.echo(f"      heard:    {textwrap.shorten(v.transcript, 70)}")
