"""Casting commands: the character→voice assignment and cross-book series voice links."""

from pathlib import Path

import click

from seiyuu.cli import main
from seiyuu.cli.common import _resolve_book_dir, _voices_dir_option


@main.command()
@click.argument("book_id")
@click.option(
    "--narrator",
    "narrator_voice_id",
    default=None,
    help="Narration voice id (default: auto preset).",
)
@click.option(
    "--thought",
    "thought_voice_id",
    default=None,
    help="Interior-thought voice id (default: speaker's own).",
)
@click.option(
    "--accent", default="a", show_default=True, help="Auto-draft accent for character voices."
)
@click.option(
    "--stage",
    type=click.Choice(["draft", "final"]),
    default="draft",
    show_default=True,
    help="Assignment stage (final typically maps characters to cloud voices via --map).",
)
@click.option(
    "--map",
    "maps",
    multiple=True,
    help="Override a character's voice: CHARACTER_ID=VOICE_ID (repeatable).",
)
@click.option(
    "--strategy",
    type=click.Choice(["hash", "smart"]),
    default="hash",
    show_default=True,
    help="Auto-caster: 'hash' (legacy per-character blend) or 'smart' (book-level, no two "
    "characters share a voice).",
)
@click.option(
    "--recast",
    is_flag=True,
    default=False,
    help="smart only: overwrite existing {char}_auto voices to apply the new cast "
    "(re-renders those voices' segments).",
)
@click.option(
    "--use-llm",
    is_flag=True,
    default=False,
    help="smart only: run the opt-in Layer-2 LLM caster for per-character voice-trait "
    "preferences (biases the tie-breaker; distinctness stays guaranteed).",
)
@click.option(
    "--cast-provider",
    default=None,
    help="LLM caster provider: 'local' (free) or 'anthropic' (PAID). Default: settings.",
)
@click.option(
    "--confirm-paid",
    is_flag=True,
    default=False,
    help="Required to run the anthropic (paid) LLM caster.",
)
@click.option(
    "--books-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where attributed books live (default: settings.books_dir).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where assignments.json is written (default: settings.output_dir).",
)
@_voices_dir_option
def assign(
    book_id: str,
    narrator_voice_id: str | None,
    thought_voice_id: str | None,
    accent: str,
    stage: str,
    maps: tuple[str, ...],
    strategy: str,
    recast: bool,
    use_llm: bool,
    cast_provider: str | None,
    confirm_paid: bool,
    books_dir: Path | None,
    output_dir: Path | None,
    voices_dir: Path | None,
) -> None:
    """Build a character→voice assignment (auto-drafts locals; --map overrides, e.g. to cloud)."""
    from seiyuu.attribute import ATTRIBUTION_NAME
    from seiyuu.gpu import GpuBusyError
    from seiyuu.services import ServiceError, draft_assignment, load_report, save_assignment
    from seiyuu.services.llm_advisory import resolve_advisory, run_cast_hints
    from seiyuu.settings import get_settings
    from seiyuu.voices import AssignmentStage, VoiceLibrary

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    overrides: dict[str, str] = {}
    for entry in maps:
        char_id, _, vid = entry.partition("=")
        if not vid:
            raise click.ClickException(f"bad --map {entry!r}; expected CHARACTER_ID=VOICE_ID")
        overrides[char_id] = vid
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    try:
        report, edit_warnings = load_report(book_dir)
        # F4: opt-in LLM caster. Only meaningful on the smart path; the paid gate mirrors
        # attribution — anthropic needs --confirm-paid + the key, local is free-but-explicit.
        trait_hints = None
        if use_llm and strategy == "smart":
            resolved = resolve_advisory(cfg, cfg.cast_provider, cfg.cast_model, cast_provider)
            if resolved.is_paid:
                if not confirm_paid:
                    raise click.ClickException(
                        f"cast provider {resolved.provider_id!r} is a PAID Anthropic call; "
                        "re-run with --confirm-paid to approve the spend."
                    )
                if not cfg.anthropic_api_key:
                    raise click.ClickException(
                        "ANTHROPIC_API_KEY not set; required for the anthropic LLM caster"
                    )
            castable = [c for c in report.registry.characters if c.id not in overrides]
            trait_hints = run_cast_hints(cfg, resolved, castable)
        elif use_llm:
            click.echo("--use-llm has no effect on the 'hash' strategy; ignoring.")
        assignment = draft_assignment(
            report,
            lib,
            narrator_voice_id=narrator_voice_id,
            thought_voice_id=thought_voice_id,
            accent=accent,
            default_preset=cfg.kokoro_default_voice,
            stage=AssignmentStage(stage),
            overrides=overrides,
            strategy=strategy,
            recast=recast,
            trait_hints=trait_hints,
        )
    except (GpuBusyError, ServiceError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    for warning in edit_warnings:
        click.echo(f"edit overlay: {warning}")

    path = save_assignment(assignment, output_dir or cfg.output_dir)

    click.echo(f"stage: {assignment.stage.value}  narrator: {assignment.narrator_voice_id}")
    for char in report.registry.characters:
        click.echo(f"  {char.canonical_name} [{char.id}] -> {assignment.assignments[char.id]}")
    if assignment.thought_voice_id:
        click.echo(f"thoughts: {assignment.thought_voice_id}")
    click.echo(f"wrote: {path}")


@main.group()
def series() -> None:
    """Series / library voice consistency (data/series.json): reuse a character's voice across
    books in a series. Inheritance flows in as `assign --map` overrides — no render change."""


def _series_report_and_assignment(cfg, book_id: str):
    """(report, assignment) for a book contributing its cast to a series (create/save-cast)."""
    from seiyuu.attribute import ATTRIBUTION_NAME
    from seiyuu.services import load_assignment, load_report

    book_dir = _resolve_book_dir(
        cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    report, _warnings = load_report(book_dir)
    assignment = load_assignment(cfg.output_dir, report.book_id)
    return report, assignment


def _series_report(cfg, book_id: str):
    """Attributed report for a book inheriting from a series (suggest-links/overrides)."""
    from seiyuu.attribute import ATTRIBUTION_NAME
    from seiyuu.services import load_report

    book_dir = _resolve_book_dir(
        cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    report, _warnings = load_report(book_dir)
    return report


def _get_series_or_fail(registry, series_id: str):
    series_obj = registry.get(series_id)
    if series_obj is None:
        raise click.ClickException(f"series {series_id!r} not found")
    return series_obj


@series.command("create")
@click.argument("name")
@click.argument("book_id")
@click.option("--series-id", default=None, help="Explicit id (default: slug + hex).")
def series_create(name: str, book_id: str, series_id: str | None) -> None:
    """Create a series SEEDED from a book's cast (canonical_name -> voice_id)."""
    from seiyuu.services import ServiceError
    from seiyuu.settings import get_settings
    from seiyuu.voices import (
        Series,
        load_registry,
        make_series_id,
        save_registry,
        seed_voice_links,
    )

    cfg = get_settings()
    try:
        report, assignment = _series_report_and_assignment(cfg, book_id)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    registry = load_registry(cfg.data_dir)
    sid = series_id or make_series_id(name)
    if registry.get(sid) is not None:
        raise click.ClickException(f"series {sid!r} already exists")
    obj = Series(
        series_id=sid,
        name=name.strip(),
        book_ids=[report.book_id],
        voice_links=seed_voice_links(report, assignment),
    )
    registry.upsert(obj)
    save_registry(cfg.data_dir, registry)
    click.echo(f"created series {sid!r} ({obj.name}) with {len(obj.voice_links)} voice link(s)")


@series.command("list")
def series_list() -> None:
    """List every declared series."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import load_registry

    registry = load_registry(get_settings().data_dir)
    if not registry.series:
        click.echo("(no series)")
        return
    for s in registry.series:
        click.echo(
            f"{s.series_id}  {s.name!r}  books={len(s.book_ids)}  links={len(s.voice_links)}"
        )


@series.command("show")
@click.argument("series_id")
def series_show(series_id: str) -> None:
    """Print a series' books and voice links."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import load_registry

    registry = load_registry(get_settings().data_dir)
    s = _get_series_or_fail(registry, series_id)
    click.echo(f"{s.series_id}  {s.name!r}")
    click.echo(f"books: {', '.join(s.book_ids) or '(none)'}")
    if not s.voice_links:
        click.echo("links: (none)")
        return
    click.echo("links:")
    for key, vid in sorted(s.voice_links.items()):
        click.echo(f"  {key} -> {vid}")


@series.command("add-book")
@click.argument("series_id")
@click.argument("book_id")
def series_add_book(series_id: str, book_id: str) -> None:
    """Attach a book to the series (does NOT learn its cast — use save-cast for that)."""
    from seiyuu.repository import RepositoryError, resolve_book_id
    from seiyuu.settings import get_settings
    from seiyuu.voices import load_registry, save_registry

    cfg = get_settings()
    try:
        resolved = resolve_book_id(book_id, books_dir=cfg.books_dir, output_dir=cfg.output_dir)
    except RepositoryError as exc:
        raise click.ClickException(str(exc)) from exc
    registry = load_registry(cfg.data_dir)
    s = _get_series_or_fail(registry, series_id)
    if resolved not in s.book_ids:
        s.book_ids.append(resolved)
        registry.upsert(s)
        save_registry(cfg.data_dir, registry)
    click.echo(f"series {series_id!r} books: {', '.join(s.book_ids)}")


@series.command("remove-book")
@click.argument("series_id")
@click.argument("book_id")
@click.option("--prune", is_flag=True, help="Also drop links whose voice no longer exists.")
def series_remove_book(series_id: str, book_id: str, prune: bool) -> None:
    """Drop a book from the series' membership."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceLibrary, load_registry, prune_dangling_links, save_registry

    cfg = get_settings()
    registry = load_registry(cfg.data_dir)
    s = _get_series_or_fail(registry, series_id)
    s.book_ids = [b for b in s.book_ids if b != book_id]
    pruned: list[str] = []
    if prune:
        pruned = prune_dangling_links(s, VoiceLibrary(cfg.voices_dir))
    registry.upsert(s)
    save_registry(cfg.data_dir, registry)
    extra = f", pruned {len(pruned)} dangling link(s)" if pruned else ""
    click.echo(f"series {series_id!r} books: {', '.join(s.book_ids) or '(none)'}{extra}")


@series.command("suggest-links")
@click.argument("series_id")
@click.argument("book_id")
def series_suggest_links(series_id: str, book_id: str) -> None:
    """Name-match a joining book's characters against the series' voice links (for review)."""
    from seiyuu.services import ServiceError
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceLibrary, load_registry, suggest_links

    cfg = get_settings()
    registry = load_registry(cfg.data_dir)
    s = _get_series_or_fail(registry, series_id)
    try:
        report = _series_report(cfg, book_id)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    suggestions = suggest_links(report, s, VoiceLibrary(cfg.voices_dir))
    if not suggestions:
        click.echo("(no cross-book links suggested)")
        return
    for sug in suggestions:
        avail = "" if sug.voice_exists else "  [voice missing — will be skipped]"
        click.echo(f"{sug.character_id} ({sug.canonical_name}) -> {sug.voice_id}{avail}")


@series.command("overrides")
@click.argument("series_id")
@click.argument("book_id")
def series_overrides(series_id: str, book_id: str) -> None:
    """Print resolved inheritance as CHARACTER_ID=VOICE_ID lines (feed to `assign --map`)."""
    from seiyuu.services import ServiceError
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceLibrary, load_registry, resolve_series_overrides

    cfg = get_settings()
    registry = load_registry(cfg.data_dir)
    s = _get_series_or_fail(registry, series_id)
    try:
        report = _series_report(cfg, book_id)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    overrides = resolve_series_overrides(report, s, VoiceLibrary(cfg.voices_dir))
    if not overrides:
        click.echo("(no inherited voices)")
        return
    for char_id, vid in sorted(overrides.items()):
        click.echo(f"{char_id}={vid}")


@series.command("save-cast")
@click.argument("series_id")
@click.argument("book_id")
def series_save_cast(series_id: str, book_id: str) -> None:
    """Explicit write-back: fold a book's cast into the series' voice links (last-write-wins)."""
    from seiyuu.services import ServiceError
    from seiyuu.settings import get_settings
    from seiyuu.voices import load_registry, save_cast_to_series, save_registry

    cfg = get_settings()
    registry = load_registry(cfg.data_dir)
    s = _get_series_or_fail(registry, series_id)
    try:
        report, assignment = _series_report_and_assignment(cfg, book_id)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    linked = save_cast_to_series(s, report, assignment)
    if report.book_id not in s.book_ids:
        s.book_ids.append(report.book_id)
    registry.upsert(s)
    save_registry(cfg.data_dir, registry)
    click.echo(f"saved {len(linked)} link(s) to series {series_id!r} ({len(s.voice_links)} total)")


@series.command("unlink")
@click.argument("series_id")
@click.argument("name")
def series_unlink(series_id: str, name: str) -> None:
    """Remove a character's voice link by NAME (case-insensitive)."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import identity_key, load_registry, save_registry

    cfg = get_settings()
    registry = load_registry(cfg.data_dir)
    s = _get_series_or_fail(registry, series_id)
    removed = s.voice_links.pop(identity_key(name), None)
    registry.upsert(s)
    save_registry(cfg.data_dir, registry)
    click.echo(
        f"unlinked {name!r} ({removed})" if removed else f"no link for {name!r} in {series_id!r}"
    )
