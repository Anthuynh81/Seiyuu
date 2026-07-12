"""Seiyuu CLI. Runnable as `seiyuu` or `python -m seiyuu.cli`."""

from pathlib import Path

import click

from seiyuu import __version__
from seiyuu.attribute.models import EmotionLabel


@click.group()
@click.version_option(__version__, prog_name="seiyuu")
def main() -> None:
    """Seiyuu — multi-voice audiobook creator."""


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
def ingest(
    book_path: Path,
    include_items: tuple[str, ...],
    exclude_items: tuple[str, ...],
    split_level: int,
    books_dir: Path | None,
) -> None:
    """Ingest an EPUB or PDF into normalized JSON (books/{book_id}/normalized.json)."""
    from seiyuu.ingest import IngestError, parse_book, write_normalized
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


def _voices_dir_option(fn):
    """Shared --voices-dir flag (the voice library root)."""
    return click.option(
        "--voices-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Voice library root (default: settings.voices_dir).",
    )(fn)


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
            lexicon=lexicon,
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


def _build_validator(cfg):
    """A whisper validator from settings; loads lazily, so Kokoro-only renders never touch it."""
    from seiyuu.validate import Validator

    return Validator(
        model_size=cfg.validation_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.validation_compute_type,
        min_ratio=cfg.validation_min_ratio,
    )


def _pass_cost_gate(cfg, est, *, book_id, chapters, assignment_hash, confirm_cost, cost_token):
    """The M6a money gate: no paid synthesis without the ceiling AND explicit approval.

    Returns the approved paid budget in USD (threaded into the render loop as its hard
    spend cap), or None when nothing paid may run — free renders never engage the gate
    and the per-segment refusal in the pipeline stays as the safety net. A provided
    ``cost_token`` is verified against the FRESH estimate (single-use; any drift refuses
    with the reason); otherwise the ceiling is enforced and approval is
    ``--confirm-cost`` or the interactive prompt.
    """
    if est.total_usd <= 0:
        return None
    from seiyuu.render import CostGateError, CostQuote, check_ceiling, verify_quote

    click.echo(
        f"cost estimate: ${est.total_usd:.2f} over {est.paid_segments} paid segment(s) "
        f"({est.cached_segments} cached)"
    )
    try:
        if cost_token:
            quote = CostQuote.decode(cost_token)
            verify_quote(
                quote,
                book_id=book_id,
                chapters=chapters,
                fingerprint=est.fingerprint,
                assignment_hash=assignment_hash,
                recomputed_total_usd=est.total_usd,
                max_usd=cfg.render_max_usd,
                data_dir=cfg.data_dir,
            )
            return quote.total_usd
        check_ceiling(est.total_usd, cfg.render_max_usd)
    except CostGateError as exc:
        raise click.ClickException(str(exc)) from exc
    if not confirm_cost:
        click.confirm("This makes paid cloud API calls. Continue?", abort=True)
    return est.total_usd


def _render_multivoice_cli(
    cfg, book, book_dir, output_dir, voices_dir, chapter_indices, *,
    confirm_cost=False, cost_token=None, apply_emotion=None,
):  # fmt: skip
    """Shared multi-voice render: load inputs, cost-gate paid engines, render, echo summary."""
    from seiyuu.render import (
        RenderError,
        estimate_render_cost,
        hash_assignment,
        render_book_multivoice,
    )
    from seiyuu.services import ServiceError, load_assignment, load_report
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceLibrary

    book_id = book.book_meta.book_id
    try:
        report, edit_warnings = load_report(book_dir)  # manual edits applied
        assignment = load_assignment(output_dir or get_settings().output_dir, book_id)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    for warning in edit_warnings:
        click.echo(f"edit overlay: {warning}")
    out_book_dir = (output_dir or get_settings().output_dir) / book_id
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    from seiyuu.normalize.lexicon import load_compiled_lexicon

    lexicon = load_compiled_lexicon(book_dir)  # F3: same lexicon for estimate + render
    # F2: resolve the opt-in flag once; estimate AND render MUST use the same value or the cost
    # gate authorizes a different bill than render runs up.
    apply_emotion = cfg.apply_emotion if apply_emotion is None else apply_emotion

    # cost gate: estimate first; paid segments require the ceiling + explicit approval
    est = estimate_render_cost(
        report, book, lib, assignment, out_book_dir, chapters=chapter_indices, lexicon=lexicon,
        apply_emotion=apply_emotion,
    )  # fmt: skip
    approved_usd = _pass_cost_gate(
        cfg, est,
        book_id=book_id,
        chapters=chapter_indices,
        assignment_hash=hash_assignment(assignment),
        confirm_cost=confirm_cost,
        cost_token=cost_token,
    )  # fmt: skip

    try:
        result = render_book_multivoice(
            report,
            book,
            lib,
            assignment,
            out_book_dir,
            chapters=chapter_indices,
            progress=click.echo,
            validator=_build_validator(cfg),
            validation_max_retries=cfg.validation_max_retries,
            allow_paid=approved_usd is not None,
            max_paid_usd=approved_usd,
            cloud_max_slots=cfg.elevenlabs_max_voice_slots,
            lexicon=lexicon,
            apply_emotion=apply_emotion,
        )
    except (RenderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    minutes = result.total_audio_seconds / 60
    click.echo(
        f"done: {result.synthesized} segments synthesized, {result.cache_hits} from cache, "
        f"{len(result.manifest.voices_used)} voices, {minutes:.1f} min of audio"
    )
    if result.validation_failures:
        click.echo(f"  {result.validation_failures} segment(s) failed whisper validation")
    click.echo(f"manifest: {result.manifest_path}")
    return result


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
    """convert --multivoice: attribute → auto-assign draft voices → multi-voice render."""
    from seiyuu.attribute import AttributionError
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
    except (AttributionError, ValueError) as exc:
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
    except (AttributionError, ValueError) as exc:
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
    from seiyuu.services import ServiceError, run_adjudication
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    try:
        report = run_adjudication(book_dir, cfg=cfg, progress=click.echo)
    except (ServiceError, AttributionError, ValueError) as exc:
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


def _edit_books_dir_option(fn):
    return click.option(
        "--books-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Where attributed books live (default: settings.books_dir).",
    )(fn)


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


@main.group()
def lexicon() -> None:
    """Per-book pronunciation dictionary (books/{id}/lexicon.json): fix mispronounced names."""


def _lexicon_book_dir(book_id: str, books_dir: Path | None) -> Path:
    from seiyuu.settings import get_settings

    return _resolve_book_dir(
        books_dir or get_settings().books_dir,
        book_id,
        "normalized.json",
        "Run `seiyuu ingest` first.",
    )


@lexicon.command("show")
@click.argument("book_id")
@_edit_books_dir_option
def lexicon_show(book_id: str, books_dir: Path | None) -> None:
    """Print the book's pronunciation entries."""
    from seiyuu.normalize.lexicon import load_lexicon

    book_dir = _lexicon_book_dir(book_id, books_dir)
    lex = load_lexicon(book_dir, book_id=book_id)
    if not lex.entries:
        click.echo("(no lexicon entries)")
        return
    for entry in lex.entries:
        extra = []
        if entry.ipa:
            extra.append(f"ipa={entry.ipa!r} [kokoro-only]")
        if entry.case_sensitive:
            extra.append("case-sensitive")
        if entry.note:
            extra.append(f"note={entry.note!r}")
        suffix = ("  " + ", ".join(extra)) if extra else ""
        click.echo(f"{entry.term!r} -> {entry.respelling!r}{suffix}")


@lexicon.command("set")
@click.argument("book_id")
@click.option("--term", required=True, help="The word as it appears in the book.")
@click.option("--respelling", required=True, help="Grapheme respelling spoken on every engine.")
@click.option("--ipa", default=None, help="Optional IPA — applied ONLY on the Kokoro profile.")
@click.option("--note", default=None, help="Optional note for your own reference.")
@click.option("--case-sensitive", is_flag=True, help="Match the term's exact capitalization.")
@_edit_books_dir_option
def lexicon_set(
    book_id: str,
    term: str,
    respelling: str,
    ipa: str | None,
    note: str | None,
    case_sensitive: bool,
    books_dir: Path | None,
) -> None:
    """Add or update one pronunciation entry (matched by term, case-insensitively)."""
    from seiyuu.normalize.lexicon import LexiconEntry, load_lexicon, save_lexicon

    book_dir = _lexicon_book_dir(book_id, books_dir)
    lex = load_lexicon(book_dir, book_id=book_id)
    try:
        entry = LexiconEntry(
            term=term,
            respelling=respelling,
            ipa=ipa,
            note=note,
            case_sensitive=case_sensitive,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    kept = [e for e in lex.entries if e.term.casefold() != entry.term.casefold()]
    kept.append(entry)
    lex.entries = kept
    save_lexicon(book_dir, lex)
    click.echo(f"saved {entry.term!r} -> {entry.respelling!r} ({len(lex.entries)} entries)")


@lexicon.command("remove")
@click.argument("book_id")
@click.option("--term", required=True, help="The term to remove (case-insensitive).")
@_edit_books_dir_option
def lexicon_remove(book_id: str, term: str, books_dir: Path | None) -> None:
    """Remove a pronunciation entry by term."""
    from seiyuu.normalize.lexicon import load_lexicon, save_lexicon

    book_dir = _lexicon_book_dir(book_id, books_dir)
    lex = load_lexicon(book_dir, book_id=book_id)
    kept = [e for e in lex.entries if e.term.casefold() != term.casefold()]
    if len(kept) == len(lex.entries):
        raise click.ClickException(f"no lexicon entry for term {term!r}")
    lex.entries = kept
    save_lexicon(book_dir, lex)
    click.echo(f"removed {term!r} ({len(lex.entries)} entries remain)")


@lexicon.command("suggest")
@click.argument("book_id")
@_edit_books_dir_option
def lexicon_suggest(book_id: str, books_dir: Path | None) -> None:
    """Surface likely hard-to-pronounce names (deterministic; free, no LLM)."""
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.normalize.lexicon import load_lexicon, suggest_terms

    book_dir = _lexicon_book_dir(book_id, books_dir)
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )
    lex = load_lexicon(book_dir, book_id=book_id)
    texts = [b.text for c in book.chapters for b in c.blocks if b.is_speakable]
    suggestions = suggest_terms(texts, existing_terms=[e.term for e in lex.entries])
    if not suggestions:
        click.echo("(no candidate terms found)")
        return
    for s in suggestions:
        click.echo(f"{s.term}  (x{s.count})  …{s.sample}…")


@lexicon.command("suggest-ai")
@click.argument("book_id")
@click.option(
    "--term",
    "terms",
    multiple=True,
    help="Term to respell (repeatable). Omit to use the deterministic hard-name suggestions.",
)
@click.option(
    "--provider",
    default=None,
    help="Suggestion provider: 'local' (Ollama, free) or 'anthropic' (PAID). Default: settings.",
)
@click.option(
    "--confirm-paid",
    is_flag=True,
    default=False,
    help="Required to run the anthropic (paid) suggester.",
)
@_edit_books_dir_option
def lexicon_suggest_ai(
    book_id: str,
    terms: tuple[str, ...],
    provider: str | None,
    confirm_paid: bool,
    books_dir: Path | None,
) -> None:
    """ADVISORY LLM respellings for hard terms (opt-in enrichment of `suggest`).

    Prints proposals only — accept one with `seiyuu lexicon set --term ... --respelling ...`.
    The deterministic `suggest` stays the free default; this adds an LLM layer on top."""
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.normalize.lexicon import load_lexicon, suggest_terms
    from seiyuu.services.llm_advisory import resolve_advisory, run_respell_suggestions
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _lexicon_book_dir(book_id, books_dir)
    requested = [t.strip() for t in terms if t.strip()]
    if not requested:
        book = NormalizedBook.model_validate_json(
            (book_dir / "normalized.json").read_text(encoding="utf-8")
        )
        lex = load_lexicon(book_dir, book_id=book_id)
        texts = [b.text for c in book.chapters for b in c.blocks if b.is_speakable]
        requested = [
            s.term for s in suggest_terms(texts, existing_terms=[e.term for e in lex.entries])
        ]
    if not requested:
        click.echo("(no candidate terms found)")
        return

    resolved = resolve_advisory(cfg, cfg.respell_provider, cfg.respell_model, provider)
    if resolved.is_paid:
        if not confirm_paid:
            raise click.ClickException(
                f"provider {resolved.provider_id!r} is a PAID Anthropic call; "
                "re-run with --confirm-paid to approve the spend."
            )
        if not cfg.anthropic_api_key:
            raise click.ClickException(
                "ANTHROPIC_API_KEY not set; required for the anthropic suggester"
            )

    click.echo(f"suggester: {resolved.provider_id}/{resolved.model}  ({len(requested)} term(s))")
    try:
        suggestions = run_respell_suggestions(cfg, resolved, requested)
    except Exception as exc:
        raise click.ClickException(f"LLM respell suggester failed: {exc}") from exc
    if not suggestions:
        click.echo("(no suggestions returned)")
        return
    for s in suggestions:
        note = f"  # {s.note}" if s.note else ""
        click.echo(f"{s.term!r} -> {s.respelling!r}{note}")


@main.group()
def voice() -> None:
    """Manage the voice library (voices/{voice_id}/)."""


@voice.command("list")
@_voices_dir_option
def voice_list(voices_dir: Path | None) -> None:
    """List voices in the library."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceLibrary

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    metas = lib.list_voices()
    if not metas:
        click.echo("no voices yet — try `seiyuu voice add-preset` or `seiyuu voice blend`")
        return
    for m in metas:
        if m.blend:
            detail = " + ".join(f"{c.preset_id}:{c.weight:g}" for c in m.blend)
        else:
            detail = m.preset_id or m.reference_audio or "—"
        click.echo(f"  {m.voice_id}  [{m.kind}/{m.engine}]  {m.name}  ({detail})  seed={m.seed}")


@voice.command("add-preset")
@click.argument("name")
@click.argument("preset_id")
@click.option("--engine", default="kokoro", show_default=True, help="TTS engine.")
@click.option("--seed", default=41172, show_default=True, help="Pinned synthesis seed.")
@click.option("--voice-id", default=None, help="Explicit voice_id (default: slug + random).")
@_voices_dir_option
def voice_add_preset(
    name: str, preset_id: str, engine: str, seed: int, voice_id: str | None, voices_dir: Path | None
) -> None:
    """Create a single-preset voice, e.g. `voice add-preset Narrator af_heart`."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceLibraryError, VoiceMeta

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    vid = voice_id or lib.new_voice_id(name)
    try:
        lib.save(
            VoiceMeta(
                voice_id=vid,
                name=name,
                kind=VoiceKind.PRESET,
                engine=engine,
                preset_id=preset_id,
                seed=seed,
                source="preset",
            )
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"added preset voice {vid} -> {preset_id}")


@voice.command("blend")
@click.argument("name")
@click.option(
    "--component",
    "components",
    multiple=True,
    help="preset_id:weight (repeatable; >=2 components). Omit to auto-draft from --gender.",
)
@click.option(
    "--gender", default=None, help="Auto-draft: same-family 2-preset blend for this gender."
)
@click.option(
    "--accent",
    default="a",
    show_default=True,
    help="Auto-draft accent: a (American) / b (British).",
)
@click.option("--seed", default=41172, show_default=True, help="Pinned synthesis seed.")
@click.option("--voice-id", default=None, help="Explicit voice_id (default: slug + random).")
@_voices_dir_option
def voice_blend(
    name: str,
    components: tuple[str, ...],
    gender: str | None,
    accent: str,
    seed: int,
    voice_id: str | None,
    voices_dir: Path | None,
) -> None:
    """Create a Kokoro blend voice from explicit components or an auto-draft."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import (
        BlendComponent,
        VoiceKind,
        VoiceLibrary,
        VoiceLibraryError,
        VoiceMeta,
        auto_blend_recipe,
        canonical_recipe,
    )

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    if components:
        parsed = []
        for spec in components:
            preset, _, weight = spec.partition(":")
            try:
                parsed.append((preset, float(weight)))
            except ValueError as exc:
                raise click.ClickException(
                    f"bad --component {spec!r}; expected preset_id:weight"
                ) from exc
        recipe = canonical_recipe(parsed)
        source = "manual_blend"
    else:
        recipe = auto_blend_recipe(name, gender, accent=accent)
        source = "auto_blend"

    blend = [BlendComponent(preset_id=p, weight=w) for p, w in recipe]
    vid = voice_id or lib.new_voice_id(name)
    try:
        lib.save(
            VoiceMeta(
                voice_id=vid,
                name=name,
                kind=VoiceKind.BLEND,
                engine="kokoro",
                blend=blend,
                seed=seed,
                source=source,
            )
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"added blend voice {vid}: {', '.join(f'{p}:{w:g}' for p, w in recipe)}")


@voice.command("audition")
@click.argument("voice_id")
@click.option(
    "--text",
    default='The quick brown fox jumps over the lazy dog. "Well," she said, "how about that?"',
    help="Sample line to synthesize.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output WAV (default: voices/{voice_id}/audition.wav).",
)
@_voices_dir_option
def voice_audition(voice_id: str, text: str, out: Path | None, voices_dir: Path | None) -> None:
    """Synthesize a sample line with a voice so you can hear it (loads a TTS engine)."""
    from contextlib import nullcontext

    from seiyuu.engines import get_engine, voices_dir_kwargs
    from seiyuu.engines.base import SynthesisError
    from seiyuu.gpu import get_gpu_manager
    from seiyuu.normalize import normalize_text, profile_for
    from seiyuu.settings import get_settings
    from seiyuu.voices import (
        VoiceKind,
        VoiceLibrary,
        VoiceLibraryError,
        ensure_cloud_voice,
        render_voice_args,
    )

    cfg = get_settings()
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    try:
        meta = lib.load(voice_id)
    except VoiceLibraryError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        lib.verify_consent(meta)  # cloned: attested + reference hash-matches the record
    except VoiceLibraryError as exc:
        raise click.ClickException(str(exc)) from exc

    engine = get_engine(meta.engine, **voices_dir_kwargs(meta.engine, lib.voices_dir))
    engine_voice, settings = render_voice_args(meta)
    norm = normalize_text(text, profile=profile_for(meta.engine))

    # cloud voices are paid: confirm the (small) cost and resolve the cloud handle first
    cost = engine.cost_estimate(norm)
    if cost > 0:
        click.confirm(
            f"Auditioning {voice_id} is a paid call (~${cost:.4f}). Continue?", abort=True
        )
        if meta.engine == "elevenlabs" and meta.kind is VoiceKind.CLONED:
            from seiyuu.repository import RepositoryError
            from seiyuu.voices import CloudVoiceError

            try:
                engine_voice = ensure_cloud_voice(
                    meta, engine.client, lib, max_slots=cfg.elevenlabs_max_voice_slots
                )
            except (SynthesisError, CloudVoiceError, RepositoryError) as exc:
                raise click.ClickException(str(exc)) from exc

    gpu = get_gpu_manager()
    ctx = gpu.acquire(engine, engine.engine_id) if engine.uses_gpu else nullcontext()
    try:
        with ctx:
            audio = engine.synthesize(norm, engine_voice, {**settings, "seed": meta.seed})
    except SynthesisError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        if engine.uses_gpu:
            gpu.free_all()
    out_path = Path(out) if out else lib.dir_for(voice_id) / "audition.wav"
    audio.save(out_path)
    click.echo(f"wrote {out_path} ({audio.duration_seconds:.1f}s)")


@voice.command("add-cloud")
@click.argument("name")
@click.argument("remote_voice_id")
@click.option("--engine", default="elevenlabs", show_default=True, help="Cloud engine.")
@click.option("--seed", default=41172, show_default=True, help="Pinned synthesis seed.")
@click.option("--voice-id", default=None, help="Explicit voice_id (default: slug + random).")
@_voices_dir_option
def voice_add_cloud(
    name: str, remote_voice_id: str, engine: str, seed: int, voice_id: str | None,
    voices_dir: Path | None,
) -> None:  # fmt: skip
    """Add a stock cloud voice, e.g. `voice add-cloud Rachel EXAVITQu` (a preset on the cloud)."""
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceLibraryError, VoiceMeta

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    vid = voice_id or lib.new_voice_id(name)
    try:
        lib.save(
            VoiceMeta(
                voice_id=vid,
                name=name,
                kind=VoiceKind.PRESET,
                engine=engine,
                preset_id=remote_voice_id,
                seed=seed,
                source="preset",
            )  # fmt: skip
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"added cloud voice {vid} -> {engine}:{remote_voice_id}")


@voice.command("delete")
@click.argument("voice_id")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Render output root scanned for assignments (default: settings.output_dir). "
    "Only THIS root is scanned — assignments under other --output-dir roots are not seen.",
)
@_voices_dir_option
def voice_delete(
    voice_id: str, yes: bool, output_dir: Path | None, voices_dir: Path | None
) -> None:
    """Delete a voice (refused while any book's assignment still references it)."""
    from seiyuu.services import ServiceError, delete_voice
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceLibrary

    cfg = get_settings()
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    if not yes:
        click.confirm(
            f"Delete voice {voice_id!r} and its directory (including any reference.wav — "
            f"the consent-attested source of a clone)?",
            abort=True,
        )
    try:
        gone = delete_voice(voice_id, library=lib, output_dir=output_dir or cfg.output_dir)
    except ServiceError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"deleted {gone}")


@voice.command("clone")
@click.argument("name")
@click.argument("reference", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--engine",
    type=click.Choice(["chatterbox", "indextts2", "elevenlabs"]),
    default="chatterbox",
    show_default=True,
    help="Cloning engine: chatterbox/indextts2 (local) or elevenlabs (cloud IVC).",
)
@click.option(
    "--consent", is_flag=True, help="Attest you have the rights/consent to clone this voice."
)
@click.option(
    "--consent-by",
    default=None,
    help="Who is attesting (recorded in the attestation; default: the OS user).",
)
@click.option("--seed", default=41172, show_default=True, help="Pinned synthesis seed.")
@click.option("--voice-id", default=None, help="Explicit voice_id (default: slug + random).")
@_voices_dir_option
def voice_clone(
    name: str, reference: Path, engine: str, consent: bool, consent_by: str | None,
    seed: int, voice_id: str | None, voices_dir: Path | None,
) -> None:  # fmt: skip
    """Create a cloned voice from a reference clip (chatterbox/indextts2 local, elevenlabs IVC)."""
    import getpass
    import shutil

    from seiyuu.settings import get_settings
    from seiyuu.voices import (
        ConsentAttestation,
        VoiceKind,
        VoiceLibrary,
        VoiceLibraryError,
        VoiceMeta,
    )
    from seiyuu.voices.library import sha256_file

    if not consent:
        raise click.ClickException(
            "cloning requires --consent; you must have the rights/permission to clone this voice"
        )
    import os

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    vid = voice_id or lib.new_voice_id(name)
    # Order matters: copy first, hash the COPY, publish it atomically, THEN save the meta.
    # The attestation provably describes the exact bytes the library will serve — a failed
    # copy can't leave an attested meta pointing at missing/partial audio, and the source
    # clip changing mid-command can't be attested for.
    ref_path = lib.reference_path(vid)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ref_path.with_name("reference.wav.part")
    shutil.copyfile(reference, tmp)
    attestation = ConsentAttestation(
        attested_by=consent_by or getpass.getuser(),
        reference_sha256=sha256_file(tmp),
    )
    os.replace(tmp, ref_path)
    # a re-clone under an existing voice_id replaces the audio: purge conds derived from
    # the OLD reference so nothing can speak the previously-attested speaker
    for stale in ref_path.parent.glob("conds_*.pt"):
        stale.unlink(missing_ok=True)
    try:
        lib.save(
            VoiceMeta(
                voice_id=vid,
                name=name,
                kind=VoiceKind.CLONED,
                engine=engine,
                reference_audio="reference.wav",
                consent_attested=True,
                consent=attestation,
                seed=seed,
                source="user_upload",
            )  # fmt: skip
        )
    except (VoiceLibraryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"cloned voice {vid} ({engine}) from {reference.name}")
    click.echo(
        f"consent recorded: {attestation.attested_by}, "
        f"sha256 {attestation.reference_sha256[:12]}…, {attestation.attested_at[:19]}"
    )


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
    except (ServiceError, ValueError) as exc:
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


def _pause_options(fn):
    """Pause-tuning flags shared by `assemble` and `convert` (seconds)."""
    for name, help_text in reversed(
        [
            ("--pause-paragraph", "Silence between paragraphs."),
            ("--pause-after-heading", "Silence after a chapter heading."),
            ("--pause-scene-break", "Silence at a scene break (replaces the paragraph gap)."),
            ("--pause-dialogue", "Silence between consecutive dialogue turns (multi-voice)."),
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
        dialogue=overrides.get("pause_dialogue") or defaults.dialogue,
        chapter_lead_in=overrides.get("pause_lead_in") or defaults.chapter_lead_in,
        chapter_lead_out=overrides.get("pause_lead_out") or defaults.chapter_lead_out,
    )


def _loudness_options(fn):
    """Loudness-normalization flags shared by `assemble` and `convert`."""
    fn = click.option(
        "--target-lufs", type=float, default=None, help="Integrated loudness target (LUFS)."
    )(fn)
    fn = click.option(
        "--loudness/--no-loudness",
        "loudness",
        default=None,
        help="Loudness-normalize chapters to the target LUFS (default: on).",
    )(fn)
    return fn


def _build_loudness(cfg, **overrides):
    from seiyuu.assemble import LoudnessTarget

    enabled = cfg.loudness_enabled if overrides.get("loudness") is None else overrides["loudness"]
    if not enabled:
        return None
    target = overrides.get("target_lufs")
    return LoudnessTarget(
        i=cfg.loudness_target_lufs if target is None else target,
        tp=cfg.loudness_true_peak,
        lra=cfg.loudness_range,
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


if __name__ == "__main__":
    main()
