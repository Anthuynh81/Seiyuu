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
            confirm_cost=confirm_cost, cost_token=cost_token,
        )  # fmt: skip
        return

    from seiyuu.engines import get_engine
    from seiyuu.render import RenderError, estimate_render_cost_single, render_book
    from seiyuu.voices import VoiceLibrary

    engine_id = engine_id or cfg.tts_engine
    voice = voice or cfg.kokoro_default_voice
    out_book_dir = (output_dir or cfg.output_dir) / book.book_meta.book_id
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    try:
        # chatterbox resolves clones from the library dir; the consent gate in render_book
        # needs the same library, or --voices-dir clones would render ungated
        extra = {"voices_dir": lib.voices_dir} if engine_id == "chatterbox" else {}
        engine = get_engine(engine_id, **extra)
        # pre-flight: single-voice paid renders get the same estimate + ceiling + approval
        # gate as multivoice (the M5 gap: --confirm-cost used to authorize blind)
        est = estimate_render_cost_single(
            book, engine, voice, out_book_dir,
            settings={"speed": speed}, seed=seed, chapters=chapter_indices,
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
        )
    except (RenderError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    minutes = result.total_audio_seconds / 60
    click.echo(
        f"done: {result.synthesized} segments synthesized, "
        f"{result.cache_hits} from cache, {minutes:.1f} min of audio"
    )
    if result.validation_failures:
        click.echo(f"  {result.validation_failures} segment(s) failed whisper validation")
    click.echo(f"manifest: {result.manifest_path}")


def _build_provider(cfg, provider_id: str, model: str, prompt_version: str):
    """Construct an attribution provider, passing only the kwargs each backend needs."""
    from seiyuu.attribute.providers import get_provider

    kwargs = {"prompt_version": prompt_version}
    if provider_id == "local":
        kwargs["base_url"] = cfg.ollama_base_url
        kwargs["transport"] = cfg.ollama_transport
        kwargs["num_ctx"] = cfg.ollama_num_ctx
        kwargs["keep_alive"] = cfg.ollama_keep_alive
        kwargs["unload_poll_timeout"] = cfg.gpu_unload_poll_timeout
    elif provider_id == "anthropic":
        kwargs["api_key"] = cfg.anthropic_api_key
    return get_provider(provider_id, model=model, prompts_dir=cfg.prompts_dir, **kwargs)


def _build_validator(cfg):
    """A whisper validator from settings; loads lazily, so Kokoro-only renders never touch it."""
    from seiyuu.validate import Validator

    return Validator(
        model_size=cfg.validation_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.validation_compute_type,
        min_ratio=cfg.validation_min_ratio,
    )


def _run_attribution(
    cfg,
    book,
    book_dir: Path,
    *,
    provider_id: str,
    model: str,
    prompt_version: str,
    use_hybrid: bool,
    chapter_indices: tuple[int, ...],
    progress,
):
    """Attribute `book`, write attribution.json + the cache DB, return (report, provider).

    Shared by `seiyuu attribute` and `seiyuu convert --multivoice`. Does NOT unload the
    provider — the caller frees Ollama VRAM before any TTS engine loads (GPU discipline).
    """
    from seiyuu.attribute import AttributionCache, attribute_book, write_attribution

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
            progress=progress,
        )
    write_attribution(report, book_dir)
    return report, provider


def _auto_assign(report, lib, *, narrator_voice_id, thought_voice_id, accent, default_preset):
    """Build a draft VoiceAssignment, creating any missing draft voices in `lib`.

    Narrator is an explicit existing voice or an auto preset for `default_preset`. Each
    character gets a deterministic auto-blend voice keyed by its character id, so re-running
    `assign` reproduces the same draft voices (and therefore the same segment cache entries).
    """
    from seiyuu.voices import (
        BlendComponent,
        VoiceAssignment,
        VoiceKind,
        VoiceMeta,
        auto_blend_recipe,
        slugify,
    )

    if narrator_voice_id is None:
        narrator_voice_id = f"narrator_{slugify(default_preset)}"
        if not lib.meta_path(narrator_voice_id).is_file():
            lib.save(
                VoiceMeta(
                    voice_id=narrator_voice_id,
                    name="Narrator",
                    kind=VoiceKind.PRESET,
                    engine="kokoro",
                    preset_id=default_preset,
                    source="preset",
                )
            )
    elif not lib.meta_path(narrator_voice_id).is_file():
        raise click.ClickException(f"narrator voice {narrator_voice_id!r} not in the library")
    if thought_voice_id is not None and not lib.meta_path(thought_voice_id).is_file():
        raise click.ClickException(f"thought voice {thought_voice_id!r} not in the library")

    assignments: dict[str, str] = {}
    for char in report.registry.characters:
        voice_id = f"{char.id}_auto"
        if not lib.meta_path(voice_id).is_file():
            recipe = auto_blend_recipe(char.canonical_name, char.gender, accent=accent)
            blend = [BlendComponent(preset_id=p, weight=w) for p, w in recipe]
            lib.save(
                VoiceMeta(
                    voice_id=voice_id,
                    name=char.canonical_name,
                    kind=VoiceKind.BLEND,
                    engine="kokoro",
                    blend=blend,
                    source="auto_blend",
                )
            )
        assignments[char.id] = voice_id

    return VoiceAssignment(
        book_id=report.book_id,
        narrator_voice_id=narrator_voice_id,
        assignments=assignments,
        thought_voice_id=thought_voice_id,
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
    confirm_cost=False, cost_token=None,
):  # fmt: skip
    """Shared multi-voice render: load inputs, cost-gate paid engines, render, echo summary."""
    from seiyuu.attribute import ATTRIBUTION_NAME, AttributionReport
    from seiyuu.render import (
        RenderError,
        estimate_render_cost,
        hash_assignment,
        render_book_multivoice,
    )
    from seiyuu.settings import get_settings
    from seiyuu.voices import ASSIGNMENT_NAME, VoiceAssignment, VoiceLibrary

    book_id = book.book_meta.book_id
    attr_path = book_dir / ATTRIBUTION_NAME
    if not attr_path.is_file():
        raise click.ClickException(
            f"no attribution at {attr_path}; run `seiyuu attribute {book_id}` first"
        )
    report = AttributionReport.model_validate_json(attr_path.read_text(encoding="utf-8"))

    out_book_dir = (output_dir or get_settings().output_dir) / book_id
    assign_path = out_book_dir / ASSIGNMENT_NAME
    if not assign_path.is_file():
        raise click.ClickException(
            f"no assignment at {assign_path}; run `seiyuu assign {book_id}` first"
        )
    assignment = VoiceAssignment.model_validate_json(assign_path.read_text(encoding="utf-8"))
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)

    # cost gate: estimate first; paid segments require the ceiling + explicit approval
    est = estimate_render_cost(
        report, book, lib, assignment, out_book_dir, chapters=chapter_indices
    )
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
    from seiyuu.repository import atomic_write_text
    from seiyuu.voices import ASSIGNMENT_NAME, VoiceLibrary

    click.echo("== attribute ==")
    use_hybrid = cfg.attribution_hybrid if hybrid is None else hybrid
    try:
        report, provider = _run_attribution(
            cfg,
            book,
            attr_book_dir,
            provider_id=cfg.attribution_provider,
            model=cfg.attribution_model,
            prompt_version=cfg.attribution_prompt_version,
            use_hybrid=use_hybrid,
            chapter_indices=chapter_indices,
            progress=click.echo,
        )
    except (AttributionError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    provider.unload()  # free Ollama VRAM before the TTS engine loads (GPU discipline)
    flagged = f", {len(report.flagged)} flagged" if report.flagged else ""
    click.echo(f"{len(report.registry.characters)} characters{flagged}")

    click.echo("== assign ==")
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    assignment = _auto_assign(
        report,
        lib,
        narrator_voice_id=narrator_voice_id,
        thought_voice_id=thought_voice_id,
        accent=accent,
        default_preset=cfg.kokoro_default_voice,
    )
    out_book_dir = (output_dir or cfg.output_dir) / book.book_meta.book_id
    atomic_write_text(out_book_dir / ASSIGNMENT_NAME, assignment.model_dump_json(indent=2))
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

    provider_id = provider_id or cfg.attribution_provider
    model = model or cfg.attribution_model
    prompt_version = prompt_version or cfg.attribution_prompt_version
    use_hybrid = cfg.attribution_hybrid if hybrid is None else hybrid

    try:
        report, provider = _run_attribution(
            cfg,
            book,
            book_dir,
            provider_id=provider_id,
            model=model,
            prompt_version=prompt_version,
            use_hybrid=use_hybrid,
            chapter_indices=chapter_indices,
            progress=click.echo,
        )
    except (AttributionError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

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

    from seiyuu.engines import get_engine
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

    extra = {"voices_dir": lib.voices_dir} if meta.engine == "chatterbox" else {}
    engine = get_engine(meta.engine, **extra)
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


@voice.command("clone")
@click.argument("name")
@click.argument("reference", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--engine", default="chatterbox", show_default=True, help="Cloning engine.")
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
    """Create a cloned voice from a reference clip (chatterbox local or elevenlabs IVC)."""
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
    books_dir: Path | None,
    output_dir: Path | None,
    voices_dir: Path | None,
) -> None:
    """Build a character→voice assignment (auto-drafts locals; --map overrides, e.g. to cloud)."""
    from seiyuu.attribute import ATTRIBUTION_NAME, AttributionReport
    from seiyuu.repository import atomic_write_text
    from seiyuu.settings import get_settings
    from seiyuu.voices import ASSIGNMENT_NAME, AssignmentStage, VoiceLibrary

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    report = AttributionReport.model_validate_json(
        (book_dir / ATTRIBUTION_NAME).read_text(encoding="utf-8")
    )
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    assignment = _auto_assign(
        report,
        lib,
        narrator_voice_id=narrator_voice_id,
        thought_voice_id=thought_voice_id,
        accent=accent,
        default_preset=cfg.kokoro_default_voice,
    )
    known_ids = {c.id for c in report.registry.characters}
    for entry in maps:
        char_id, _, vid = entry.partition("=")
        if not vid:
            raise click.ClickException(f"bad --map {entry!r}; expected CHARACTER_ID=VOICE_ID")
        if char_id not in known_ids:
            raise click.ClickException(f"--map: unknown character {char_id!r}")
        if not lib.meta_path(vid).is_file():
            raise click.ClickException(f"--map: voice {vid!r} not in the library")
        assignment.assignments[char_id] = vid
    assignment.stage = AssignmentStage(stage)

    path = (output_dir or cfg.output_dir) / report.book_id / ASSIGNMENT_NAME
    atomic_write_text(path, assignment.model_dump_json(indent=2))

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
    """Shared loader for the multi-voice cost/render path: (report, book, lib, assignment, dir)."""
    from seiyuu.attribute import ATTRIBUTION_NAME, AttributionReport
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.voices import ASSIGNMENT_NAME, VoiceAssignment, VoiceLibrary

    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, ATTRIBUTION_NAME, "Run `seiyuu attribute` first."
    )
    report = AttributionReport.model_validate_json(
        (book_dir / ATTRIBUTION_NAME).read_text(encoding="utf-8")
    )
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )
    out_book_dir = (output_dir or cfg.output_dir) / report.book_id
    assign_path = out_book_dir / ASSIGNMENT_NAME
    if not assign_path.is_file():
        raise click.ClickException(f"no assignment at {assign_path}; run `seiyuu assign {book_id}`")
    assignment = VoiceAssignment.model_validate_json(assign_path.read_text(encoding="utf-8"))
    lib = VoiceLibrary(voices_dir or cfg.voices_dir)
    return report, book, lib, assignment, out_book_dir


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
    from seiyuu.settings import get_settings

    cfg = get_settings()
    if voice or engine_id:
        # single-voice mode: mirrors the render command's defaults exactly
        from seiyuu.engines import get_engine
        from seiyuu.ingest.models import NormalizedBook
        from seiyuu.render import estimate_render_cost_single

        book_dir = _resolve_book_dir(
            books_dir or cfg.books_dir, book_id, "normalized.json", "Run `seiyuu ingest` first."
        )
        book = NormalizedBook.model_validate_json(
            (book_dir / "normalized.json").read_text(encoding="utf-8")
        )
        est = estimate_render_cost_single(
            book,
            get_engine(engine_id or cfg.tts_engine),
            voice or cfg.kokoro_default_voice,
            (output_dir or cfg.output_dir) / book.book_meta.book_id,
            settings={"speed": speed},
            seed=seed,
            chapters=chapter_indices,
        )
        assignment_hash = None
    else:
        from seiyuu.render import estimate_render_cost, hash_assignment

        report, book, lib, assignment, out_book_dir = _load_multivoice_inputs(
            cfg, book_id, books_dir, output_dir, voices_dir, chapter_indices
        )
        est = estimate_render_cost(
            report, book, lib, assignment, out_book_dir, chapters=chapter_indices
        )
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
    epub_path: Path,
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
    """Full pipeline: EPUB -> normalized JSON -> render (single- or multi-voice) -> chapter MP3s."""
    from seiyuu.assemble import AssembleError, assemble_book, master_book
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
        from seiyuu.render import estimate_render_cost_single
        from seiyuu.voices import VoiceLibrary

        lib = VoiceLibrary(voices_dir or cfg.voices_dir)
        try:
            single_engine_id = engine_id or cfg.tts_engine
            extra = {"voices_dir": lib.voices_dir} if single_engine_id == "chatterbox" else {}
            engine = get_engine(single_engine_id, **extra)
            single_voice = voice or cfg.kokoro_default_voice
            est = estimate_render_cost_single(
                book, engine, single_voice, book_dir,
                settings={"speed": speed}, seed=seed, chapters=chapter_indices,
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
            )
        except (RenderError, ValueError) as exc:
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


if __name__ == "__main__":
    main()
