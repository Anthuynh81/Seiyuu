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
        _render_multivoice_cli(cfg, book, book_dir, output_dir, voices_dir, chapter_indices)
        return

    from seiyuu.engines import get_engine
    from seiyuu.render import RenderError, render_book

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
            validator=_build_validator(cfg),
            validation_max_retries=cfg.validation_max_retries,
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


def _render_multivoice_cli(cfg, book, book_dir, output_dir, voices_dir, chapter_indices):
    """Shared multi-voice render: load attribution + assignment + library, render, echo summary."""
    from seiyuu.attribute import ATTRIBUTION_NAME, AttributionReport
    from seiyuu.render import RenderError, render_book_multivoice
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
):
    """convert --multivoice: attribute → auto-assign draft voices → multi-voice render."""
    from seiyuu.attribute import AttributionError
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
    out_book_dir.mkdir(parents=True, exist_ok=True)
    (out_book_dir / ASSIGNMENT_NAME).write_text(
        assignment.model_dump_json(indent=2), encoding="utf-8"
    )
    click.echo(
        f"narrator {assignment.narrator_voice_id}, {len(assignment.assignments)} character voices"
    )

    click.echo("== render (multi-voice) ==")
    _render_multivoice_cli(cfg, book, attr_book_dir, output_dir, voices_dir, chapter_indices)


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
    from seiyuu.engines import get_engine
    from seiyuu.gpu import get_gpu_manager
    from seiyuu.normalize import normalize_text, profile_for
    from seiyuu.settings import get_settings
    from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceLibraryError, render_voice_args

    lib = VoiceLibrary(voices_dir or get_settings().voices_dir)
    try:
        meta = lib.load(voice_id)
    except VoiceLibraryError as exc:
        raise click.ClickException(str(exc)) from exc
    if meta.kind is VoiceKind.CLONED and not meta.consent_attested:
        raise click.ClickException(f"voice {voice_id} (cloned) has no consent attestation")

    extra = {"voices_dir": lib.voices_dir} if meta.engine == "chatterbox" else {}
    engine = get_engine(meta.engine, **extra)
    engine_voice, settings = render_voice_args(meta)
    norm = normalize_text(text, profile=profile_for(meta.engine))

    gpu = get_gpu_manager()
    try:
        with gpu.acquire(engine, engine.engine_id):
            audio = engine.synthesize(norm, engine_voice, {**settings, "seed": meta.seed})
    finally:
        gpu.free_all()
    out_path = Path(out) if out else lib.dir_for(voice_id) / "audition.wav"
    audio.save(out_path)
    click.echo(f"wrote {out_path} ({audio.duration_seconds:.1f}s)")


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
    books_dir: Path | None,
    output_dir: Path | None,
    voices_dir: Path | None,
) -> None:
    """Build a draft character→voice assignment (auto-creating draft voices)."""
    from seiyuu.attribute import ATTRIBUTION_NAME, AttributionReport
    from seiyuu.settings import get_settings
    from seiyuu.voices import ASSIGNMENT_NAME, VoiceLibrary

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
    out_dir = (output_dir or cfg.output_dir) / report.book_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ASSIGNMENT_NAME
    path.write_text(assignment.model_dump_json(indent=2), encoding="utf-8")

    click.echo(f"narrator: {assignment.narrator_voice_id}")
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
        )
    else:
        click.echo("== render ==")
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
                validator=_build_validator(cfg),
                validation_max_retries=cfg.validation_max_retries,
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
