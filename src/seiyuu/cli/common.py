"""Shared CLI helpers: book-dir resolution, reusable option decorators, the cost gate,
and the multi-voice render glue used by both `render` and `convert`."""

from pathlib import Path

import click


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
    confirm_cost=False, cost_token=None, apply_emotion=None, force=False,
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
        apply_emotion=apply_emotion, force=force,
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
            force=force,
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


def _edit_books_dir_option(fn):
    return click.option(
        "--books-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Where attributed books live (default: settings.books_dir).",
    )(fn)


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
