"""Shared CLI helpers: book-dir resolution, reusable option decorators, the cost gate,
and the multi-voice render glue used by both `render` and `convert`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class _PendingCost:
    """A paid render that passed every NON-consuming money check — nothing approved yet.

    Produced by ``_check_cost_gate``; only ``_commit_cost_gate`` turns it into a spend
    budget (consuming the token / asking the user), so the render's cheap pre-flight can
    run in between with the single-use token still unburned."""

    est: Any  # seiyuu.render.CostEstimate (lazy import keeps CLI startup light)
    book_id: str
    chapters: tuple[int, ...]
    assignment_hash: str | None
    quote: Any | None  # the decoded CostQuote, or None for --confirm-cost / prompt


def _check_cost_gate(
    cfg, est, *, book_id, chapters, assignment_hash, cost_token
) -> _PendingCost | None:
    """Money checks that never spend the approval (the API's enqueue dry-run, CLI-side).

    None for a free render (the gate never engages; the pipeline's per-segment refusal
    stays the safety net). With a token: decode + full ``verify_quote(consume=False)``
    against the FRESH estimate + a single-use ledger probe, so a tampered / expired /
    drifted / already-spent token refuses now. Without one: the ``render_max_usd``
    ceiling. Raises ClickException with the gate's reason."""
    if est.total_usd <= 0:
        return None
    from seiyuu.render import CostGateError, CostQuote, check_ceiling, verify_quote
    from seiyuu.render.gate import quote_consumed

    click.echo(
        f"cost estimate: ${est.total_usd:.2f} over {est.paid_segments} paid segment(s) "
        f"({est.cached_segments} cached)"
    )
    chapters = tuple(chapters)
    quote = None
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
                consume=False,  # dry-run: the commit step re-verifies and consumes
            )
            # verify(consume=False) never reads the ledger; best-effort probe so a spent
            # token refuses here — the commit's atomic consume remains the enforcement
            if quote_consumed(cfg.data_dir, quote.sig):
                raise CostGateError(
                    "cost token already used (tokens are single-use); re-run estimate-cost "
                    "for a new approval"
                )
        else:
            check_ceiling(est.total_usd, cfg.render_max_usd)
    except CostGateError as exc:
        raise click.ClickException(str(exc)) from exc
    return _PendingCost(
        est=est, book_id=book_id, chapters=chapters, assignment_hash=assignment_hash, quote=quote
    )


def _commit_cost_gate(cfg, pending: _PendingCost | None, *, confirm_cost: bool) -> float | None:
    """The ONLY step that approves spend: call it after every cheap pre-flight, right
    before synthesis. Returns the approved paid budget in USD (the render loop's hard
    spend cap), or None when nothing paid may run.

    A token is re-verified in full against the same estimate and atomically CONSUMED
    (``verify_quote(consume=True)`` — consumption is its last step, so a refusal here
    still never burns it). Without a token, approval is ``--confirm-cost`` or the
    interactive prompt (asked only once the render is known to be startable)."""
    if pending is None:
        return None
    if pending.quote is not None:
        from seiyuu.render import CostGateError, verify_quote

        try:
            verify_quote(
                pending.quote,
                book_id=pending.book_id,
                chapters=pending.chapters,
                fingerprint=pending.est.fingerprint,
                assignment_hash=pending.assignment_hash,
                recomputed_total_usd=pending.est.total_usd,
                max_usd=cfg.render_max_usd,
                data_dir=cfg.data_dir,
                consume=True,
            )
        except CostGateError as exc:
            raise click.ClickException(str(exc)) from exc
        return pending.quote.total_usd
    if not confirm_cost:
        click.confirm("This makes paid cloud API calls. Continue?", abort=True)
    return pending.est.total_usd


def _pass_cost_gate(cfg, est, *, book_id, chapters, assignment_hash, confirm_cost, cost_token):
    """The M6a money gate in one step: no paid synthesis without the ceiling AND explicit
    approval. ``_check_cost_gate`` then ``_commit_cost_gate`` with nothing in between —
    for callers with no pre-flight to run. Paths that DO have cheap pre-flight checks
    (the render commands) call the two halves separately so a pre-flight failure never
    consumes a single-use token."""
    pending = _check_cost_gate(
        cfg, est, book_id=book_id, chapters=chapters, assignment_hash=assignment_hash,
        cost_token=cost_token,
    )  # fmt: skip
    return _commit_cost_gate(cfg, pending, confirm_cost=confirm_cost)


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

    from seiyuu.engines import SynthesisError
    from seiyuu.render.preflight import preflight_multivoice
    from seiyuu.voices import VoiceLibraryError

    # Verify-then-consume, the API's order: estimate -> non-consuming money check ->
    # cheap pre-flight -> consume -> render. Nothing between the commit and synthesis can
    # fail for a reason the pre-flight could have caught, so a dead-end never burns the
    # user's single-use token (or asks them to approve spend for a doomed render).
    try:
        est = estimate_render_cost(
            report, book, lib, assignment, out_book_dir, chapters=chapter_indices,
            lexicon=lexicon, apply_emotion=apply_emotion, force=force,
        )  # fmt: skip
        pending = _check_cost_gate(
            cfg, est,
            book_id=book_id,
            chapters=chapter_indices,
            assignment_hash=hash_assignment(assignment),
            cost_token=cost_token,
        )  # fmt: skip
        preflight_multivoice(
            report, book, lib, assignment, out_book_dir, est, chapters=chapter_indices
        )
        validator = _build_validator(cfg)
    # SynthesisError: an indextts2 voice without checkpoints fails in model_version
    except (RenderError, SynthesisError, ValueError, VoiceLibraryError) as exc:
        raise click.ClickException(str(exc)) from exc
    approved_usd = _commit_cost_gate(cfg, pending, confirm_cost=confirm_cost)

    try:
        result = render_book_multivoice(
            report,
            book,
            lib,
            assignment,
            out_book_dir,
            chapters=chapter_indices,
            progress=click.echo,
            validator=validator,
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
    """Loudness flags shared by `assemble`, `master` and `convert`."""
    fn = click.option(
        "--match-voices/--no-match-voices",
        "match_voices",
        default=None,
        help="Match each voice's loudness to the narrator before normalizing (default: on).",
    )(fn)
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


def _build_voice_match(cfg, **overrides):
    from seiyuu.assemble import VoiceMatch

    flag = overrides.get("match_voices")
    enabled = cfg.voice_loudness_match if flag is None else flag
    return VoiceMatch(max_gain_db=cfg.voice_loudness_max_gain_db) if enabled else None
