"""`seiyuu eval`: the attribution accuracy harness (gold export + multi-model comparison)."""

from pathlib import Path

import click

from seiyuu.cli import main
from seiyuu.cli.common import _resolve_book_dir


def parse_chapters(spec: str | None) -> tuple[int, ...]:
    """``"1-5,8"`` -> (1, 2, 3, 4, 5, 8); empty/None -> () (all / gold's chapters)."""
    if not spec:
        return ()
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if lo > hi:
                    raise ValueError
                out.update(range(lo, hi + 1))
            else:
                out.add(int(part))
        except ValueError:
            raise click.BadParameter(f"bad chapter range {part!r}; use e.g. 1-5,8") from None
    if any(c < 1 for c in out):
        raise click.BadParameter("chapters are 1-based")
    return tuple(sorted(out))


_books_dir_option = click.option(
    "--books-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where normalized books live (default: settings.books_dir).",
)


@main.group("eval")
def eval_group() -> None:
    """Attribution accuracy harness: export gold labels, compare models on them."""


@eval_group.command("gold-export")
@click.argument("book_id")
@click.option("--chapters", "chapters_spec", default=None, help="1-based chapters, e.g. 1-5,8.")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Gold JSONL to write.",
)
@_books_dir_option
def gold_export(
    book_id: str, chapters_spec: str | None, output: Path, books_dir: Path | None
) -> None:
    """Export gold quote labels from the book's EFFECTIVE attribution (raw + Review edits).

    Correct a few chapters in Review first; every attributed quote in them becomes one gold
    row. Quotes still unattributed are skipped (no speaker to record).
    """
    from seiyuu.attribute.evaluate import describe_conflict, dump_gold, export_gold
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.repository import atomic_write_text
    from seiyuu.services import ServiceError
    from seiyuu.services.evaluation import load_effective_report
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, "attribution.json", "Run `seiyuu attribute` first."
    )
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )
    try:
        report, warnings = load_effective_report(book_dir)
        result = export_gold(book, report, parse_chapters(chapters_spec))
    except (ServiceError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    for warning in warnings:
        click.echo(f"edit overlay: {warning}")
    for conflict in result.conflicts:
        click.echo(f"WARNING: {describe_conflict(conflict)}", err=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, dump_gold(result.rows))
    speakers = len({r.speaker for r in result.rows})
    click.echo(f"gold: {len(result.rows)} quotes, {speakers} speakers")
    if result.skipped_unattributed:
        click.echo(f"  skipped {result.skipped_unattributed} unattributed quote(s)")
    if result.skipped_unaligned:
        click.echo(
            f"  skipped {result.skipped_unaligned} quote(s) with no attributed segment "
            "(chapter not attributed?)"
        )
    click.echo(f"wrote: {output}")


@eval_group.command("run")
@click.argument("book_id")
@click.option(
    "--gold",
    "gold_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--models", required=True, help="Comma-separated model ids, one arm each.")
@click.option("--provider", "provider_id", default="local", show_default=True)
@click.option("--chapters", "chapters_spec", default=None, help="Default: the gold's chapters.")
@click.option("--prompt-version", default=None, help="Default from settings.")
@click.option(
    "--hybrid",
    is_flag=True,
    help="Escalate failing chunks to the anthropic provider (PAID; off unless passed).",
)
@click.option("--fresh", is_flag=True, help="Wipe each arm's own chunk cache before running.")
@click.option(
    "-o",
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory (per-arm subdirs + eval_report.json/.md).",
)
@_books_dir_option
def eval_run(
    book_id: str,
    gold_path: Path,
    models: str,
    provider_id: str,
    chapters_spec: str | None,
    prompt_version: str | None,
    hybrid: bool,
    fresh: bool,
    out_dir: Path,
    books_dir: Path | None,
) -> None:
    """Attribute the gold chapters with each model into isolated dirs and score them.

    The book's own attribution.json / edits / cache are never touched.
    """
    from seiyuu.attribute import AttributionError
    from seiyuu.attribute.evaluate import load_gold
    from seiyuu.gpu import GpuBusyError
    from seiyuu.ingest.models import NormalizedBook
    from seiyuu.services import ServiceError
    from seiyuu.services.evaluation import REPORT_JSON, REPORT_MD, console_summary, run_eval
    from seiyuu.settings import get_settings

    cfg = get_settings()
    book_dir = _resolve_book_dir(
        books_dir or cfg.books_dir, book_id, "normalized.json", "Run `seiyuu ingest` first."
    )
    book = NormalizedBook.model_validate_json(
        (book_dir / "normalized.json").read_text(encoding="utf-8")
    )
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    try:
        gold = load_gold(gold_path)
        report = run_eval(
            book,
            gold,
            cfg=cfg,
            models=model_list,
            out_dir=out_dir,
            provider_id=provider_id,
            chapters=parse_chapters(chapters_spec),
            prompt_version=prompt_version,
            hybrid=hybrid,
            fresh=fresh,
            book_dir=book_dir,
            progress=click.echo,
        )
    except (AttributionError, GpuBusyError, ServiceError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    for line in console_summary(report):
        click.echo(line)
    click.echo(f"wrote: {out_dir / REPORT_JSON}")
    click.echo(f"wrote: {out_dir / REPORT_MD}")
    if report.failed:
        # The reports above still hold every arm that finished; exit non-zero for scripts.
        names = ", ".join(f.model for f in report.failed)
        raise click.ClickException(f"{len(report.failed)} arm(s) failed: {names}")
