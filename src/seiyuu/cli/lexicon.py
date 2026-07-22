"""Per-book pronunciation lexicon commands: show, set, remove, suggest, suggest-ai."""

from pathlib import Path

import click

from seiyuu.cli import main
from seiyuu.cli.common import _edit_books_dir_option, _resolve_book_dir


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
