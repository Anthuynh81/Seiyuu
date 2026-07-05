"""Book registry: the Click-free single source of truth for "what books exist and how far
along the pipeline each one is".

Until M6 the only way to enumerate books was ``cli._resolve_book_dir``, which resolves ONE id
and raises ``click.ClickException`` — unusable from an API. A book's state is split across two
roots (``books/{id}/`` holds ingest + attribution; ``output/{id}/`` holds assignment + render +
assembly + master), and pipeline stage is inferred purely from marker-file existence. This
module owns that two-root join so no caller hardcodes the split.

Stage is inferred from markers, so it can represent "reached" but NOT "running"/"failed" — the
durable job/status store (a later M6a commit) fills that gap. Reading ``title``/``authors`` here
parses the whole ``normalized.json``; a denormalized per-book summary is a deferred optimization,
so avoid calling ``list_books`` in a hot loop over huge books.
"""

import json
import shutil
from pathlib import Path

from pydantic import BaseModel

# Marker filenames — the on-disk contract mirrored from the stage modules. Kept as local
# literals so this lightweight module doesn't import the heavy stage packages just to list
# books; ``tests/test_repository.py`` asserts they stay in sync with the stage constants.
NORMALIZED_NAME = "normalized.json"
ATTRIBUTION_NAME = "attribution.json"
ASSIGNMENT_NAME = "assignments.json"
MANIFEST_NAME = "manifest.json"
CHAPTERS_DIR = "chapters"


class RepositoryError(Exception):
    """Loud repository failure (unknown book/job, ambiguous prefix, illegal transition)."""


class BookStatus(BaseModel):
    """Per-book pipeline state derived from marker files across both roots."""

    book_id: str
    title: str | None = None
    authors: list[str] = []
    ingested: bool = False
    attributed: bool = False
    assigned: bool = False
    rendered: bool = False
    assembled: bool = False
    mastered: bool = False


def _default_roots(books_dir: Path | None, output_dir: Path | None) -> tuple[Path, Path]:
    if books_dir is not None and output_dir is not None:
        return Path(books_dir), Path(output_dir)
    from seiyuu.settings import get_settings

    cfg = get_settings()
    return Path(books_dir or cfg.books_dir), Path(output_dir or cfg.output_dir)


def _has_output(odir: Path) -> bool:
    """True if ``odir`` shows any output-stage marker (assignment/render/assemble/master)."""
    return (
        (odir / MANIFEST_NAME).is_file()
        or (odir / ASSIGNMENT_NAME).is_file()
        or (odir / f"{odir.name}.m4b").is_file()
        or (odir / CHAPTERS_DIR).is_dir()
    )


def _known_book_ids(books_dir: Path, output_dir: Path) -> set[str]:
    ids: set[str] = set()
    if books_dir.is_dir():
        ids |= {
            d.name for d in books_dir.iterdir() if d.is_dir() and (d / NORMALIZED_NAME).is_file()
        }
    if output_dir.is_dir():
        ids |= {d.name for d in output_dir.iterdir() if d.is_dir() and _has_output(d)}
    return ids


def _read_book_meta(normalized_path: Path) -> tuple[str | None, list[str]]:
    """Title + authors from a normalized.json, or (None, []) if unreadable."""
    try:
        meta = json.loads(normalized_path.read_text(encoding="utf-8")).get("book_meta", {})
    except (OSError, ValueError):
        return None, []
    return meta.get("title"), list(meta.get("authors") or [])


def get_book_status(
    book_id: str,
    *,
    books_dir: Path | None = None,
    output_dir: Path | None = None,
) -> BookStatus:
    """Compute a single book's pipeline status from its markers. ``book_id`` must be exact."""
    books_dir, output_dir = _default_roots(books_dir, output_dir)
    bdir = books_dir / book_id
    odir = output_dir / book_id
    normalized = bdir / NORMALIZED_NAME
    ingested = normalized.is_file()
    title, authors = _read_book_meta(normalized) if ingested else (None, [])
    assembled = (odir / CHAPTERS_DIR).is_dir() and any((odir / CHAPTERS_DIR).glob("*.mp3"))
    return BookStatus(
        book_id=book_id,
        title=title,
        authors=authors,
        ingested=ingested,
        attributed=(bdir / ATTRIBUTION_NAME).is_file(),
        assigned=(odir / ASSIGNMENT_NAME).is_file(),
        rendered=(odir / MANIFEST_NAME).is_file(),
        assembled=assembled,
        mastered=(odir / f"{book_id}.m4b").is_file(),
    )


def list_books(
    *,
    books_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[BookStatus]:
    """Every known book (present under either root), sorted by ``book_id``, with status."""
    books_dir, output_dir = _default_roots(books_dir, output_dir)
    ids = _known_book_ids(books_dir, output_dir)
    return [get_book_status(bid, books_dir=books_dir, output_dir=output_dir) for bid in sorted(ids)]


def resolve_book_id(
    book_id: str,
    *,
    books_dir: Path | None = None,
    output_dir: Path | None = None,
) -> str:
    """Resolve a full id or unambiguous prefix to a canonical ``book_id`` (the Click-free
    counterpart of ``cli._resolve_book_dir``). Raises ``RepositoryError`` on miss/ambiguity."""
    books_dir, output_dir = _default_roots(books_dir, output_dir)
    ids = _known_book_ids(books_dir, output_dir)
    if book_id in ids:
        return book_id
    matches = sorted(i for i in ids if i.startswith(book_id))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        known = ", ".join(sorted(ids)) or "(none)"
        raise RepositoryError(f"book {book_id!r} not found; known: {known}")
    raise RepositoryError(f"book {book_id!r} is ambiguous; candidates: {', '.join(matches)}")


class BookPurgeResult(BaseModel):
    """Outcome of a two-root on-disk purge. ``*_removed`` reports a root that EXISTED and is
    now gone; ``survivors`` lists any paths ``rmtree`` could not delete (on Windows a file a
    concurrent process holds open raises a sharing violation) so the caller can fail loudly
    and keep the delete retryable."""

    book_id: str
    output_removed: bool
    books_removed: bool
    survivors: list[str] = []


def _delete_target(root: Path, book_id: str) -> Path:
    """The book's directory under ``root``, refusing anything that could escape it. A
    traversal here would ``rmtree`` an arbitrary directory, so this is a hard safety gate:
    the id must be a single path component (no separator, no ``..``) whose resolved parent
    IS the root. Never trusts an already-validated id — deletion is irreversible."""
    if not book_id or book_id in (".", ".."):
        raise RepositoryError(f"refusing to delete book with empty/reserved id {book_id!r}")
    if "/" in book_id or "\\" in book_id or "\x00" in book_id:
        raise RepositoryError(f"refusing to delete book {book_id!r}: id contains a path separator")
    if ".." in Path(book_id).parts:
        raise RepositoryError(f"refusing to delete book {book_id!r}: id contains '..'")
    target = root / book_id
    resolved = target.resolve()
    if resolved.name != book_id or resolved.parent != root.resolve():
        raise RepositoryError(f"refusing to delete {target}: not a direct child of {root}")
    return target


def delete_book_trees(
    book_id: str,
    *,
    books_dir: Path | None = None,
    output_dir: Path | None = None,
) -> BookPurgeResult:
    """Purge a book's on-disk state across BOTH roots. A book has no DB row — its identity
    is inferred from marker files under ``output/{id}/`` AND ``books/{id}/`` (see
    ``_known_book_ids``), so a one-root delete leaves a ghost that reappears in the library;
    both must go. Output (render/assemble/master) is removed first, then books (ingest +
    attribution); if the output root cannot be FULLY removed the books root is left
    UNTOUCHED so the book still resolves (via ``books/{id}/normalized.json``) and the delete
    stays retryable — deleting books after a partial output failure would strand the output
    survivors as an unresolvable ghost the library can't surface. A missing root is not an
    error (nothing to remove). Partial failures are COLLECTED (Python 3.11
    ``rmtree(onerror=...)``; ``onexc`` is 3.12+), never raised, so the caller can report
    survivors and retry — this function never touches the voice library or the global jobs DB."""
    books_dir, output_dir = _default_roots(books_dir, output_dir)
    odir = _delete_target(output_dir, book_id)
    bdir = _delete_target(books_dir, book_id)

    survivors: list[str] = []

    def _collect(_func, path, _exc_info) -> None:
        survivors.append(str(path))

    output_existed = odir.is_dir()
    books_existed = bdir.is_dir()
    if output_existed:
        shutil.rmtree(odir, onerror=_collect)
        if survivors:
            # Output could not be fully removed. Stop BEFORE touching the books root so the
            # book still resolves and the caller can retry both roots once the lock clears.
            return BookPurgeResult(
                book_id=book_id, output_removed=False, books_removed=False, survivors=survivors
            )
    if books_existed:
        shutil.rmtree(bdir, onerror=_collect)
    return BookPurgeResult(
        book_id=book_id,
        output_removed=output_existed and not odir.exists(),
        books_removed=books_existed and not bdir.exists(),
        survivors=survivors,
    )
