"""Ingest stage: EPUB/PDF → normalized JSON."""

from pathlib import Path

from seiyuu.ingest.common import IngestError, IngestResult
from seiyuu.ingest.epub import parse_epub, write_normalized
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.ingest.pdf import parse_pdf
from seiyuu.repository.covers import has_cover, sniff_cover_type, write_cover

# One parser per source format; parse_book dispatches on the file suffix.
PARSERS = {".epub": parse_epub, ".pdf": parse_pdf}
SUPPORTED_SUFFIXES = tuple(sorted(PARSERS))


def parse_book(
    path: Path,
    include_items: tuple[str, ...] = (),
    exclude_items: tuple[str, ...] = (),
    split_level: int = 2,
) -> IngestResult:
    """Parse any supported book format by file suffix (.epub or .pdf)."""
    suffix = Path(path).suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        supported = ", ".join(SUPPORTED_SUFFIXES)
        raise IngestError(f"unsupported book format {suffix!r} for {path} (supported: {supported})")
    return parser(
        Path(path),
        include_items=include_items,
        exclude_items=exclude_items,
        split_level=split_level,
    )


def extract_cover_art(result: IngestResult, output_dir: Path) -> Path | None:
    """Write the source's embedded cover into the book's OUTPUT dir (where uploaded
    covers live) through the shared validation path.

    Covers are optional garnish — never fail an ingest over one: no declared cover,
    bytes that aren't jpeg/png, and a cover already on disk (a user upload wins over
    re-ingest) all skip silently by returning None.
    """
    if result.cover is None:
        return None
    book_output_dir = Path(output_dir) / result.book.book_meta.book_id
    if has_cover(book_output_dir):
        return None
    content_type = sniff_cover_type(result.cover)
    if content_type is None:
        return None
    return write_cover(book_output_dir, result.cover, content_type)


__all__ = [
    "PARSERS",
    "SUPPORTED_SUFFIXES",
    "Block",
    "BlockType",
    "BookMeta",
    "Chapter",
    "IngestError",
    "IngestResult",
    "NormalizedBook",
    "extract_cover_art",
    "parse_book",
    "parse_epub",
    "parse_pdf",
    "write_normalized",
]
