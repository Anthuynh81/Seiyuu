"""Ingest stage: EPUB/PDF → normalized JSON."""

from pathlib import Path

from seiyuu.ingest.common import IngestError, IngestResult
from seiyuu.ingest.epub import parse_epub, write_normalized
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.ingest.pdf import parse_pdf

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
    "parse_book",
    "parse_epub",
    "parse_pdf",
    "write_normalized",
]
