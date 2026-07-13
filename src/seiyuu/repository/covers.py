"""Cover-art write discipline for a book's output dir.

ONE writer for every path that lands cover art (the upload route, EPUB ingest
extraction): magic-byte validation — the bytes must prove the claimed jpeg/png
type — then an atomic write that evicts the other extension, so a book never
carries two covers and a reader never sees a torn image.
"""

from pathlib import Path

from seiyuu.repository.atomic import atomic_write_bytes

# The two cover files a book may carry (exactly one at a time) and their types.
COVER_TYPES: dict[str, str] = {"cover.jpg": "image/jpeg", "cover.png": "image/png"}

# content-type -> (leading magic bytes, canonical file name).
COVER_MAGIC: dict[str, tuple[bytes, str]] = {
    "image/jpeg": (b"\xff\xd8\xff", "cover.jpg"),
    "image/png": (b"\x89PNG", "cover.png"),
}


class CoverTypeError(ValueError):
    """The bytes do not prove the claimed (or any supported) cover image type."""


def sniff_cover_type(data: bytes) -> str | None:
    """Content type proven by the leading magic bytes; None for anything else."""
    for content_type, (magic, _name) in COVER_MAGIC.items():
        if data.startswith(magic):
            return content_type
    return None


def has_cover(book_output_dir: Path) -> bool:
    """True when the book already carries a cover in either supported extension."""
    return any((Path(book_output_dir) / name).is_file() for name in COVER_TYPES)


def write_cover(book_output_dir: Path, data: bytes, content_type: str) -> Path:
    """Validate and atomically write ``data`` as the book's single cover.

    Raises :class:`CoverTypeError` when ``content_type`` isn't jpeg/png or ``data``
    doesn't start with its magic bytes — before ANY disk mutation, so a rejected
    cover leaves an existing one untouched.
    """
    entry = COVER_MAGIC.get(content_type)
    if entry is None:
        raise CoverTypeError(f"cover must be image/jpeg or image/png, got {content_type!r}")
    magic, target_name = entry
    if not data.startswith(magic):
        raise CoverTypeError("file content does not match its image type")
    odir = Path(book_output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    for other in COVER_TYPES:
        if other != target_name:
            (odir / other).unlink(missing_ok=True)
    return atomic_write_bytes(odir / target_name, data)
