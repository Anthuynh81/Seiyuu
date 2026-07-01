"""Repository layer: atomic state writes + the book registry.

Per SPEC the pipeline's durable state should sit behind a repository seam (SQLite + a storage
abstraction). M6a builds it incrementally, starting here with (a) atomic writes so a concurrent
server can never read a torn state file, and (b) a Click-free book registry that owns the
two-root (``books/`` + ``output/``) join and per-book status. The jobs/status store lands in a
later M6a commit.
"""

from seiyuu.repository.atomic import atomic_write_bytes, atomic_write_text
from seiyuu.repository.books import (
    BookStatus,
    RepositoryError,
    get_book_status,
    list_books,
    resolve_book_id,
)

__all__ = [
    "BookStatus",
    "RepositoryError",
    "atomic_write_bytes",
    "atomic_write_text",
    "get_book_status",
    "list_books",
    "resolve_book_id",
]
