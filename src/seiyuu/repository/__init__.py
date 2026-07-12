"""Repository layer: atomic state writes, the book registry, and the durable job store.

Per SPEC the pipeline's durable state should sit behind a repository seam (SQLite + a storage
abstraction). M6a builds it incrementally: (a) atomic writes so a concurrent server can never
read a torn state file, (b) a Click-free book registry that owns the two-root (``books/`` +
``output/``) join and per-book status, and (c) the SQLite (WAL) job store that gives server
runs the queued/running/failed/canceled states marker files cannot express.
"""

from seiyuu.repository.atomic import atomic_write_bytes, atomic_write_text
from seiyuu.repository.books import (
    BookPurgeResult,
    BookStatus,
    RepositoryError,
    delete_book_trees,
    get_book_status,
    list_books,
    resolve_book_id,
)
from seiyuu.repository.jobs import (
    IllegalTransitionError,
    Job,
    JobKind,
    JobNotFoundError,
    JobState,
    JobStore,
    default_jobs_db_path,
)
from seiyuu.repository.lock import FileLockHandle, file_lock

__all__ = [
    "BookPurgeResult",
    "BookStatus",
    "FileLockHandle",
    "IllegalTransitionError",
    "Job",
    "JobKind",
    "JobNotFoundError",
    "JobState",
    "JobStore",
    "RepositoryError",
    "atomic_write_bytes",
    "atomic_write_text",
    "default_jobs_db_path",
    "delete_book_trees",
    "file_lock",
    "get_book_status",
    "list_books",
    "resolve_book_id",
]
