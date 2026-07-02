"""Cross-process file lock (M6a): the mutual exclusion SQLite can't give plain-JSON state.

The cloud-voice registry (voices/cloud_voices.json) is read-modify-write; two processes
(or an API thread racing a job) that both see one free ElevenLabs slot would both create
a voice and blow the tier limit. ``file_lock`` wraps such critical sections with an
OS-level exclusive lock on a sidecar ``.lock`` file — msvcrt on Windows, flock elsewhere —
so it works across processes AND across threads (each holder opens its own descriptor).
The lock file itself is empty and persistent; only the OS lock on it matters, so a crashed
holder releases automatically when its handle dies.
"""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from seiyuu.repository.books import RepositoryError

if os.name == "nt":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:  # pragma: no cover — Windows-native project; kept portable per CLAUDE.md
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def file_lock(path: Path, *, timeout: float = 30.0, poll_seconds: float = 0.05) -> Iterator[None]:
    """Hold an exclusive OS lock on ``path`` for the ``with`` body.

    Blocks up to ``timeout`` seconds waiting for a competing holder, then raises
    ``RepositoryError`` — a hung competitor should surface loudly, not deadlock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")  # created if missing; content is irrelevant, only the lock is
    try:
        f.seek(0)
        deadline = time.monotonic() + timeout
        while not _try_lock(f.fileno()):
            if time.monotonic() >= deadline:
                raise RepositoryError(
                    f"could not acquire lock {path} within {timeout:.0f}s "
                    f"(another seiyuu process holding it?)"
                )
            time.sleep(poll_seconds)
        try:
            yield
        finally:
            f.seek(0)
            _unlock(f.fileno())
    finally:
        f.close()
