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
from typing import IO

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
    ``RepositoryError`` — a hung competitor should surface loudly, not deadlock.
    Built on ``FileLockHandle`` so the descriptor lifecycle lives in one place;
    this wrapper only adds the poll-until-deadline and with-scoped release."""
    handle = FileLockHandle(path)
    deadline = time.monotonic() + timeout
    while not handle.try_acquire():
        if time.monotonic() >= deadline:
            raise RepositoryError(
                f"could not acquire lock {handle.path} within {timeout:.0f}s "
                f"(another seiyuu process holding it?)"
            )
        time.sleep(poll_seconds)
    try:
        yield
    finally:
        handle.release()


class FileLockHandle:
    """A non-blocking exclusive lock on ``path`` whose hold outlives any one with-block.

    ``file_lock`` above waits for the lock and scopes the hold to a with-body; the GPU
    manager needs the opposite on both counts: refuse IMMEDIATELY on contention (a second
    process about to double-load the card must fail loudly, not queue behind a multi-hour
    job) and keep holding across calls while a model stays lazily resident. Same OS
    primitive (each handle owns its own descriptor), so a crashed holder still
    self-releases when its handle dies with the process.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._file: IO[bytes] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._file is not None

    def try_acquire(self) -> bool:
        """Take the lock without waiting; True if now (or already) held by this handle."""
        if self._file is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self._path, "a+b")  # created if missing; content is irrelevant, only the lock
        try:
            f.seek(0)
            if not _try_lock(f.fileno()):
                f.close()
                return False
        except BaseException:  # a seek/fileno failure must not leak the descriptor
            f.close()
            raise
        self._file = f
        return True

    def release(self) -> None:
        """Drop the lock if held; idempotent, closes the descriptor even if unlock fails."""
        f, self._file = self._file, None
        if f is None:
            return
        try:
            f.seek(0)
            _unlock(f.fileno())
        finally:
            f.close()
