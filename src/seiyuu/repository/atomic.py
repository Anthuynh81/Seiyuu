"""Atomic file writes: write to a temp file in the target's directory, then os.replace.

A bare ``Path.write_text`` truncates the destination *before* writing, so a crash or a
concurrent reader mid-write sees an empty or half-written file. ``os.replace`` is atomic on
a single filesystem (Windows included), so a reader always sees either the complete old file
or the complete new one — never a torn write. This matters the moment M6 turns the pipeline
into a concurrent server; the CLI got away with plain writes only because it was single-flight.

The temp file is created in the destination's own directory so the final ``os.replace`` stays
on one volume (a cross-volume replace is a copy+delete, which is neither atomic nor allowed on
Windows).
"""

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Atomically write ``data`` to ``path`` (creating parent dirs). Returns ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave an orphan temp file behind on failure; the destination is untouched.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write ``text`` to ``path`` (creating parent dirs). Returns ``path``."""
    return atomic_write_bytes(Path(path), text.encode(encoding))
