"""SQLite attribution cache (WAL), one DB per book at ``books/{book_id}/attribution.db``.

Cache key (SPEC stage 2): (book, chapter, chunk_hash, provider_id, model_id,
prompt_version). Switching provider/model/prompt naturally misses without clobbering
other providers' rows, which is exactly what the Qwen-vs-Gemma bake-off needs. Stores the
validated :class:`ChunkAttribution` JSON only — metadata and text, never audio blobs. All
access goes through this repository layer.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from seiyuu.attribute.models import ChunkAttribution

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attribution_chunks (
    book_id        TEXT NOT NULL,
    chapter_index  INTEGER NOT NULL,
    chunk_hash     TEXT NOT NULL,
    provider_id    TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    payload        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (book_id, chapter_index, chunk_hash, provider_id, model_id, prompt_version)
)
"""


class ChunkCacheKey(BaseModel, frozen=True):
    book_id: str
    chapter_index: int
    chunk_hash: str
    provider_id: str
    model_id: str
    prompt_version: str

    def _columns(self) -> tuple:
        return (
            self.book_id,
            self.chapter_index,
            self.chunk_hash,
            self.provider_id,
            self.model_id,
            self.prompt_version,
        )


class AttributionCache:
    """Repository over the per-book SQLite attribution DB. Use as a context manager."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, key: ChunkCacheKey) -> ChunkAttribution | None:
        row = self._conn.execute(
            "SELECT payload FROM attribution_chunks WHERE "
            "book_id=? AND chapter_index=? AND chunk_hash=? AND "
            "provider_id=? AND model_id=? AND prompt_version=?",
            key._columns(),
        ).fetchone()
        return ChunkAttribution.model_validate_json(row[0]) if row else None

    def put(self, key: ChunkCacheKey, attribution: ChunkAttribution) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO attribution_chunks "
            "(book_id, chapter_index, chunk_hash, provider_id, model_id, prompt_version, "
            "payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (*key._columns(), attribution.model_dump_json(), datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AttributionCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
