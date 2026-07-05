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

from pydantic import BaseModel, TypeAdapter

from seiyuu.attribute.models import ChunkAttribution, PairVerdict

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
);
CREATE TABLE IF NOT EXISTS alias_adjudications (
    book_id                    TEXT NOT NULL,
    provider_id                TEXT NOT NULL,
    model_id                   TEXT NOT NULL,
    adjudication_prompt_version TEXT NOT NULL,
    candidates_digest          TEXT NOT NULL,
    payload                    TEXT NOT NULL,
    created_at                 TEXT NOT NULL,
    PRIMARY KEY (book_id, provider_id, model_id, adjudication_prompt_version, candidates_digest)
)
"""

# The list[PairVerdict] payload is (de)serialized through this adapter so a cache hit
# replays the exact verdicts the LLM returned — no re-billing on an unchanged candidate set.
_VERDICTS = TypeAdapter(list[PairVerdict])


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


class AdjudicationCacheKey(BaseModel, frozen=True):
    """Per-book alias-adjudication cache key (mirrors :class:`ChunkCacheKey`).

    ``candidates_digest`` is a deterministic hash over the sorted candidate pairs (see
    ``adjudicate.candidates_digest``); because the registry is rebuilt deterministically from
    the cached chunks, the digest is stable across reruns, so the LLM fires ONLY when the
    candidate set genuinely changes and ``attribution.json`` never churns on a no-op rerun.
    """

    book_id: str
    provider_id: str
    model_id: str
    adjudication_prompt_version: str
    candidates_digest: str

    def _columns(self) -> tuple:
        return (
            self.book_id,
            self.provider_id,
            self.model_id,
            self.adjudication_prompt_version,
            self.candidates_digest,
        )


class AttributionCache:
    """Repository over the per-book SQLite attribution DB. Use as a context manager."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
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

    def get_adjudication(self, key: AdjudicationCacheKey) -> list[PairVerdict] | None:
        row = self._conn.execute(
            "SELECT payload FROM alias_adjudications WHERE "
            "book_id=? AND provider_id=? AND model_id=? AND "
            "adjudication_prompt_version=? AND candidates_digest=?",
            key._columns(),
        ).fetchone()
        return _VERDICTS.validate_json(row[0]) if row else None

    def put_adjudication(self, key: AdjudicationCacheKey, verdicts: list[PairVerdict]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO alias_adjudications "
            "(book_id, provider_id, model_id, adjudication_prompt_version, candidates_digest, "
            "payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                *key._columns(),
                _VERDICTS.dump_json(verdicts).decode(),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AttributionCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
