"""File-based TTS segment cache (M1; a SQLite index arrives in M2).

Key per SPEC: (engine, engine_model_version, voice_id, settings_hash, seed,
normalized_text_hash). Layout: cache_dir/{key_hash}.wav plus a {key_hash}.json
sidecar holding the full key for debuggability.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from seiyuu.engines import AudioFile
from seiyuu.repository import atomic_write_text
from seiyuu.validate import ValidationResult


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_json(obj: Any) -> str:
    return _sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")))


class SegmentKey(BaseModel):
    engine: str
    engine_model_version: str
    voice_id: str
    settings_hash: str
    seed: int | None
    normalized_text_hash: str

    @classmethod
    def build(
        cls,
        *,
        engine: str,
        engine_model_version: str,
        voice_id: str,
        settings: dict[str, Any],
        seed: int | None,
        normalized_text: str,
    ) -> "SegmentKey":
        return cls(
            engine=engine,
            engine_model_version=engine_model_version,
            voice_id=voice_id,
            settings_hash=_hash_json(settings),
            seed=seed,
            normalized_text_hash=_sha256(normalized_text),
        )

    @property
    def key_hash(self) -> str:
        # 32 hex chars keeps Windows paths short while staying collision-safe.
        return _hash_json(self.model_dump())[:32]


class SegmentCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for(self, key: SegmentKey) -> Path:
        return self.cache_dir / f"{key.key_hash}.wav"

    def get(self, key: SegmentKey) -> Path | None:
        path = self.path_for(key)
        return path if path.is_file() else None

    def put(self, key: SegmentKey, audio: AudioFile) -> Path:
        path = audio.save(self.path_for(key))
        sidecar = path.with_suffix(".json")
        atomic_write_text(sidecar, key.model_dump_json(indent=2))
        return path

    def validation_path(self, key: SegmentKey) -> Path:
        return self.cache_dir / f"{key.key_hash}.validation.json"

    def get_validation(self, key: SegmentKey) -> ValidationResult | None:
        """The cached whisper verdict, so a cache hit keeps its validation data in the manifest."""
        path = self.validation_path(key)
        if not path.is_file():
            return None
        return ValidationResult.model_validate_json(path.read_text(encoding="utf-8"))

    def put_validation(self, key: SegmentKey, result: ValidationResult) -> Path:
        return atomic_write_text(self.validation_path(key), result.model_dump_json(indent=2))
