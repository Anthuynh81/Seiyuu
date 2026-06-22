"""Voice library: file-first I/O over voices/{voice_id}/ (meta.json is the truth).

No SQLite in M3 (matching the M1/M2 file-first convention). The consent gate lives here: a
cloned voice cannot be persisted (or, in M3 §6, rendered) without consent_attested=True.
"""

import re
import secrets
from pathlib import Path

from seiyuu.voices.models import VoiceKind, VoiceMeta

_SLUG = re.compile(r"[^a-z0-9]+")


class VoiceLibraryError(Exception):
    """Loud voice-library failure (missing voice, consent not attested)."""


def slugify(name: str) -> str:
    return _SLUG.sub("_", name.casefold()).strip("_") or "voice"


class VoiceLibrary:
    def __init__(self, voices_dir: Path) -> None:
        self.voices_dir = Path(voices_dir)

    def dir_for(self, voice_id: str) -> Path:
        return self.voices_dir / voice_id

    def meta_path(self, voice_id: str) -> Path:
        return self.dir_for(voice_id) / "meta.json"

    def reference_path(self, voice_id: str) -> Path:
        return self.dir_for(voice_id) / "reference.wav"

    def load(self, voice_id: str) -> VoiceMeta:
        path = self.meta_path(voice_id)
        if not path.is_file():
            raise VoiceLibraryError(f"voice {voice_id!r} not found at {path}")
        return VoiceMeta.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, meta: VoiceMeta) -> Path:
        if meta.kind is VoiceKind.CLONED and not meta.consent_attested:
            raise VoiceLibraryError(
                f"refusing to save cloned voice {meta.voice_id!r} without consent attestation"
            )
        self.dir_for(meta.voice_id).mkdir(parents=True, exist_ok=True)
        path = self.meta_path(meta.voice_id)
        path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        return path

    def list_voices(self) -> list[VoiceMeta]:
        if not self.voices_dir.is_dir():
            return []
        metas = []
        for d in sorted(self.voices_dir.iterdir()):
            if (d / "meta.json").is_file():
                metas.append(self.load(d.name))
        return metas

    def new_voice_id(self, name: str, *, suffix: str | None = None) -> str:
        """A slug from `name` + a short hex suffix, so voice_id != character_id and two voices
        can share a display name. `suffix` is injectable for deterministic tests."""
        return f"{slugify(name)}_{suffix or secrets.token_hex(2)}"
