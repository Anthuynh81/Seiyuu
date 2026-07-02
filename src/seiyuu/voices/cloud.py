"""ElevenLabs cloud-voice + slot manager.

Stock (preset) voices are addressed by their stock id directly and consume no account slot.
Cloned (IVC) voices are created from reference.wav and their cloud handle cached in a central
registry (voices/cloud_voices.json) keyed by library voice_id — a DERIVED cache, regenerable
from reference.wav. ElevenLabs slots are tier-limited, so creation evicts the least-recently-used
seiyuu-managed cloud voice when the account is full, and a handle the account no longer has (slot
reclaimed elsewhere) is transparently recreated. Never errors on voice-not-found.

This module is SDK-free: callers pass an authenticated client object (the ElevenLabs engine
exposes one), so the slot logic stays unit-testable with a fake.
"""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from seiyuu.repository import atomic_write_text, file_lock
from seiyuu.voices.library import VoiceLibraryError
from seiyuu.voices.models import VoiceKind, VoiceMeta

REGISTRY_NAME = "cloud_voices.json"


class CloudVoiceError(Exception):
    """Loud cloud-voice failure (missing reference for an IVC voice, unsupported kind/engine)."""


class CloudVoiceRegistry:
    """LRU registry of seiyuu-managed cloud voice handles (library voice_id → cloud id + seq)."""

    def __init__(self, voices_dir: Path) -> None:
        self.path = Path(voices_dir) / REGISTRY_NAME
        self._data = self._load()

    @contextmanager
    def locked(self, *, timeout: float = 600.0) -> Iterator["CloudVoiceRegistry"]:
        """Cross-process critical section for a whole slot transaction: re-reads the
        registry under an exclusive file lock so two creators can never both count the
        same free slot (ElevenLabs slots are tier-limited and eviction is destructive).
        The default timeout is sized for the slowest legitimate holder — an IVC upload
        of a minutes-long reference clip — not for a quick JSON edit; a waiter timing
        out at 30s would wrongly abort a healthy concurrent render."""
        with file_lock(self.path.with_name(REGISTRY_NAME + ".lock"), timeout=timeout):
            self._data = self._load()  # another process may have written since __init__
            yield self

    def _load(self) -> dict:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"voices": {}, "next_seq": 0}

    def _save(self) -> None:
        # On Windows, os.replace fails with PermissionError while another handle has the
        # JSON open for reading (no FILE_SHARE_DELETE) — e.g. a second process's __init__
        # load racing the lock holder's save. Readers are transient, so retry briefly
        # rather than aborting a slot transaction that may have already evicted a voice.
        for attempt in range(5):
            try:
                atomic_write_text(self.path, json.dumps(self._data, indent=2))
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05)

    def get(self, voice_id: str) -> str | None:
        entry = self._data["voices"].get(voice_id)
        return entry["cloud_id"] if entry else None

    def touch(self, voice_id: str, cloud_id: str) -> None:
        """Record/refresh a handle as most-recently-used (monotonic seq, no wall clock)."""
        self._data["voices"][voice_id] = {"cloud_id": cloud_id, "seq": self._data["next_seq"]}
        self._data["next_seq"] += 1
        self._save()

    def remove(self, voice_id: str) -> None:
        self._data["voices"].pop(voice_id, None)
        self._save()

    def count(self) -> int:
        return len(self._data["voices"])

    def evict_lru(self) -> tuple[str, str] | None:
        """Drop and return the (voice_id, cloud_id) with the lowest seq, or None if empty."""
        voices = self._data["voices"]
        if not voices:
            return None
        victim = min(voices, key=lambda k: voices[k]["seq"])
        cloud_id = voices[victim]["cloud_id"]
        del voices[victim]
        self._save()
        return victim, cloud_id


def _voice_exists(client, cloud_id: str) -> bool:
    try:
        client.voices.get(cloud_id)
        return True
    except Exception:  # not-found or transient: treat as gone and recreate (never error out)
        return False


def _ivc_create(client, name: str, reference: Path) -> str:
    return client.voices.ivc.create(name=name, files=[str(reference)]).voice_id


def _safe_delete(client, cloud_id: str) -> None:
    try:
        client.voices.delete(cloud_id)
    except Exception:  # already gone is fine
        pass


def ensure_cloud_voice(
    meta: VoiceMeta,
    client,
    library,
    *,
    max_slots: int,
    registry: CloudVoiceRegistry | None = None,
) -> str:
    """Return a usable ElevenLabs cloud voice id for `meta`, creating/recreating as needed.

    Preset voices return their stock id (no slot). Cloned voices return a cached IVC handle if it
    still exists, otherwise create one from reference.wav — evicting the LRU seiyuu voice first
    when the account is at its slot limit.
    """
    if meta.engine != "elevenlabs":
        raise CloudVoiceError(
            f"voice {meta.voice_id!r}: not an elevenlabs voice (engine={meta.engine!r})"
        )
    if meta.kind is VoiceKind.PRESET:
        if not meta.preset_id:
            raise CloudVoiceError(f"voice {meta.voice_id!r}: preset voice missing preset_id")
        return meta.preset_id  # stock voice id, no account slot consumed
    if meta.kind is not VoiceKind.CLONED:
        raise CloudVoiceError(
            f"voice {meta.voice_id!r}: kind {meta.kind.value!r} not supported on elevenlabs"
        )
    # consent gates the upload itself: reference.wav is about to ship to a third party,
    # so the attestation (and, for M6a records, its hash binding) must hold NOW
    try:
        library.verify_consent(meta)
    except VoiceLibraryError as exc:
        raise CloudVoiceError(str(exc)) from exc

    registry = registry or CloudVoiceRegistry(library.voices_dir)
    with registry.locked():  # one slot transaction at a time, across processes
        cloud_id = registry.get(meta.voice_id)
        if cloud_id and _voice_exists(client, cloud_id):
            registry.touch(meta.voice_id, cloud_id)  # refresh LRU
            return cloud_id

        reference = library.reference_path(meta.voice_id)
        if not reference.is_file():
            raise CloudVoiceError(
                f"voice {meta.voice_id!r}: {reference} missing; cannot create the cloud voice"
            )
        registry.remove(meta.voice_id)  # drop any stale handle so it isn't counted/evicted
        while registry.count() >= max_slots:
            evicted = registry.evict_lru()
            if evicted is None:
                break
            _safe_delete(client, evicted[1])
        cloud_id = _ivc_create(client, meta.name, reference)
        registry.touch(meta.voice_id, cloud_id)
        return cloud_id
