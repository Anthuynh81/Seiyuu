"""Cloud voice + slot manager: stock passthrough, IVC create/cache, recreate, LRU eviction."""

from types import SimpleNamespace

import numpy as np
import pytest

from seiyuu.engines import AudioFile
from seiyuu.voices import (
    CloudVoiceError,
    CloudVoiceRegistry,
    VoiceKind,
    VoiceLibrary,
    VoiceMeta,
    ensure_cloud_voice,
)


class FakeVoices:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.created: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self._n = 0
        self.ivc = SimpleNamespace(create=self._create)

    def get(self, voice_id, **kwargs):
        if voice_id not in self.existing:
            raise RuntimeError("voice not found")
        return SimpleNamespace(voice_id=voice_id)

    def delete(self, voice_id, **kwargs):
        self.deleted.append(voice_id)
        self.existing.discard(voice_id)

    def _create(self, *, name, files, **kwargs):
        self._n += 1
        cloud_id = f"cloud_{name}_{self._n}"
        self.existing.add(cloud_id)
        self.created.append((name, cloud_id))
        return SimpleNamespace(voice_id=cloud_id)


class FakeClient:
    def __init__(self, existing=()):
        self.voices = FakeVoices(existing)


def _preset(lib, voice_id="rachel_x", stock="EXAVITQu"):
    meta = VoiceMeta(
        voice_id=voice_id, name="Rachel", kind=VoiceKind.PRESET, engine="elevenlabs",
        preset_id=stock,
    )  # fmt: skip
    lib.save(meta)
    return meta


def _cloned(lib, voice_id="elena_x", name="Elena", consent=True):
    meta = VoiceMeta(
        voice_id=voice_id, name=name, kind=VoiceKind.CLONED, engine="elevenlabs",
        reference_audio="reference.wav", consent_attested=consent,
    )  # fmt: skip
    if consent:
        lib.save(meta)
    else:  # bypass the save consent gate to test the render-time gate
        d = lib.dir_for(voice_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    AudioFile(samples=np.zeros(2400, dtype=np.float32)).save(lib.reference_path(voice_id))
    return meta


def test_preset_returns_stock_id_no_api(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    meta = _preset(lib, stock="EXAVITQu")
    client = FakeClient()
    assert ensure_cloud_voice(meta, client, lib, max_slots=10) == "EXAVITQu"
    assert client.voices.created == []  # no slot consumed


def test_clone_creates_then_caches(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    meta = _cloned(lib)
    client = FakeClient()
    first = ensure_cloud_voice(meta, client, lib, max_slots=10)
    assert first.startswith("cloud_Elena")
    assert len(client.voices.created) == 1
    # second call reuses the cached handle (it still exists) — no new IVC create
    second = ensure_cloud_voice(meta, client, lib, max_slots=10)
    assert second == first
    assert len(client.voices.created) == 1


def test_recreates_when_handle_reclaimed(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    meta = _cloned(lib)
    registry = CloudVoiceRegistry(lib.voices_dir)
    registry.touch(meta.voice_id, "stale_cloud_id")  # cached, but account doesn't have it
    client = FakeClient()  # existing is empty -> stale handle not found
    cloud_id = ensure_cloud_voice(meta, client, lib, max_slots=10, registry=registry)
    assert cloud_id != "stale_cloud_id"
    assert len(client.voices.created) == 1


def test_evicts_lru_when_slots_full(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    a = _cloned(lib, voice_id="a_x", name="A")
    b = _cloned(lib, voice_id="b_x", name="B")
    client = FakeClient()
    registry = CloudVoiceRegistry(lib.voices_dir)
    cloud_a = ensure_cloud_voice(a, client, lib, max_slots=1, registry=registry)
    cloud_b = ensure_cloud_voice(b, client, lib, max_slots=1, registry=registry)
    assert cloud_a in client.voices.deleted  # A evicted to make room for B
    assert registry.get("a_x") is None and registry.get("b_x") == cloud_b
    assert registry.count() == 1


def test_missing_reference_is_loud(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    meta = _cloned(lib)
    lib.reference_path(meta.voice_id).unlink()  # remove the reference
    with pytest.raises(CloudVoiceError, match="missing"):
        ensure_cloud_voice(meta, FakeClient(), lib, max_slots=10)


def test_cloned_without_consent_is_refused(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    meta = _cloned(lib, consent=False)
    with pytest.raises(CloudVoiceError, match="consent"):
        ensure_cloud_voice(meta, FakeClient(), lib, max_slots=10)


def test_non_elevenlabs_voice_rejected(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    meta = VoiceMeta(
        voice_id="k_x", name="K", kind=VoiceKind.PRESET, engine="kokoro", preset_id="af_heart"
    )
    lib.save(meta)
    with pytest.raises(CloudVoiceError, match="not an elevenlabs voice"):
        ensure_cloud_voice(meta, FakeClient(), lib, max_slots=10)
