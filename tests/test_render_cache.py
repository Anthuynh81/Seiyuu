import numpy as np
import pytest

from seiyuu.engines import CANONICAL_SAMPLE_RATE, AudioFile
from seiyuu.render import SegmentCache, SegmentKey
from seiyuu.render import cache as cache_mod


def make_key(**overrides) -> SegmentKey:
    kwargs = dict(
        engine="kokoro",
        engine_model_version="kokoro-0.9.4",
        voice_id="af_heart",
        settings={"speed": 1.0},
        seed=41172,
        normalized_text="Hello there.",
    )
    kwargs.update(overrides)
    return SegmentKey.build(**kwargs)


def test_identical_inputs_same_key() -> None:
    assert make_key().key_hash == make_key().key_hash


def test_settings_order_insensitive() -> None:
    a = make_key(settings={"speed": 1.0, "x": 2})
    b = make_key(settings={"x": 2, "speed": 1.0})
    assert a.key_hash == b.key_hash


def test_every_field_changes_key() -> None:
    base = make_key().key_hash
    assert make_key(engine="other").key_hash != base
    assert make_key(engine_model_version="kokoro-9.9.9").key_hash != base
    assert make_key(voice_id="am_adam").key_hash != base
    assert make_key(settings={"speed": 1.2}).key_hash != base
    assert make_key(seed=7).key_hash != base
    assert make_key(normalized_text="Hello there!").key_hash != base


def test_cache_roundtrip(tmp_path) -> None:
    cache = SegmentCache(tmp_path / "cache")
    key = make_key()
    assert cache.get(key) is None

    audio = AudioFile(samples=np.zeros(2400, dtype=np.float32))
    path = cache.put(key, audio)
    assert cache.get(key) == path
    assert path.name == f"{key.key_hash}.wav"
    sidecar = path.with_suffix(".json")
    assert sidecar.is_file()
    assert SegmentKey.model_validate_json(sidecar.read_text(encoding="utf-8")) == key
    assert cache.get(make_key(seed=7)) is None


def test_put_is_crash_atomic(tmp_path, monkeypatch) -> None:
    """A failed put must leave neither a final wav (get() would take it as a hit against
    an unchanging key, poisoning the segment forever) nor a stray temp file."""
    cache = SegmentCache(tmp_path / "cache")
    key = make_key()
    audio = AudioFile(samples=np.zeros(2400, dtype=np.float32))

    def boom(src, dst):
        raise OSError("simulated crash at publish")

    monkeypatch.setattr(cache_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        cache.put(key, audio)
    assert cache.get(key) is None  # no torn wav at the final name
    assert list((tmp_path / "cache").iterdir()) == []  # no orphan temp either

    monkeypatch.undo()
    cache.put(key, audio)
    assert cache.get(key) is not None
    assert not any(p.name.endswith(".part.wav") for p in (tmp_path / "cache").iterdir())


def test_canonical_rate_constant_unchanged() -> None:
    # the cache stores canonical WAVs; guard the constant the policy hangs on
    assert CANONICAL_SAMPLE_RATE == 24_000
