"""F2 — lazy forced alignment (render/align.ensure_words) + the {key_hash}.words.json cache
sidecar. No live whisper: a fake aligner counts transcribe_words calls so we can prove the hit
path never re-transcribes. Real canonical wavs are written so soundfile reports a true duration.
"""

import threading

import numpy as np
import pytest

from seiyuu.engines import AudioFile
from seiyuu.render import ensure_words, words_sidecar_for_wav
from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.validate import SegmentWords, WordTiming


class FakeAligner:
    """Stands in for Validator: returns scripted words and counts transcribe_words calls."""

    def __init__(self, words: list[WordTiming]) -> None:
        self._words = words
        self.calls = 0

    def transcribe_words(self, wav_path):
        self.calls += 1
        return list(self._words)


def _make_wav(path, seconds: float = 0.5) -> None:
    AudioFile(samples=np.zeros(int(seconds * 24_000), dtype=np.float32)).save(path)


def _words() -> list[WordTiming]:
    return [WordTiming(start=0.0, end=0.2, word=" hi"), WordTiming(start=0.2, end=0.5, word=" you")]


def _key(**overrides) -> SegmentKey:
    kwargs = dict(
        engine="kokoro",
        engine_model_version="kokoro-0.9.4",
        voice_id="af_heart",
        settings={"speed": 1.0},
        seed=41172,
        normalized_text="Hi you.",
    )
    kwargs.update(overrides)
    return SegmentKey.build(**kwargs)


# -- cache sidecar ------------------------------------------------------------------------


def test_words_sidecar_roundtrip(tmp_path):
    cache = SegmentCache(tmp_path / "cache")
    cache.cache_dir.mkdir(parents=True)
    key = _key()
    assert cache.get_words(key) is None
    sw = SegmentWords(words=_words(), audio_duration=0.5)
    path = cache.put_words(key, sw)
    assert path.name == f"{key.key_hash}.words.json"
    assert cache.get_words(key) == sw


def test_words_path_agrees_with_wav_stem(tmp_path):
    cache = SegmentCache(tmp_path / "cache")
    key = _key()
    # the endpoint derives the sidecar from the wav path alone; it MUST match put_words' target
    assert words_sidecar_for_wav(cache.path_for(key)) == cache.words_path(key)


def test_words_sidecar_is_crash_atomic(tmp_path, monkeypatch):
    cache = SegmentCache(tmp_path / "cache")
    cache.cache_dir.mkdir(parents=True)
    key = _key()
    from seiyuu import repository

    def boom(src, dst):
        raise OSError("simulated crash at publish")

    monkeypatch.setattr(repository.atomic.os, "replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        cache.put_words(key, SegmentWords(words=_words(), audio_duration=0.5))
    assert cache.get_words(key) is None  # no torn sidecar at the final name
    assert not any(p.name.endswith(".words.json") for p in cache.cache_dir.iterdir())


# -- ensure_words orchestrator ------------------------------------------------------------


def test_ensure_words_miss_then_hit(tmp_path):
    wav = tmp_path / "abc123.wav"
    _make_wav(wav, seconds=0.5)
    aligner = FakeAligner(_words())
    lock = threading.Lock()

    first = ensure_words(wav, aligner, lock)
    assert [w.word for w in first.words] == [" hi", " you"]
    assert abs(first.audio_duration - 0.5) < 0.01
    assert aligner.calls == 1
    assert words_sidecar_for_wav(wav).is_file()

    second = ensure_words(wav, aligner, lock)  # hit: sidecar already on disk
    assert second == first
    assert aligner.calls == 1  # NOT re-transcribed


def test_ensure_words_recomputes_for_a_new_hash(tmp_path):
    # a re-render mints a new key_hash -> new wav name -> a fresh alignment, never stale timings
    aligner = FakeAligner(_words())
    lock = threading.Lock()
    wav_a = tmp_path / "hashA.wav"
    wav_b = tmp_path / "hashB.wav"
    _make_wav(wav_a)
    _make_wav(wav_b)
    ensure_words(wav_a, aligner, lock)
    ensure_words(wav_b, aligner, lock)
    assert aligner.calls == 2
    assert words_sidecar_for_wav(wav_a).is_file()
    assert words_sidecar_for_wav(wav_b).is_file()


def test_ensure_words_serializes_under_lock(tmp_path):
    """The shared lock must be taken on the compute path — CTranslate2 is not concurrency-safe.
    A blocking aligner run on many threads must never overlap inside transcribe_words."""
    wav = tmp_path / "seg.wav"
    _make_wav(wav)

    overlap = {"max": 0, "cur": 0}
    barrier_lock = threading.Lock()

    class SlowAligner:
        calls = 0

        def transcribe_words(self, wav_path):
            with barrier_lock:
                overlap["cur"] += 1
                overlap["max"] = max(overlap["max"], overlap["cur"])
            # hold long enough that a second thread would overlap if the lock didn't serialize
            import time

            time.sleep(0.05)
            with barrier_lock:
                overlap["cur"] -= 1
                SlowAligner.calls += 1
            return _words()

    aligner = SlowAligner()
    lock = threading.Lock()
    # distinct sidecars would let both compute; same wav means the re-check makes only one compute
    threads = [threading.Thread(target=ensure_words, args=(wav, aligner, lock)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlap["max"] == 1  # never two transcriptions at once
    assert SlowAligner.calls == 1  # re-check inside the lock means exactly one compute


def test_ensure_words_recomputes_on_corrupt_sidecar(tmp_path):
    # a torn/corrupt sidecar is a miss, not a 500: recompute and atomically overwrite it
    wav = tmp_path / "seg.wav"
    _make_wav(wav)
    words_sidecar_for_wav(wav).write_text("{not valid json", encoding="utf-8")
    aligner = FakeAligner(_words())

    result = ensure_words(wav, aligner, threading.Lock())
    assert [w.word for w in result.words] == [" hi", " you"]
    assert aligner.calls == 1  # recomputed despite the corrupt file
    assert (
        SegmentWords.model_validate_json(words_sidecar_for_wav(wav).read_text(encoding="utf-8"))
        == result
    )  # the corrupt sidecar was replaced with valid content
