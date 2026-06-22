"""Attribution cache: round-trip, provider/model/prompt isolation, WAL mode."""

from seiyuu.attribute.cache import AttributionCache, ChunkCacheKey
from seiyuu.attribute.models import ChunkAttribution, Segment, SegmentType


def _attr(text: str) -> ChunkAttribution:
    return ChunkAttribution(
        segments=[Segment(block_id="ch001_b0001", type=SegmentType.NARRATION, text=text)]
    )


def _key(**over) -> ChunkCacheKey:
    base = dict(
        book_id="b",
        chapter_index=1,
        chunk_hash="abc",
        provider_id="local",
        model_id="qwen3.5:9b",
        prompt_version="v1",
    )
    base.update(over)
    return ChunkCacheKey(**base)


def test_round_trip(tmp_path):
    with AttributionCache(tmp_path / "attribution.db") as cache:
        assert cache.get(_key()) is None
        cache.put(_key(), _attr("Hello."))
        got = cache.get(_key())
    assert got.segments[0].text == "Hello."


def test_provider_and_model_isolation(tmp_path):
    with AttributionCache(tmp_path / "attribution.db") as cache:
        cache.put(_key(model_id="qwen3.5:9b"), _attr("qwen"))
        cache.put(_key(model_id="gemma4:8b"), _attr("gemma"))
        cache.put(_key(provider_id="anthropic"), _attr("claude"))

        assert cache.get(_key(model_id="qwen3.5:9b")).segments[0].text == "qwen"
        assert cache.get(_key(model_id="gemma4:8b")).segments[0].text == "gemma"
        assert cache.get(_key(provider_id="anthropic")).segments[0].text == "claude"
        # A different prompt version is a clean miss, not a stale hit.
        assert cache.get(_key(prompt_version="v2")) is None


def test_put_replaces_existing(tmp_path):
    with AttributionCache(tmp_path / "attribution.db") as cache:
        cache.put(_key(), _attr("first"))
        cache.put(_key(), _attr("second"))
        assert cache.get(_key()).segments[0].text == "second"


def test_uses_wal_journal_mode(tmp_path):
    with AttributionCache(tmp_path / "attribution.db") as cache:
        mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
