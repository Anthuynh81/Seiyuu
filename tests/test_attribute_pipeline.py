"""Attribution pipeline: registry threading, caching, retry, flagging, escalation."""

import pytest

from factories import make_book
from fake_provider import FakeProvider
from seiyuu.attribute import AttributionCache, attribute_book, write_attribution
from seiyuu.attribute.models import (
    AttributionReport,
    CharacterMention,
    ChunkAttribution,
    Segment,
    SegmentType,
)
from seiyuu.attribute.providers import AttributionError, MalformedOutputError


def _dialogue_by(speaker: str):
    """A provider script: each owned block becomes one dialogue line by `speaker`."""

    def script(chunk, registry, attempt):
        return ChunkAttribution(
            segments=[
                Segment(block_id=b.id, type=SegmentType.DIALOGUE, text=b.text, speaker=speaker)
                for b in chunk.owned_blocks
            ],
            characters=[CharacterMention(name=speaker, gender="female")],
        )

    return script


def _paraphrase(chunk, registry, attempt):
    return ChunkAttribution(
        segments=[
            Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text + " (reworded)")
            for b in chunk.owned_blocks
        ]
    )


def test_honest_attribution_builds_registry_and_segments(tmp_path):
    provider = FakeProvider(_dialogue_by("Alice"))
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(make_book(), provider, cache=cache)

    assert report.flagged == []
    assert report.registry.get("alice").gender == "female"
    ch1 = report.chapters[0]
    # heading -> narration, then the two paragraphs as Alice dialogue; scene break absent.
    assert [s.type for s in ch1.segments] == [
        SegmentType.NARRATION,
        SegmentType.DIALOGUE,
        SegmentType.DIALOGUE,
    ]
    assert ch1.segments[0].text == "Chapter 1" and ch1.segments[0].speaker is None
    assert all(s.speaker == "alice" for s in ch1.segments[1:])


def test_second_run_is_fully_cached(tmp_path):
    book = make_book()
    with AttributionCache(tmp_path / "attribution.db") as cache:
        attribute_book(book, FakeProvider(_dialogue_by("Alice")), cache=cache)
        second = FakeProvider(_dialogue_by("Alice"))
        attribute_book(book, second, cache=cache)
    assert second.calls == []  # every chunk served from cache


def test_retry_then_success_is_not_flagged(tmp_path):
    def flaky(chunk, registry, attempt):
        return (
            _paraphrase(chunk, registry, attempt)
            if attempt == 0
            else _dialogue_by("Bob")(chunk, registry, attempt)
        )

    provider = FakeProvider(flaky)
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(make_book(), provider, cache=cache, max_local_retries=2)

    assert report.flagged == []
    # First chunk needed a second attempt; attempts are recorded per (chunk, attempt).
    assert (0, 0) in provider.calls and (0, 1) in provider.calls
    assert report.registry.get("bob") is not None


def test_persistent_failure_flags_and_falls_back_to_narration(tmp_path):
    provider = FakeProvider(_paraphrase)
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(make_book(), provider, cache=cache, max_local_retries=1)

    assert report.flagged, "paraphrasing provider should be flagged for review"
    non_narration = [
        s for c in report.chapters for s in c.segments if s.type != SegmentType.NARRATION
    ]
    assert non_narration == []  # everything fell back to narration
    fallbacks = [s for c in report.chapters for s in c.segments if s.confidence == 0.0]
    assert fallbacks and all(s.speaker is None for s in fallbacks)


def test_hybrid_escalation_recovers_a_failed_chunk(tmp_path):
    local = FakeProvider(_paraphrase)  # never reconstructs
    premium = FakeProvider(_dialogue_by("Cara"), model="premium-1")
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(
            make_book(), local, cache=cache, max_local_retries=1, escalation_provider=premium
        )
    assert report.flagged == []
    assert premium.calls  # escalation actually ran
    assert report.registry.get("cara") is not None


def test_malformed_output_is_flagged_not_fatal(tmp_path):
    def bad(chunk, registry, attempt):
        raise MalformedOutputError("model returned invalid JSON")

    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(make_book(), FakeProvider(bad), cache=cache, max_local_retries=1)

    assert report.flagged, "unusable model output should flag the chunk, not crash"
    non_narration = [
        s for c in report.chapters for s in c.segments if s.type != SegmentType.NARRATION
    ]
    assert non_narration == []  # fell back to verbatim narration


def test_fatal_provider_error_aborts(tmp_path):
    def boom(chunk, registry, attempt):
        raise AttributionError("Ollama truncated output (hit the context window); raise num_ctx")

    with AttributionCache(tmp_path / "attribution.db") as cache:
        with pytest.raises(AttributionError, match="num_ctx"):
            attribute_book(make_book(), FakeProvider(boom), cache=cache, max_local_retries=1)


def test_write_attribution_round_trips(tmp_path):
    provider = FakeProvider(_dialogue_by("Alice"))
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(make_book(), provider, cache=cache)
    path = write_attribution(report, tmp_path)
    assert path.name == "attribution.json"
    reloaded = AttributionReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded.registry.get("alice") is not None
