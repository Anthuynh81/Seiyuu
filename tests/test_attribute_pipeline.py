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
        report = attribute_book(make_book(), provider, cache=cache, narration_fast_path=False)

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
        attribute_book(
            book, FakeProvider(_dialogue_by("Alice")), cache=cache, narration_fast_path=False
        )
        second = FakeProvider(_dialogue_by("Alice"))
        attribute_book(book, second, cache=cache, narration_fast_path=False)
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
        report = attribute_book(
            make_book(), provider, cache=cache, max_local_retries=2, narration_fast_path=False
        )

    assert report.flagged == []
    # First chunk needed a second attempt; attempts are recorded per (chunk, attempt).
    assert (0, 0) in provider.calls and (0, 1) in provider.calls
    assert report.registry.get("bob") is not None


def test_persistent_failure_flags_and_falls_back_to_narration(tmp_path):
    provider = FakeProvider(_paraphrase)
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(
            make_book(), provider, cache=cache, max_local_retries=1, narration_fast_path=False
        )

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
            make_book(), local, cache=cache, max_local_retries=1,
            escalation_provider=premium, narration_fast_path=False,
        )  # fmt: skip
    assert report.flagged == []
    assert premium.calls  # escalation actually ran
    assert report.registry.get("cara") is not None


def test_drop_superseded_notes_removes_resolved_not_merging():
    from seiyuu.attribute.pipeline import _drop_superseded_notes

    notes = [
        "not merging 'Darcy' with existing ['Mr. Darcy'] (flagged for review, not auto-applied)",
        "not merging 'Bennet' with existing ['Mr. Bennet']",
        "some unrelated note",
    ]
    # The alias post-pass merged Darcy but not Bennet -> only the Darcy note is dropped.
    assert _drop_superseded_notes(notes, {"Darcy"}) == notes[1:]
    assert _drop_superseded_notes(notes, set()) == notes


def test_malformed_output_is_flagged_not_fatal(tmp_path):
    def bad(chunk, registry, attempt):
        raise MalformedOutputError("model returned invalid JSON")

    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(
            make_book(), FakeProvider(bad), cache=cache, max_local_retries=1,
            narration_fast_path=False,
        )  # fmt: skip

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
            attribute_book(
                make_book(), FakeProvider(boom), cache=cache, max_local_retries=1,
                narration_fast_path=False,
            )  # fmt: skip


def test_write_attribution_round_trips(tmp_path):
    provider = FakeProvider(_dialogue_by("Alice"))
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(make_book(), provider, cache=cache, narration_fast_path=False)
    path = write_attribution(report, tmp_path)
    assert path.name == "attribution.json"
    reloaded = AttributionReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded.registry.get("alice") is not None


def _fail_if_called(chunk, registry, attempt):
    raise AssertionError("provider must not be consulted for a pure-narration chunk")


def test_narration_fast_path_skips_the_llm(tmp_path):
    # make_book()'s text contains no quoted spans, so every chunk is pure narration:
    # the model provably cannot affect its segments and must not be called at all.
    provider = FakeProvider(_fail_if_called)
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(make_book(), provider, cache=cache)

    assert provider.calls == []
    assert report.flagged == []
    segs = [s for c in report.chapters for s in c.segments]
    assert segs and all(s.type is SegmentType.NARRATION and s.speaker is None for s in segs)
    assert all(s.confidence == 1.0 for s in segs)  # genuine prose, not the 0.0 fallback


def test_narration_fast_path_result_is_not_cached(tmp_path):
    with AttributionCache(tmp_path / "attribution.db") as cache:
        puts = []
        original = cache.put
        cache.put = lambda key, attribution: (puts.append(key), original(key, attribution))[-1]
        attribute_book(make_book(), FakeProvider(_fail_if_called), cache=cache)
        assert puts == []  # the cache stays a record of actual LLM output


def test_narration_fast_path_matches_the_llm_path(tmp_path):
    # Byte-equivalence: a compliant model on a quote-free chunk yields all-narration
    # entries; the synthesized result must be segment-identical to that provider run.
    def compliant_narration(chunk, registry, attempt):
        return ChunkAttribution(
            segments=[
                Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text)
                for b in chunk.owned_blocks
            ]
        )

    book = make_book()
    with AttributionCache(tmp_path / "a.db") as cache:
        via_llm = attribute_book(
            book, FakeProvider(compliant_narration), cache=cache, narration_fast_path=False
        )
    with AttributionCache(tmp_path / "b.db") as cache:
        fast = attribute_book(book, FakeProvider(_fail_if_called), cache=cache)
    assert [c.segments for c in fast.chapters] == [c.segments for c in via_llm.chapters]
    assert [c.segment_emotions for c in fast.chapters] == [
        c.segment_emotions for c in via_llm.chapters
    ]


def test_quoted_chunk_still_reaches_the_llm(tmp_path):
    # One block with real quoted dialogue: the fast path must stand aside for its chunk.
    book = make_book()
    quoted_text = 'She said, "Hello there."'
    book.chapters[0].blocks[1].text = quoted_text

    def label_quotes(chunk, registry, attempt):
        segments = []
        for b in chunk.owned_blocks:
            if b.text == quoted_text:
                segments.append(
                    Segment(block_id=b.id, type=SegmentType.NARRATION, text="She said, ")
                )
                segments.append(
                    Segment(
                        block_id=b.id,
                        type=SegmentType.DIALOGUE,
                        text='"Hello there."',
                        speaker="Alice",
                    )
                )
            else:
                segments.append(Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text))
        return ChunkAttribution(segments=segments)

    provider = FakeProvider(label_quotes)
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(book, provider, cache=cache)

    assert provider.calls  # the dialogue-bearing chunk went to the model
    dialogue = [s for c in report.chapters for s in c.segments if s.type is SegmentType.DIALOGUE]
    assert len(dialogue) == 1 and dialogue[0].text == '"Hello there."'
