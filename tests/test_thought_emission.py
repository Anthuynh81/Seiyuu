"""Thought emission: ingest italic capture, the candidate predicate, THOUGHT assembly, the
reconstruction invariant, and the opt-in flag — all fixture-only (no live LLM).

The cardinal precision failure is labeling emphasis (or ordinary narration) as thought. The
suite pins the guards that make that impossible: prose-only sub-split (dialogue emphasis
stays dialogue), the substantiality threshold, deterministic-candidate + LLM-confirm, and
degrade-to-narration on a thinkerless / low-confidence / unconfirmed verdict.
"""

import json

import pytest
from pydantic import ValidationError

from seiyuu.attribute.cache import ChunkCacheKey
from seiyuu.attribute.chunking import chunk_blocks
from seiyuu.attribute.models import (
    CharacterRegistry,
    ChunkAttribution,
    Segment,
    SegmentType,
)
from seiyuu.attribute.providers.local import OllamaProvider
from seiyuu.attribute.spans import (
    is_italic_thought_candidate,
    split_block_spans,
    thought_candidate_spans,
)
from seiyuu.attribute.validate import reconstructs_block
from seiyuu.gpu import GpuResourceManager
from seiyuu.ingest.epub import _extract_doc_blocks
from seiyuu.ingest.models import Block, BlockType
from seiyuu.settings import get_settings

PROMPTS_DIR = get_settings().prompts_dir


# --------------------------------------------------------------------------------------
# Ingest: single-pass italic walker keeps offsets aligned to the FINAL collapsed text.
# --------------------------------------------------------------------------------------


def _para_blocks(body: str) -> list:
    return _extract_doc_blocks(f"<html><body>{body}</body></html>".encode())


def test_ingest_italic_offsets_slice_the_collapsed_text():
    blocks = _para_blocks("<p>She paused. <em>I must leave now.</em></p>")
    assert len(blocks) == 1
    block = blocks[0]
    assert block.text == "She paused. I must leave now."
    assert len(block.italic_spans) == 1
    start, end = block.italic_spans[0]
    assert block.text[start:end] == "I must leave now."


def test_ingest_whitespace_heavy_italic_run_still_slices_correctly():
    # Newlines/tabs inside the <em> and around it are collapsed; the offsets must index the
    # post-collapse/strip text, not the raw get_text().
    blocks = _para_blocks("<p>She paused.<em>\n\t  I must leave now.  \n</em> Onward.</p>")
    block = blocks[0]
    assert block.text == "She paused. I must leave now. Onward."
    ((start, end),) = block.italic_spans
    assert block.text[start:end] == "I must leave now."


def test_ingest_i_tag_captured_and_no_italics_yields_empty():
    (italic,) = _para_blocks("<p>He read <i>The Great Gatsby</i> twice.</p>")
    s, e = italic.italic_spans[0]
    assert italic.text[s:e] == "The Great Gatsby"
    (plain,) = _para_blocks("<p>Just ordinary narration here.</p>")
    assert plain.italic_spans == []


def test_ingest_italics_survive_normalized_round_trip():
    (block,) = _para_blocks("<p>She paused. <em>I must leave now.</em></p>")
    b = Block(
        id="ch001_b0001", type=BlockType.PARAGRAPH, text=block.text, italic_spans=block.italic_spans
    )
    reloaded = Block.model_validate_json(b.model_dump_json())
    assert reloaded.italic_spans == b.italic_spans


# --------------------------------------------------------------------------------------
# Block schema: additive italic_spans, validated; old normalized.json still loads.
# --------------------------------------------------------------------------------------


def test_block_accepts_default_and_valid_spans():
    assert Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="hello").italic_spans == []
    ok = Block(
        id="ch001_b0001", type=BlockType.PARAGRAPH, text="hello world", italic_spans=[(0, 5)]
    )
    assert ok.italic_spans == [(0, 5)]


def test_old_normalized_json_without_key_validates_to_empty():
    block = Block.model_validate({"id": "ch001_b0001", "type": "paragraph", "text": "legacy"})
    assert block.italic_spans == []


@pytest.mark.parametrize(
    "spans",
    [
        [(0, 100)],  # out of range
        [(3, 1)],  # start >= end
        [(5, 8), (0, 3)],  # unsorted
        [(0, 5), (3, 8)],  # overlapping
    ],
)
def test_block_rejects_bad_italic_spans(spans):
    with pytest.raises(ValidationError):
        Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="0123456789", italic_spans=spans)


def test_scene_break_rejects_italic_spans():
    with pytest.raises(ValidationError):
        Block(id="ch001_b0001", type=BlockType.SCENE_BREAK, text="", italic_spans=[(0, 1)])


# --------------------------------------------------------------------------------------
# Candidate predicate (D5) + prose-only sub-split (guardrail #1).
# --------------------------------------------------------------------------------------


def test_predicate_accepts_substantial_and_rejects_short_or_emphasis():
    region = "She froze. I have to get out of here. Her heart raced."
    assert is_italic_thought_candidate("I have to get out of here.", region)
    # single/short italic word -> emphasis, not thought
    assert not is_italic_thought_candidate("never", region)
    assert not is_italic_thought_candidate("so tired", region)
    # a multi-word title mid-sentence (no sentence end, not near-whole) -> not a candidate
    assert not is_italic_thought_candidate("The Great Gatsby", region)


def test_join_reproduces_text_for_every_partition():
    cases = [
        ("plain narration, no italics", []),
        ("She froze. I have to get out of here. Her heart raced.", [(11, 37)]),
        ('He said, "I never do that," and left.', []),
    ]
    for text, spans in cases:
        parts = thought_candidate_spans("ch001_b0001", text, spans)
        assert "".join(p.text for p in parts) == text


def test_prose_thought_run_becomes_candidate():
    text = "She froze. I have to get out of here. Her heart raced."
    start = text.index("I have to")
    end = text.index(" Her heart")
    spans = thought_candidate_spans("ch001_b0001", text, [(start, end)])
    cands = [s for s in spans if s.candidate_id]
    assert len(cands) == 1
    assert cands[0].text == "I have to get out of here."
    assert cands[0].candidate_id == f"ch001_b0001:{start}"


def test_single_italic_word_is_not_a_candidate():
    text = "He was never coming back."
    start = text.index("never")
    spans = thought_candidate_spans("ch001_b0001", text, [(start, start + len("never"))])
    assert all(s.candidate_id is None for s in spans)


def test_italic_emphasis_inside_a_quote_never_sub_splits():
    # The cardinal guard: an italic run inside dialogue stays part of the DIALOGUE span.
    text = 'He said, "I would never do that here," coldly.'
    run = "would never do that here"
    start = text.index(run)
    spans = thought_candidate_spans("ch001_b0001", text, [(start, start + len(run))])
    assert all(s.candidate_id is None for s in spans)
    assert any(s.quoted and "would never do that here" in s.text for s in spans)


def test_italic_run_straddling_a_quote_boundary_is_ignored():
    text = 'Prose here "and dialogue" more.'
    # a run from prose into the quote is not wholly within one prose region
    start = text.index("here")
    end = text.index("dialogue") + len("dialogue")
    spans = thought_candidate_spans("ch001_b0001", text, [(start, end)])
    assert all(s.candidate_id is None for s in spans)
    assert "".join(s.text for s in spans) == text


def test_thought_off_partition_matches_split_block_spans():
    # italic_spans=[] must reproduce the quote-only split exactly (byte-identical thought-off).
    text = 'He paused. "Hello," she said.'
    parts = thought_candidate_spans("ch001_b0001", text, [])
    assert [p.text for p in parts] == split_block_spans(text)


# --------------------------------------------------------------------------------------
# Assembly through the real provider (base.attribute_chunk) with an injected transport.
# --------------------------------------------------------------------------------------

_THOUGHT_TEXT = "She froze. I have to get out of here. Her heart raced."
_THOUGHT_START = _THOUGHT_TEXT.index("I have to")
_THOUGHT_CID = f"ch001_b0001:{_THOUGHT_START}"


def _thought_block() -> Block:
    end = _THOUGHT_TEXT.index(" Her heart")
    return Block(
        id="ch001_b0001",
        type=BlockType.PARAGRAPH,
        text=_THOUGHT_TEXT,
        italic_spans=[(_THOUGHT_START, end)],
    )


def _fake_post(labels: dict):
    def post(url, payload, timeout):
        return {"message": {"content": json.dumps(labels)}, "done_reason": "stop"}

    return post


def _run(labels: dict, *, emit_thoughts: bool) -> ChunkAttribution:
    provider = OllamaProvider(
        model="m",
        prompts_dir=PROMPTS_DIR,
        prompt_version="v4" if emit_thoughts else "v3",
        emit_thoughts=emit_thoughts,
        post=_fake_post(labels),
    )
    chunk = chunk_blocks([_thought_block()], overlap_blocks=0)[0]
    return provider.attribute_chunk(chunk, CharacterRegistry())


def _reconstructs(result: ChunkAttribution) -> bool:
    return reconstructs_block(_THOUGHT_TEXT, [s.text for s in result.segments])


def test_confirmed_candidate_with_thinker_becomes_thought_segment():
    labels = {
        "blocks": [{"block_id": "ch001_b0001", "speaker": None}],
        "thoughts": [
            {"candidate_id": _THOUGHT_CID, "is_thought": True, "thinker": "Mira", "confidence": 0.9}
        ],
    }
    result = _run(labels, emit_thoughts=True)
    thoughts = [s for s in result.segments if s.type is SegmentType.THOUGHT]
    assert len(thoughts) == 1
    assert thoughts[0].text == "I have to get out of here."  # verbatim source slice, no markers
    assert thoughts[0].speaker == "Mira"
    assert _reconstructs(result)  # thought text is a plain slice; reconstruction holds


@pytest.mark.parametrize(
    "verdict",
    [
        {"candidate_id": _THOUGHT_CID, "is_thought": False, "thinker": "Mira", "confidence": 0.9},
        {"candidate_id": _THOUGHT_CID, "is_thought": True, "thinker": None, "confidence": 0.9},
        {"candidate_id": _THOUGHT_CID, "is_thought": True, "thinker": "Mira", "confidence": 0.2},
        {
            "candidate_id": "ch001_b0001:999",
            "is_thought": True,
            "thinker": "Mira",
            "confidence": 1.0,
        },
    ],
)
def test_unconfirmed_thinkerless_lowconf_or_unknown_id_degrades_to_narration(verdict):
    labels = {"blocks": [{"block_id": "ch001_b0001", "speaker": None}], "thoughts": [verdict]}
    result = _run(labels, emit_thoughts=True)
    assert all(s.type is SegmentType.NARRATION for s in result.segments)
    assert _reconstructs(result)  # degrade path is byte-identical


def test_emit_thoughts_off_ignores_candidates_and_verdicts():
    # Same block + a confirming verdict, but thought-off: no THOUGHT, output identical to v3.
    labels = {
        "blocks": [{"block_id": "ch001_b0001", "speaker": None}],
        "thoughts": [
            {"candidate_id": _THOUGHT_CID, "is_thought": True, "thinker": "Mira", "confidence": 0.9}
        ],
    }
    off = _run(labels, emit_thoughts=False)
    assert all(s.type is SegmentType.NARRATION for s in off.segments)
    # thought-off sees the block as a single narration span (no prose sub-split at all).
    assert [s.text for s in off.segments] == [_THOUGHT_TEXT]


def test_book_with_no_italics_yields_zero_thoughts():
    block = Block(id="ch001_b0001", type=BlockType.PARAGRAPH, text="Plain narration, no italics.")
    provider = OllamaProvider(
        model="m",
        prompts_dir=PROMPTS_DIR,
        prompt_version="v4",
        emit_thoughts=True,
        post=_fake_post({"blocks": [{"block_id": "ch001_b0001", "speaker": None}], "thoughts": []}),
    )
    result = provider.attribute_chunk(
        chunk_blocks([block], overlap_blocks=0)[0], CharacterRegistry()
    )
    assert all(s.type is SegmentType.NARRATION for s in result.segments)


# --------------------------------------------------------------------------------------
# Opt-in wiring: v3<->v4 cache-key distinctness + run_attribution defaulting.
# --------------------------------------------------------------------------------------


def test_v3_and_v4_produce_distinct_cache_keys():
    common = dict(book_id="b", chapter_index=1, chunk_hash="h", provider_id="local", model_id="m")
    assert ChunkCacheKey(**common, prompt_version="v3") != ChunkCacheKey(
        **common, prompt_version="v4"
    )


def _narration_script(chunk, registry, attempt):
    return ChunkAttribution(
        segments=[
            Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text)
            for b in chunk.owned_blocks
        ]
    )


@pytest.mark.parametrize("emit,expected", [(True, "v4"), (False, "v3")])
def test_run_attribution_selects_prompt_version_from_emit_thoughts(
    tmp_path, monkeypatch, emit, expected
):
    import seiyuu.services.attribution as svc
    from factories import make_book
    from fake_provider import FakeProvider

    captured: dict = {}

    def fake_build(cfg, provider_id, model, prompt_version, *, emit_thoughts=False):
        captured["prompt_version"] = prompt_version
        captured["emit_thoughts"] = emit_thoughts
        return FakeProvider(_narration_script)

    monkeypatch.setattr(svc, "build_provider", fake_build)
    cfg = get_settings().model_copy(
        update={"emit_thoughts": emit, "attribution_prompt_version": "v3"}
    )
    svc.run_attribution(make_book(), tmp_path, cfg=cfg, provider=None, gpu=GpuResourceManager())
    assert captured == {"prompt_version": expected, "emit_thoughts": emit}
