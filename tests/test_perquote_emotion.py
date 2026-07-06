"""Phase 1 (F1 per-quote speaker attribution + F2 per-segment emotion) — fixtures only.

No live LLM/TTS. F1 drives the provider's assembly through a fake Ollama client returning
canned per-quote JSON; F2 exercises the pure mapping and the render/estimate SegmentKey
parity (apply_emotion off = byte-identical cache; on = emotion folds into settings_hash).
"""

import json
import types

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.attribute.chunking import chunk_blocks
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    EmotionLabel,
    EmotionVerdict,
    Segment,
    SegmentType,
)
from seiyuu.attribute.providers.base import _render_owned_blocks_indexed, render_prompt
from seiyuu.attribute.providers.local import OllamaProvider
from seiyuu.attribute.spans import quoted_ordinals, thought_candidate_spans
from seiyuu.gpu import GpuResourceManager
from seiyuu.ingest.models import Block, BlockType
from seiyuu.render import estimate_render_cost, render_book_multivoice
from seiyuu.settings import get_settings
from seiyuu.voices import VoiceAssignment, VoiceLibrary, VoiceMeta, map_emotion
from seiyuu.voices.models import VoiceKind

PROMPTS_DIR = get_settings().prompts_dir


def _fake_client(content: str):
    def create(**kwargs):
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice])

    completions = types.SimpleNamespace(create=create)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def _provider(content: str, version: str = "v5", emit_thoughts: bool = False) -> OllamaProvider:
    return OllamaProvider(
        model="m",
        prompts_dir=PROMPTS_DIR,
        transport="openai",
        prompt_version=version,
        emit_thoughts=emit_thoughts,
        client=_fake_client(content),
    )


def _one_block_chunk(text: str, italic_spans=None):
    block = Block(
        id="ch001_b0001", type=BlockType.PARAGRAPH, text=text, italic_spans=italic_spans or []
    )
    return chunk_blocks([block], overlap_blocks=0)[0]


# ---------------------------------------------------------------------------------------
# F1 — per-quote speaker attribution
# ---------------------------------------------------------------------------------------

_TWO_SPEAKERS = '"Hi," said Ann. "Bye," said Bob.'


def test_two_speakers_trading_lines_get_own_speakers():
    labels = {
        "blocks": [
            {
                "block_id": "ch001_b0001",
                "quotes": [
                    {"index": 0, "speaker": "Ann"},
                    {"index": 1, "speaker": "Bob"},
                ],
            }
        ]
    }
    result = _provider(json.dumps(labels)).attribute_chunk(
        _one_block_chunk(_TWO_SPEAKERS), CharacterRegistry()
    )
    segs = result.segments
    assert [(s.type.value, s.speaker, s.text) for s in segs] == [
        ("dialogue", "Ann", '"Hi,"'),
        ("narration", None, " said Ann. "),
        ("dialogue", "Bob", '"Bye,"'),
        ("narration", None, " said Bob."),
    ]
    # reconstruction holds — every span text is a verbatim source slice.
    assert "".join(s.text for s in segs) == _TWO_SPEAKERS


def test_unattributed_quote_among_attributed_degrades_to_narration():
    labels = {
        "blocks": [
            {
                "block_id": "ch001_b0001",
                "quotes": [
                    {"index": 0, "speaker": "Ann"},
                    {"index": 1, "speaker": None},  # unattributed -> narration, never guessed
                ],
            }
        ]
    }
    result = _provider(json.dumps(labels)).attribute_chunk(
        _one_block_chunk(_TWO_SPEAKERS), CharacterRegistry()
    )
    types_ = [(s.type.value, s.speaker) for s in result.segments]
    assert types_ == [
        ("dialogue", "Ann"),
        ("narration", None),
        ("narration", None),  # the unlabeled quote
        ("narration", None),
    ]
    assert "".join(s.text for s in result.segments) == _TWO_SPEAKERS


_THREE = '"A," she said. "B," he said. "C," they said.'


def test_three_quotes_keyed_by_index_not_position():
    # Quotes returned OUT OF ORDER — assembly must key on `index`, not list position.
    labels = {
        "blocks": [
            {
                "block_id": "ch001_b0001",
                "quotes": [
                    {"index": 2, "speaker": "Cara"},
                    {"index": 0, "speaker": "Ann"},
                    {"index": 1, "speaker": "Bob"},
                ],
            }
        ]
    }
    result = _provider(json.dumps(labels)).attribute_chunk(
        _one_block_chunk(_THREE), CharacterRegistry()
    )
    dialogue = [(s.speaker, s.text) for s in result.segments if s.type is SegmentType.DIALOGUE]
    assert dialogue == [("Ann", '"A,"'), ("Bob", '"B,"'), ("Cara", '"C,"')]
    assert "".join(s.text for s in result.segments) == _THREE


def test_v3_shaped_output_assembles_via_whole_block_fallback():
    # A v3/v4-shaped row has no `quotes`; the whole-block speaker labels every quote.
    labels = {"blocks": [{"block_id": "ch001_b0001", "speaker": "Ann"}]}
    result = _provider(json.dumps(labels)).attribute_chunk(
        _one_block_chunk(_TWO_SPEAKERS), CharacterRegistry()
    )
    dialogue = [(s.speaker, s.text) for s in result.segments if s.type is SegmentType.DIALOGUE]
    assert dialogue == [("Ann", '"Hi,"'), ("Ann", '"Bye,"')]
    assert "".join(s.text for s in result.segments) == _TWO_SPEAKERS


def test_out_of_range_quote_index_is_dropped():
    # An index that isn't an actual quoted-span ordinal is discarded; with no valid label the
    # block is not in per-quote mode and (lacking a block speaker) both quotes are narration.
    labels = {"blocks": [{"block_id": "ch001_b0001", "quotes": [{"index": 9, "speaker": "Ghost"}]}]}
    result = _provider(json.dumps(labels)).attribute_chunk(
        _one_block_chunk(_TWO_SPEAKERS), CharacterRegistry()
    )
    assert all(s.type is SegmentType.NARRATION for s in result.segments)
    assert "".join(s.text for s in result.segments) == _TWO_SPEAKERS


def test_markers_never_leak_into_segment_text_or_mentions():
    labels = {
        "blocks": [
            {
                "block_id": "ch001_b0001",
                "quotes": [
                    {"index": 0, "speaker": "Ann"},
                    {"index": 1, "speaker": "Bob"},
                ],
            }
        ],
        "characters": [{"name": "Ann"}, {"name": "Bob"}],
    }
    result = _provider(json.dumps(labels)).attribute_chunk(
        _one_block_chunk(_TWO_SPEAKERS), CharacterRegistry()
    )
    assert all("⟦" not in s.text and "⟧" not in s.text for s in result.segments)
    assert all("⟦" not in c.name for c in result.characters)


def test_prompt_indexes_only_multi_quote_blocks_at_the_quoted_ordinal():
    chunk = _one_block_chunk(_TWO_SPEAKERS)
    spans = thought_candidate_spans("ch001_b0001", _TWO_SPEAKERS, [])
    rendered = _render_owned_blocks_indexed(chunk, {"ch001_b0001": spans})
    # ⟦Q0⟧ precedes the first quoted run, ⟦Q1⟧ the second — the shared quoted-ordinal space.
    assert '⟦Q0⟧"Hi,"' in rendered
    assert '⟦Q1⟧"Bye,"' in rendered
    # The prompt-side ordinal equals the assembly-side ordinal (one shared helper).
    ordinals = quoted_ordinals(spans)
    assert [text for _, text in ((o, s.text) for o, s in ordinals)] == ['"Hi,"', '"Bye,"']


def test_single_quote_block_stays_plain_in_the_prompt():
    text = 'He paused. "Hello," she said.'  # one quoted span -> no markers (hybrid)
    chunk = _one_block_chunk(text)
    spans = thought_candidate_spans("ch001_b0001", text, [])
    rendered = _render_owned_blocks_indexed(chunk, {"ch001_b0001": spans})
    assert "⟦Q" not in rendered
    assert rendered == f"[ch001_b0001]\n{text}"


def test_v3_prompt_render_is_unchanged_without_owned_render():
    # render_prompt with no owned_render keeps the plain whole-block render (v3/v4 path).
    from seiyuu.attribute.providers.base import _prompt_template

    prompt = render_prompt(
        _prompt_template(PROMPTS_DIR, "v3"), CharacterRegistry(), _one_block_chunk(_TWO_SPEAKERS)
    )
    assert "⟦Q" not in prompt
    assert _TWO_SPEAKERS in prompt


# ---------------------------------------------------------------------------------------
# F2 — per-segment emotion capture + composition with F1 and thoughts
# ---------------------------------------------------------------------------------------


def test_emotion_captured_per_quote_index_aligned_to_segments():
    labels = {
        "blocks": [
            {
                "block_id": "ch001_b0001",
                "quotes": [
                    {"index": 0, "speaker": "Ann", "emotion": {"label": "happy", "intensity": 2}},
                    {"index": 1, "speaker": "Bob", "emotion": {"label": "angry", "intensity": 3}},
                ],
            }
        ]
    }
    result = _provider(json.dumps(labels)).attribute_chunk(
        _one_block_chunk(_TWO_SPEAKERS), CharacterRegistry()
    )
    # segment_emotions is 1:1 with segments; narration carries None, dialogue its verdict.
    assert len(result.segment_emotions) == len(result.segments)
    pairs = [
        (s.type.value, None if e is None else (e.label.value, e.intensity))
        for s, e in zip(result.segments, result.segment_emotions, strict=True)
    ]
    assert pairs == [
        ("dialogue", ("happy", 2)),
        ("narration", None),
        ("dialogue", ("angry", 3)),
        ("narration", None),
    ]


def test_v3_shaped_output_has_no_emotions():
    labels = {"blocks": [{"block_id": "ch001_b0001", "speaker": "Ann"}]}
    result = _provider(json.dumps(labels)).attribute_chunk(
        _one_block_chunk(_TWO_SPEAKERS), CharacterRegistry()
    )
    assert all(e is None for e in result.segment_emotions)


_THOUGHT_BLOCK = 'I must leave now. "Fine," she said.'


def test_v6_composes_perquote_thoughts_and_emotion():
    italic = [(0, 17)]  # "I must leave now."  (a thought candidate: >=3 words, ends a sentence)
    chunk = _one_block_chunk(_THOUGHT_BLOCK, italic_spans=italic)
    labels = {
        "blocks": [
            {
                "block_id": "ch001_b0001",
                "speaker": "Ann",
                "emotion": {"label": "sad", "intensity": 1},
            }
        ],
        "thoughts": [
            {
                "candidate_id": "ch001_b0001:0",
                "is_thought": True,
                "thinker": "Bob",
                "confidence": 0.9,
            }
        ],
    }
    result = _provider(json.dumps(labels), version="v6", emit_thoughts=True).attribute_chunk(
        chunk, CharacterRegistry()
    )
    kinds = [(s.type.value, s.speaker) for s in result.segments]
    assert ("thought", "Bob") in kinds  # thought section still works under v6
    assert ("dialogue", "Ann") in kinds  # single-quote block -> whole-block speaker + emotion
    # reconstruction holds (the invariant is whitespace-insensitive at segment seams).
    from seiyuu.attribute.validate import reconstructs_block

    assert reconstructs_block(_THOUGHT_BLOCK, [s.text for s in result.segments])
    # emotion rides the dialogue segment; thought/narration are None.
    for seg, emo in zip(result.segments, result.segment_emotions, strict=True):
        assert (emo is not None) == (seg.type is SegmentType.DIALOGUE)


# ---------------------------------------------------------------------------------------
# F2 — pure emotion mapping
# ---------------------------------------------------------------------------------------


def test_map_emotion_neutral_and_none_yield_no_override():
    assert map_emotion("chatterbox", None) == {}
    assert map_emotion("chatterbox", EmotionVerdict(label=EmotionLabel.NEUTRAL, intensity=3)) == {}
    assert map_emotion("elevenlabs", EmotionVerdict(label=EmotionLabel.NEUTRAL, intensity=2)) == {}


def test_map_emotion_kokoro_never_overrides():
    for label in EmotionLabel:
        assert map_emotion("kokoro", EmotionVerdict(label=label, intensity=3)) == {}


def test_map_emotion_indextts2_stub_and_unknown_engines_no_override():
    assert map_emotion("indextts2", EmotionVerdict(label=EmotionLabel.ANGRY, intensity=3)) == {}
    assert map_emotion("something-new", EmotionVerdict(label=EmotionLabel.HAPPY, intensity=2)) == {}


def test_map_emotion_chatterbox_sets_capped_exaggeration_and_temperature():
    out = map_emotion("chatterbox", EmotionVerdict(label=EmotionLabel.ANGRY, intensity=3))
    assert set(out) == {"exaggeration", "temperature"}
    assert out["exaggeration"] <= 0.8  # capped so validation failures don't spike
    # intensity raises exaggeration monotonically up to the cap
    low = map_emotion("chatterbox", EmotionVerdict(label=EmotionLabel.HAPPY, intensity=1))
    high = map_emotion("chatterbox", EmotionVerdict(label=EmotionLabel.HAPPY, intensity=3))
    assert high["exaggeration"] > low["exaggeration"]


def test_map_emotion_elevenlabs_lowers_stability_raises_style_with_intensity():
    low = map_emotion("elevenlabs", EmotionVerdict(label=EmotionLabel.ANGRY, intensity=1))
    high = map_emotion("elevenlabs", EmotionVerdict(label=EmotionLabel.ANGRY, intensity=3))
    assert set(high) == {"stability", "style"}
    assert high["stability"] < low["stability"]
    assert high["style"] > low["style"]


def test_map_emotion_is_pure_and_deterministic():
    v = EmotionVerdict(label=EmotionLabel.TENSE, intensity=2)
    assert map_emotion("chatterbox", v) == map_emotion("chatterbox", v)


# ---------------------------------------------------------------------------------------
# F2 — render/estimate SegmentKey parity (apply_emotion off = cache-stable; on = folds in)
# ---------------------------------------------------------------------------------------

_BOOK_ID = "test-book-00000000"


def _emotive_report(with_emotion: bool) -> AttributionReport:
    # Mirrors factories.make_book chapter/block layout; alice speaks chatterbox dialogue.
    def chapter(idx, title, dia_block, dia_text):
        segs = [
            Segment(block_id=f"ch{idx:03d}_b0001", type=SegmentType.NARRATION, text=title),
            Segment(block_id=dia_block, type=SegmentType.DIALOGUE, text=dia_text, speaker="alice"),
        ]
        emotions = [
            None,
            EmotionVerdict(label=EmotionLabel.ANGRY, intensity=3) if with_emotion else None,
        ]
        extra = []
        extra_emo = []
        if idx == 1:  # ch1 has a post-scene-break narration paragraph (block ...b0004)
            extra = [Segment(block_id="ch001_b0004", type=SegmentType.NARRATION, text="After.")]
            extra_emo = [None]
        return AttributedChapter(
            index=idx, title=title, segments=segs + extra, segment_emotions=emotions + extra_emo
        )

    return AttributionReport(
        book_id=_BOOK_ID,
        provider_id="local",
        model_id="qwen2.5:7b",
        prompt_version="v5",
        registry=CharacterRegistry(characters=[Character(id="alice", canonical_name="Alice")]),
        chapters=[
            chapter(1, "Chapter 1", "ch001_b0002", "Hello world."),
            chapter(2, "Chapter 2", "ch002_b0002", "Second chapter."),
        ],
    )


def _library(tmp_path) -> VoiceLibrary:
    lib = VoiceLibrary(tmp_path / "voices")
    lib.save(
        VoiceMeta(
            voice_id="narrator_v",
            name="N",
            kind=VoiceKind.PRESET,
            engine="kokoro",
            preset_id="af_heart",
        )  # fmt: skip
    )
    lib.save(
        VoiceMeta(
            voice_id="alice_v",
            name="Alice",
            kind=VoiceKind.PRESET,
            engine="chatterbox",
            preset_id="clone_alice",
        )  # fmt: skip
    )
    return lib


def _assignment() -> VoiceAssignment:
    return VoiceAssignment(
        book_id=_BOOK_ID, narrator_voice_id="narrator_v", assignments={"alice": "alice_v"}
    )


def _patch(monkeypatch, engine):
    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda engine_id, **kw: engine)


def test_apply_emotion_off_is_cache_identical_to_no_emotion(tmp_path, monkeypatch):
    _patch(monkeypatch, FakeEngine())
    lib, assign, out = _library(tmp_path), _assignment(), tmp_path / "out"
    # Render an emotion-tagged report with apply_emotion OFF...
    render_book_multivoice(
        _emotive_report(with_emotion=True), make_book(), lib, assign, out,
        gpu=GpuResourceManager(), apply_emotion=False,
    )  # fmt: skip
    # ...then estimate a PLAIN report (no emotions) with apply_emotion OFF: fully cached, so the
    # SegmentKeys are byte-identical — off ignores segment_emotions entirely.
    est = estimate_render_cost(
        _emotive_report(with_emotion=False), make_book(), lib, assign, out, apply_emotion=False
    )
    assert est.cached_segments > 0 and est.free_segments == 0 and est.paid_segments == 0


def test_apply_emotion_on_folds_into_key_with_render_estimate_parity(tmp_path, monkeypatch):
    _patch(monkeypatch, FakeEngine())
    lib, assign, out = _library(tmp_path), _assignment(), tmp_path / "out"
    report = _emotive_report(with_emotion=True)
    # Render WITH emotion applied.
    r = render_book_multivoice(
        report, make_book(), lib, assign, out, gpu=GpuResourceManager(), apply_emotion=True
    )
    assert r.synthesized > 0
    # Parity: the estimate builds the IDENTICAL emotion-folded keys -> every segment cached.
    on = estimate_render_cost(report, make_book(), lib, assign, out, apply_emotion=True)
    assert on.free_segments == 0  # nothing uncached; render == estimate keys

    # Turning emotion OFF changes the two chatterbox dialogue segments' settings_hash, so they
    # miss the emotion-keyed cache (proof the emotion folded into the key when on). The kokoro
    # narration keys are unaffected (kokoro has no emotion knob).
    off = estimate_render_cost(report, make_book(), lib, assign, out, apply_emotion=False)
    assert off.free_segments == 2
