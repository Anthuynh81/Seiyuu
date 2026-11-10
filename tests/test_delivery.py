"""Delivery direction by segment type: the pure rule, and its render/estimate parity.

No worker and no model: a fake engine stands in, and a library voice whose engine is ``breeze``
exercises the Breeze branch (keys and the delivery rule go by ``meta.engine``).
"""

import json
from pathlib import Path

import pytest

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    Segment,
    SegmentType,
)
from seiyuu.engines import list_engine_ids
from seiyuu.engines.breeze_engine import INSTRUCTION_KEY as ENGINE_INSTRUCTION_KEY
from seiyuu.gpu import GpuResourceManager
from seiyuu.normalize import normalize_text, profile_for
from seiyuu.render import estimate_render_cost, render_book_multivoice
from seiyuu.render.cache import SegmentKey
from seiyuu.voices import VoiceAssignment, VoiceLibrary, VoiceMeta
from seiyuu.voices.delivery import (
    CAST_PLAN_NAME,
    DEFAULT_DIALOGUE,
    DEFAULT_INNER_VOICE,
    DIRECTED_ENGINES,
    INSTRUCTION_KEY,
    MAX_INSTRUCTION_CHARS,
    Delivery,
    DeliveryError,
    DeliveryUpdate,
    apply_delivery_update,
    delivery_settings,
    instructions_for,
    load_delivery,
    narrator_shares_a_voice,
)
from seiyuu.voices.models import VoiceKind

BOOK_ID = "test-book-00000000"
NARRATION, DIALOGUE, THOUGHT = SegmentType.NARRATION, SegmentType.DIALOGUE, SegmentType.THOUGHT


def _assignment(narrator: str = "kim_v", **characters: str) -> VoiceAssignment:
    cast = {"kim": "kim_v", "alice": "alice_v", "bob": "bob_v", **characters}
    return VoiceAssignment(book_id=BOOK_ID, narrator_voice_id=narrator, assignments=cast)


# -- the pure rule -------------------------------------------------------------------------------


def test_the_engine_key_and_engine_list_agree_with_the_adapter():
    assert INSTRUCTION_KEY == ENGINE_INSTRUCTION_KEY
    assert set(list_engine_ids()) >= DIRECTED_ENGINES


def test_texts_are_stored_in_one_normal_form():
    d = Delivery(inner_voice="  calm,\n  dry  ", narration="\t", dialogue="a  b")
    assert (d.inner_voice, d.narration, d.dialogue) == ("calm, dry", "", "a b")
    with pytest.raises(ValueError, match="at most"):
        Delivery(dialogue="x" * (MAX_INSTRUCTION_CHARS + 1))


def test_defaults_are_direction_on_with_plain_third_person_narration():
    d = Delivery()
    assert d.enabled and d.inner_voice == DEFAULT_INNER_VOICE and d.dialogue == DEFAULT_DIALOGUE
    assert d.narration == ""


def test_an_update_changes_only_what_it_names_and_can_clear_a_text():
    base = Delivery()
    off = apply_delivery_update(base, DeliveryUpdate(enabled=False))
    assert not off.enabled and off.inner_voice == base.inner_voice
    cleared = apply_delivery_update(base, DeliveryUpdate(dialogue="", narration=" steady  read "))
    assert cleared.dialogue == "" and cleared.narration == "steady read" and cleared.enabled
    assert DeliveryUpdate().is_empty() and not DeliveryUpdate(enabled=True).is_empty()
    assert DeliveryUpdate(enabled=None, dialogue=None).is_empty()  # null = leave it
    assert not DeliveryUpdate(dialogue="").is_empty()  # "" clears the text: a change
    with pytest.raises(ValueError):
        DeliveryUpdate(volume=3)


def test_first_person_means_the_narrator_voice_is_also_a_characters():
    assert narrator_shares_a_voice(_assignment("kim_v"))
    assert not narrator_shares_a_voice(_assignment("narrator_v"))


def test_instructions_by_segment_type():
    d = Delivery(inner_voice="inner", narration="", dialogue="aloud")
    first = instructions_for(d, _assignment("kim_v"))
    assert first == {NARRATION: "inner", THOUGHT: "inner", DIALOGUE: "aloud"}
    third = instructions_for(d, _assignment("narrator_v"))
    assert third == {THOUGHT: "inner", DIALOGUE: "aloud"}  # narration: a plain clone
    steady = instructions_for(d.model_copy(update={"narration": "steady"}), _assignment("n_v"))
    assert steady[NARRATION] == "steady"
    assert instructions_for(d.model_copy(update={"enabled": False}), _assignment("kim_v")) == {}


def test_settings_come_back_untouched_unless_a_directed_engine_has_an_instruction():
    table = {NARRATION: "inner", DIALOGUE: "aloud"}
    plain = {"ref_text_digest": 1.0}
    for engine_id in ("kokoro", "chatterbox", "qwen3tts", "indextts2", "elevenlabs", "fake"):
        assert delivery_settings(plain, engine_id, DIALOGUE, table) is plain
    assert delivery_settings(plain, "breeze", THOUGHT, table) is plain  # no text for the type
    assert delivery_settings(plain, "breeze", NARRATION, table, heading=True) is plain
    assert delivery_settings(plain, "breeze", DIALOGUE, {}) is plain  # direction off
    directed = delivery_settings(plain, "breeze", DIALOGUE, table)
    assert directed == {"ref_text_digest": 1.0, INSTRUCTION_KEY: "aloud"}
    assert plain == {"ref_text_digest": 1.0}  # never mutated


# -- the stored block ----------------------------------------------------------------------------


def _write_plan(out: Path, **delivery) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / CAST_PLAN_NAME
    body = {"book_id": BOOK_ID, "rows": [], "delivery": delivery}
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_no_plan_and_a_plan_from_before_this_block_mean_the_defaults(tmp_path):
    assert load_delivery(tmp_path) == Delivery()
    (tmp_path / CAST_PLAN_NAME).write_text(json.dumps({"book_id": BOOK_ID}), encoding="utf-8")
    assert load_delivery(tmp_path) == Delivery()


def test_the_block_is_read_from_the_cast_plan(tmp_path):
    _write_plan(tmp_path, enabled=False, dialogue="  loud  and clear ")
    d = load_delivery(tmp_path)
    assert not d.enabled and d.dialogue == "loud and clear"
    assert d.inner_voice == DEFAULT_INNER_VOICE  # an omitted text keeps its default


@pytest.mark.parametrize(
    "body",
    [
        "{not json",
        '{"delivery": {"enabled": "maybe"}}',
        "[]",
        '{"delivery": null}',
        '{"delivery": {"enable": false}}',  # a misspelled key must not mean "the defaults"
    ],
)
def test_an_unreadable_plan_fails_loudly_and_names_the_file(tmp_path, body):
    (tmp_path / CAST_PLAN_NAME).write_text(body, encoding="utf-8")
    with pytest.raises(DeliveryError, match=CAST_PLAN_NAME):
        load_delivery(tmp_path)


# -- render / estimate ---------------------------------------------------------------------------


class _RecordingEngine(FakeEngine):
    """Remembers the settings each line was spoken with."""

    def __init__(self) -> None:
        super().__init__()
        self.settings_by_text: dict[str, dict] = {}

    def _synthesize_native(self, text, voice, settings):
        self.settings_by_text[text] = dict(settings)
        return super()._synthesize_native(text, voice, settings)


def _report() -> AttributionReport:
    # factories.make_book layout: ch1 = heading b0001, paragraph b0002, scene break, paragraph
    # b0004; ch2 = heading b0001, paragraph b0002.
    def seg(block, kind, text, speaker=None):
        return Segment(block_id=block, type=kind, text=text, speaker=speaker)

    one = [
        seg("ch001_b0001", NARRATION, "Chapter One"),
        seg("ch001_b0002", NARRATION, "I counted the doors."),
        seg("ch001_b0002", DIALOGUE, "Hello there.", "kim"),
        seg("ch001_b0004", THOUGHT, "One more than before.", "kim"),
        seg("ch001_b0004", DIALOGUE, "Who is it?", "alice"),
        seg("ch001_b0004", DIALOGUE, "Nobody at all.", "bob"),
    ]
    two = [
        seg("ch002_b0001", NARRATION, "Chapter Two"),
        seg("ch002_b0002", NARRATION, "The hall was empty."),
    ]
    people = [Character(id=i, canonical_name=i.title()) for i in ("kim", "alice", "bob")]
    return AttributionReport(
        book_id=BOOK_ID,
        provider_id="local",
        model_id="m",
        prompt_version="v5",
        registry=CharacterRegistry(characters=people),
        chapters=[
            AttributedChapter(index=1, title="Chapter One", segments=one),
            AttributedChapter(index=2, title="Chapter Two", segments=two),
        ],
    )


def _library(tmp_path) -> VoiceLibrary:
    lib = VoiceLibrary(tmp_path / "voices")
    # PRESET stands in for a cloned voice (no consent or reference files needed); the delivery
    # rule and the key go by the voice's engine. Bob speaks on an engine that takes no direction.
    for voice_id, engine in (
        ("kim_v", "breeze"),
        ("alice_v", "breeze"),
        ("narrator_v", "breeze"),
        ("bob_v", "kokoro"),
    ):
        lib.save(
            VoiceMeta(
                voice_id=voice_id,
                name=voice_id,
                kind=VoiceKind.PRESET,
                engine=engine,
                preset_id=f"p_{voice_id}",
            )
        )
    return lib


def _world(tmp_path, monkeypatch):
    engine = _RecordingEngine()
    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda engine_id, **kw: engine)
    return engine, _library(tmp_path), tmp_path / "out"


def _render(lib, assignment, out, **kw):
    return render_book_multivoice(
        _report(), make_book(), lib, assignment, out, gpu=GpuResourceManager(), **kw
    )


def _estimate(lib, assignment, out, **kw):
    return estimate_render_cost(_report(), make_book(), lib, assignment, out, **kw)


def _instruction(engine, text):
    return engine.settings_by_text[text].get(INSTRUCTION_KEY)


def test_first_person_book_narrates_in_the_inner_voice(tmp_path, monkeypatch):
    engine, lib, out = _world(tmp_path, monkeypatch)
    result = _render(lib, _assignment("kim_v"), out)
    assert result.synthesized == 8
    assert _instruction(engine, "I counted the doors.") == DEFAULT_INNER_VOICE
    assert _instruction(engine, "One more than before.") == DEFAULT_INNER_VOICE  # thought
    assert _instruction(engine, "Hello there.") == DEFAULT_DIALOGUE  # the same voice, aloud
    assert _instruction(engine, "Who is it?") == DEFAULT_DIALOGUE
    assert _instruction(engine, "Chapter One") is None  # a heading is announced, not performed
    assert _instruction(engine, "Nobody at all.") is None  # Kokoro takes no instruction

    # parity: the estimate builds the same keys from the same file -> nothing left to record
    est = _estimate(lib, _assignment("kim_v"), out)
    assert est.free_segments == 0 and est.cached_segments == 8


def test_third_person_book_keeps_narration_a_plain_clone(tmp_path, monkeypatch):
    engine, lib, out = _world(tmp_path, monkeypatch)
    _render(lib, _assignment("narrator_v"), out)
    assert _instruction(engine, "I counted the doors.") is None
    assert _instruction(engine, "The hall was empty.") is None
    assert _instruction(engine, "One more than before.") == DEFAULT_INNER_VOICE  # still a thought
    assert _instruction(engine, "Hello there.") == DEFAULT_DIALOGUE
    assert _estimate(lib, _assignment("narrator_v"), out).free_segments == 0


def _plain_key(lib, voice_id, source_text) -> SegmentKey:
    """The key of a plain clone: what this line was keyed as before delivery direction."""
    meta = lib.load(voice_id)
    return SegmentKey.build(
        engine=meta.engine,
        engine_model_version="fake-1.0",
        voice_id=voice_id,
        settings=meta.engine_settings(),
        seed=meta.seed,
        normalized_text=normalize_text(source_text, profile=profile_for(meta.engine)),
    )


def _stems(result) -> set[str]:
    return {Path(s.wav).stem for c in result.manifest.chapters for s in c.segments if s.wav}


def test_direction_off_gives_exactly_the_plain_clone_keys(tmp_path, monkeypatch):
    engine, lib, out = _world(tmp_path, monkeypatch)
    _write_plan(out, enabled=False)
    result = _render(lib, _assignment("kim_v"), out)
    assert all(INSTRUCTION_KEY not in s for s in engine.settings_by_text.values())
    expected = {
        _plain_key(lib, "kim_v", "Chapter One").key_hash,
        _plain_key(lib, "kim_v", "I counted the doors.").key_hash,
        _plain_key(lib, "kim_v", "Hello there.").key_hash,
        _plain_key(lib, "kim_v", "One more than before.").key_hash,
        _plain_key(lib, "alice_v", "Who is it?").key_hash,
        _plain_key(lib, "bob_v", "Nobody at all.").key_hash,
        _plain_key(lib, "kim_v", "Chapter Two").key_hash,
        _plain_key(lib, "kim_v", "The hall was empty.").key_hash,
    }
    assert _stems(result) == expected
    assert _estimate(lib, _assignment("kim_v"), out).free_segments == 0


def test_other_engines_and_headings_keep_their_keys_with_direction_on(tmp_path, monkeypatch):
    _engine, lib, out = _world(tmp_path, monkeypatch)
    stems = _stems(_render(lib, _assignment("kim_v"), out))
    assert _plain_key(lib, "bob_v", "Nobody at all.").key_hash in stems  # Kokoro: byte-identical
    assert _plain_key(lib, "kim_v", "Chapter One").key_hash in stems
    assert _plain_key(lib, "kim_v", "Hello there.").key_hash not in stems  # directed: re-keyed


def test_editing_a_text_re_records_only_the_lines_it_directs(tmp_path, monkeypatch):
    _engine, lib, out = _world(tmp_path, monkeypatch)
    assignment = _assignment("kim_v")
    _render(lib, assignment, out)
    _write_plan(out, dialogue="Quick and clipped.")
    assert _estimate(lib, assignment, out).free_segments == 2  # the two Breeze dialogue lines
    _write_plan(out, inner_voice="Barely above a whisper.")
    assert _estimate(lib, assignment, out).free_segments == 3  # 2 narration + 1 thought
    _write_plan(out, enabled=False)
    assert _estimate(lib, assignment, out).free_segments == 5  # every directed line
    _write_plan(out, narration="Clear and steady.")  # unused text in a first-person book
    assert _estimate(lib, assignment, out).free_segments == 0


def test_recasting_the_narrator_off_the_shared_voice_changes_only_narration(tmp_path, monkeypatch):
    _engine, lib, out = _world(tmp_path, monkeypatch)
    _render(lib, _assignment("kim_v"), out)
    # Same voices for every character; the narrator moves to a voice of its own.
    est = _estimate(lib, _assignment("narrator_v"), out)
    assert est.cached_segments == 4  # Kim's line and thought, Alice, Bob
    assert est.free_segments == 4  # two headings + two narration lines, now another voice


def test_an_unreadable_plan_stops_render_and_estimate(tmp_path, monkeypatch):
    engine, lib, out = _world(tmp_path, monkeypatch)
    out.mkdir(parents=True)
    (out / CAST_PLAN_NAME).write_text("{broken", encoding="utf-8")
    with pytest.raises(DeliveryError, match=CAST_PLAN_NAME):
        _estimate(lib, _assignment("kim_v"), out)
    with pytest.raises(DeliveryError, match=CAST_PLAN_NAME):
        _render(lib, _assignment("kim_v"), out)
    assert engine.calls == []  # nothing was recorded on a guess
