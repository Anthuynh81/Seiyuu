"""resolve_voice: narrator catch-all, dialogue mapping, thought policy, unknown-speaker."""

from seiyuu.attribute.models import Segment, SegmentType
from seiyuu.voices import VoiceAssignment, resolve_voice


def _seg(type_, speaker=None):
    text = "x" if type_ is SegmentType.NARRATION else '"x"'
    return Segment(block_id="ch001_b0001", type=type_, speaker=speaker, text=text)


def _assignment(**over) -> VoiceAssignment:
    base = dict(book_id="b", narrator_voice_id="narrator_v", assignments={"elizabeth": "eliza_v"})
    base.update(over)
    return VoiceAssignment(**base)


def test_narration_uses_narrator():
    assert resolve_voice(_seg(SegmentType.NARRATION), _assignment()) == "narrator_v"


def test_dialogue_uses_assigned_voice():
    seg = _seg(SegmentType.DIALOGUE, "elizabeth")
    assert resolve_voice(seg, _assignment()) == "eliza_v"


def test_unmapped_speaker_falls_back_to_narrator():
    seg = _seg(SegmentType.DIALOGUE, "darcy")  # not in assignments
    assert resolve_voice(seg, _assignment()) == "narrator_v"


def test_thought_uses_speaker_voice_when_no_thought_voice():
    seg = _seg(SegmentType.THOUGHT, "elizabeth")
    assert resolve_voice(seg, _assignment()) == "eliza_v"


def test_thought_voice_override():
    seg = _seg(SegmentType.THOUGHT, "elizabeth")
    assert resolve_voice(seg, _assignment(thought_voice_id="inner_v")) == "inner_v"


def test_assignment_round_trips_json():
    a = _assignment(thought_voice_id="inner_v")
    reloaded = VoiceAssignment.model_validate_json(a.model_dump_json())
    assert reloaded.narrator_voice_id == "narrator_v"
    assert reloaded.assignments["elizabeth"] == "eliza_v"
    assert reloaded.stage.value == "draft"
