"""Attribution model invariants: speaker normalization and segment validation."""

import pytest
from pydantic import ValidationError

from seiyuu.attribute.models import (
    Character,
    CharacterRegistry,
    Segment,
    SegmentType,
)


def test_narration_speaker_forced_none():
    seg = Segment(
        block_id="ch001_b0001", type=SegmentType.NARRATION, text="He left.", speaker="Anyone"
    )
    assert seg.speaker is None


def test_dialogue_requires_speaker():
    with pytest.raises(ValidationError):
        Segment(block_id="ch001_b0001", type=SegmentType.DIALOGUE, text='"Hi"', speaker="  ")


def test_thought_requires_speaker():
    with pytest.raises(ValidationError):
        Segment(block_id="ch001_b0001", type=SegmentType.THOUGHT, text="Run.", speaker=None)


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        Segment(block_id="ch001_b0001", type=SegmentType.NARRATION, text="   ")


def test_bad_block_id_rejected():
    with pytest.raises(ValidationError):
        Segment(block_id="b1", type=SegmentType.NARRATION, text="x")


def test_confidence_range_enforced():
    with pytest.raises(ValidationError):
        Segment(block_id="ch001_b0001", type=SegmentType.NARRATION, text="x", confidence=1.5)


def test_registry_lookup_by_name_is_case_and_alias_insensitive():
    reg = CharacterRegistry(
        characters=[
            Character(id="elizabeth", canonical_name="Elizabeth", aliases=["Lizzy", "Miss Bennet"])
        ]
    )
    assert reg.find_by_name("lizzy").id == "elizabeth"
    assert reg.find_by_name("ELIZABETH").id == "elizabeth"
    assert reg.find_by_name("Darcy") is None
    assert reg.get("elizabeth").canonical_name == "Elizabeth"
