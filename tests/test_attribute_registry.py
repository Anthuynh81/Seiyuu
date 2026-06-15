"""Registry resolution: name→id, metadata enrichment, conservative (flagged) merges."""

from seiyuu.attribute.models import CharacterMention, CharacterRegistry, Segment, SegmentType
from seiyuu.attribute.registry import make_character_id, resolve_chunk


def _dialogue(block_id: str, speaker: str) -> Segment:
    return Segment(block_id=block_id, type=SegmentType.DIALOGUE, text='"hi"', speaker=speaker)


def test_make_character_id_slugifies_and_dedupes():
    assert make_character_id("Elizabeth Bennet", set()) == "elizabeth_bennet"
    assert make_character_id("Mr. Darcy", {"mr_darcy"}) == "mr_darcy_2"


def test_resolves_speaker_names_to_ids_and_creates_records():
    reg = CharacterRegistry()
    segs = [_dialogue("ch001_b0001", "Alice"), _dialogue("ch001_b0002", "Alice")]
    mentions = [CharacterMention(name="Alice", gender="female", description="A girl")]

    resolved, notes = resolve_chunk(reg, segs, mentions)

    assert notes == []
    assert all(s.speaker == "alice" for s in resolved)
    alice = reg.get("alice")
    assert alice.gender == "female" and alice.first_appearance == "ch001_b0001"


def test_unmentioned_speaker_gets_minimal_record():
    reg = CharacterRegistry()
    resolved, _ = resolve_chunk(reg, [_dialogue("ch001_b0001", "Bob")], [])
    assert resolved[0].speaker == "bob"
    assert reg.get("bob").canonical_name == "Bob"


def test_new_alias_is_applied():
    reg = CharacterRegistry()
    resolve_chunk(reg, [], [CharacterMention(name="Elizabeth")])
    resolve_chunk(reg, [], [CharacterMention(name="Elizabeth", aliases=["Lizzy"])])
    assert reg.find_by_name("Lizzy").id == "elizabeth"
    assert len(reg.characters) == 1


def test_merge_of_two_existing_characters_is_flagged_not_applied():
    reg = CharacterRegistry()
    # Two distinct characters established separately.
    resolve_chunk(reg, [], [CharacterMention(name="Elizabeth")])
    resolve_chunk(reg, [], [CharacterMention(name="Miss Bennet")])
    assert len(reg.characters) == 2

    # A mention claiming they are the same must not silently fuse them.
    _, notes = resolve_chunk(reg, [], [CharacterMention(name="Elizabeth", aliases=["Miss Bennet"])])
    assert len(reg.characters) == 2
    assert notes and "not merging" in notes[0]
