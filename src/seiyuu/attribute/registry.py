"""Running character registry: resolve LLM speaker NAMES to stable character ids and
merge per-chunk :class:`CharacterMention`s into the registry.

Conservative by design (SPEC: registry quality must degrade gracefully — small models are
weak at long-range alias resolution). A mention whose aliases would fuse two *already
distinct* registry characters is NOT auto-merged; the skipped merge is recorded as a note
for human review. Adding a brand-new alias or character is safe and applied directly.
"""

import re

from seiyuu.attribute.models import Character, CharacterMention, CharacterRegistry, Segment

_SLUG_NONWORD = re.compile(r"[^a-z0-9]+")


def make_character_id(name: str, existing_ids: set[str]) -> str:
    """A stable, readable slug from a display name, unique within the registry."""
    base = _SLUG_NONWORD.sub("_", name.casefold()).strip("_") or "character"
    candidate = base
    n = 2
    while candidate in existing_ids:
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def _matched_characters(registry: CharacterRegistry, mention: CharacterMention) -> list[Character]:
    """Distinct existing characters this mention touches via its name or any alias."""
    matched: list[Character] = []
    for token in (mention.name, *mention.aliases):
        char = registry.find_by_name(token)
        if char is not None and char not in matched:
            matched.append(char)
    return matched


def _enrich(primary: Character, mention: CharacterMention, registry: CharacterRegistry) -> None:
    """Fill missing metadata and add genuinely new aliases (never ones owned elsewhere)."""
    if primary.gender is None:
        primary.gender = mention.gender
    if primary.age_hint is None:
        primary.age_hint = mention.age_hint
    if primary.description is None:
        primary.description = mention.description
    for alias in (mention.name, *mention.aliases):
        # Only add an alias that is not already this character's name/alias and is not
        # claimed by a different existing character (that would be an implicit merge).
        if not primary.matches_name(alias) and registry.find_by_name(alias) is None:
            primary.aliases.append(alias)


def _integrate_mention(
    registry: CharacterRegistry,
    mention: CharacterMention,
    first_block_id: str | None,
    notes: list[str],
) -> None:
    matches = _matched_characters(registry, mention)
    if not matches:
        existing_ids = {c.id for c in registry.characters}
        registry.characters.append(
            Character(
                id=make_character_id(mention.name, existing_ids),
                canonical_name=mention.name,
                aliases=list(dict.fromkeys(a for a in mention.aliases if a != mention.name)),
                gender=mention.gender,
                age_hint=mention.age_hint,
                description=mention.description,
                first_appearance=first_block_id,
            )
        )
        return

    primary = next((c for c in matches if c.matches_name(mention.name)), matches[0])
    others = [c for c in matches if c is not primary]
    if others:
        notes.append(
            f"not merging {mention.name!r} with existing "
            f"{[c.canonical_name for c in others]} (flagged for review, not auto-applied)"
        )
    _enrich(primary, mention, registry)


def resolve_chunk(
    registry: CharacterRegistry, segments: list[Segment], mentions: list[CharacterMention]
) -> tuple[list[Segment], list[str]]:
    """Merge mentions, then return segments with ``speaker`` rewritten to character ids.

    Mutates ``registry`` in place (it is the book-wide running registry). A speaker that
    appears in a segment but was never declared as a mention still gets a minimal record.
    """
    first_seen: dict[str, str] = {}
    for seg in segments:
        if seg.speaker is not None and seg.speaker not in first_seen:
            first_seen[seg.speaker] = seg.block_id

    notes: list[str] = []
    for mention in mentions:
        _integrate_mention(registry, mention, first_seen.get(mention.name), notes)

    resolved: list[Segment] = []
    for seg in segments:
        if seg.speaker is None:
            resolved.append(seg)
            continue
        char = registry.find_by_name(seg.speaker)
        if char is None:
            existing_ids = {c.id for c in registry.characters}
            char = Character(
                id=make_character_id(seg.speaker, existing_ids),
                canonical_name=seg.speaker,
                first_appearance=first_seen.get(seg.speaker),
            )
            registry.characters.append(char)
        resolved.append(seg.model_copy(update={"speaker": char.id}))
    return resolved, notes
