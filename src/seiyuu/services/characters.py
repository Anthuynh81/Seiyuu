"""Characters overview: the read-side aggregate behind `seiyuu characters` and the M6b
Character Review screen. Pure derivation from the EFFECTIVE report (manual edits
applied), returned as pydantic models the API can serialize verbatim."""

from pathlib import Path

from pydantic import BaseModel

from seiyuu.attribute.models import FlaggedBlock, SegmentType
from seiyuu.attribute.spans import is_unattributed_quote
from seiyuu.services.attribution import load_report


class CharacterSummary(BaseModel):
    id: str
    name: str
    aliases: list[str] = []
    gender: str | None = None
    age_hint: str | None = None
    line_count: int = 0
    sample_lines: list[str] = []
    # Block id of the character's first attributed line ("ch013_b0042") — the M6c
    # spoiler-safe cast masks characters whose debut is past the reading frontier.
    first_appearance: str | None = None


class CharactersOverview(BaseModel):
    book_id: str
    provider_id: str
    model_id: str
    prompt_version: str
    narration_segments: int
    low_confidence_segments: int
    # Quoted spans no provider attempt attributed (speaker None but the text is a quoted
    # run): they render in the narrator's voice, so they are their own stat, never buried
    # in the narration count. The low-confidence ones also count into the review tally.
    unattributed_quote_segments: int
    confidence_threshold: float
    characters: list[CharacterSummary]  # sorted by line count, busiest first
    flagged: list[FlaggedBlock]
    notes: list[str]
    edit_warnings: list[str]  # overlay ops that no longer applied


def characters_overview(
    book_dir: Path, *, confidence_threshold: float, sample_lines: int = 2
) -> CharactersOverview:
    report, edit_warnings = load_report(book_dir)

    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    narration = low_confidence = unattributed_quotes = 0
    for chapter in report.chapters:
        for seg in chapter.segments:
            if is_unattributed_quote(seg.speaker, seg.text):
                unattributed_quotes += 1
                if seg.confidence < confidence_threshold:
                    low_confidence += 1
                continue
            if seg.speaker is None:
                narration += 1
                continue
            counts[seg.speaker] = counts.get(seg.speaker, 0) + 1
            if seg.confidence < confidence_threshold:
                low_confidence += 1
            if (
                seg.type is SegmentType.DIALOGUE
                and len(samples.setdefault(seg.speaker, [])) < sample_lines
            ):
                samples[seg.speaker].append(seg.text)

    characters = [
        CharacterSummary(
            id=char.id,
            name=char.canonical_name,
            aliases=char.aliases,
            gender=char.gender,
            age_hint=char.age_hint,
            line_count=counts.get(char.id, 0),
            sample_lines=samples.get(char.id, []),
            first_appearance=char.first_appearance,
        )
        for char in report.registry.characters
    ]
    characters.sort(key=lambda c: c.line_count, reverse=True)

    return CharactersOverview(
        book_id=report.book_id,
        provider_id=report.provider_id,
        model_id=report.model_id,
        prompt_version=report.prompt_version,
        narration_segments=narration,
        low_confidence_segments=low_confidence,
        unattributed_quote_segments=unattributed_quotes,
        confidence_threshold=confidence_threshold,
        characters=characters,
        flagged=report.flagged,
        notes=report.registry_notes,
        edit_warnings=edit_warnings,
    )
