"""Manual attribution edits (M6a): a durable overlay replayed over every attribution run.

The user's Character Review fixes (rename a character, merge two, reassign a segment's
speaker) must survive re-attribution — a better model or more chapters regenerates
attribution.json freely, and the LLM cache key stays untouched. Edits are an ordered op
log in ``books/{id}/edits.json``, applied deterministically after every run (mirroring
how the alias post-pass layers deterministic fixes over cached LLM output). Ops that no
longer apply (a character or segment the new attribution doesn't have) are SKIPPED with
a warning, never a crash — the overlay must not be able to brick attribution.

Segment invariant (frozen schema): NARRATION carries no speaker; DIALOGUE/THOUGHT must
name one. Reassign ops preserve it: clearing the speaker makes a segment narration,
setting one on narration makes it dialogue (THOUGHT keeps its type on a speaker change).
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from seiyuu.attribute.models import AttributionReport, SegmentType
from seiyuu.repository import atomic_write_text
from seiyuu.services.common import ServiceError

EDITS_NAME = "edits.json"

_ANCHOR_LEN = 60


def _norm_text(text: str) -> str:
    return " ".join(text.split())


class RenameCharacter(BaseModel):
    op: Literal["rename"] = "rename"
    character_id: str
    new_name: str = Field(min_length=1)
    # Content anchor: ids are LLM-derived discovery-order slugs, so after re-attribution
    # the SAME id can denote a DIFFERENT character. record_edit fills the name the user
    # actually saw; apply skips with a warning when the live character no longer matches.
    expected_name: str | None = None


class MergeCharacters(BaseModel):
    op: Literal["merge"] = "merge"
    loser_id: str  # absorbed: its segments move to winner, its names become aliases
    winner_id: str
    expected_loser_name: str | None = None  # content anchors, as for rename
    expected_winner_name: str | None = None

    @model_validator(mode="after")
    def _distinct(self) -> "MergeCharacters":
        if self.loser_id == self.winner_id:
            raise ValueError("merge needs two different characters")
        return self


class ReassignSegment(BaseModel):
    op: Literal["reassign"] = "reassign"
    block_id: str
    # index within that block's segments (a block can split into several); negative
    # indices would resolve Python-style and could NEVER go stale — refuse at the model
    segment_index: int = Field(ge=0)
    speaker: str | None  # character id, or None -> narration
    # Content anchor: segment splits are not stable across attribution runs (a flagged
    # block collapses to one fallback segment; a successful run re-splits it), so a bare
    # index that stays in range would silently retarget different text. record_edit
    # captures a normalized prefix of the text the user actually reassigned.
    text_anchor: str | None = None


EditOp = Annotated[RenameCharacter | MergeCharacters | ReassignSegment, Field(discriminator="op")]


class EditLog(BaseModel):
    version: int = 1
    ops: list[EditOp] = []


def edits_path(book_dir: Path) -> Path:
    return Path(book_dir) / EDITS_NAME


def load_edits(book_dir: Path) -> EditLog:
    path = edits_path(book_dir)
    if not path.is_file():
        return EditLog()
    try:
        return EditLog.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError) as exc:
        # loud but mapped: a corrupt overlay must not surface as a raw traceback/500
        raise ServiceError(
            f"corrupt edits file {path}: {exc}; fix or delete it (edits are replayable)"
        ) from exc


def save_edits(book_dir: Path, log: EditLog) -> Path:
    return atomic_write_text(edits_path(book_dir), log.model_dump_json(indent=2))


def append_edit(book_dir: Path, op: EditOp) -> EditLog:
    log = load_edits(book_dir)
    log.ops.append(op)
    save_edits(book_dir, log)
    return log


def pop_edit(book_dir: Path) -> EditOp | None:
    """Undo: remove and return the most recent op (None if the log is empty)."""
    log = load_edits(book_dir)
    if not log.ops:
        return None
    op = log.ops.pop()
    save_edits(book_dir, log)
    return op


def _anchor_mismatch(char, expected: str | None) -> bool:
    """True when the character the id now denotes no longer matches the recorded name."""
    return expected is not None and not char.matches_name(expected)


def _apply_rename(report: AttributionReport, op: RenameCharacter) -> str | None:
    char = report.registry.get(op.character_id)
    if char is None:
        return f"rename skipped: no character {op.character_id!r} in this attribution"
    if _anchor_mismatch(char, op.expected_name):
        return (
            f"rename skipped: {op.character_id!r} is now {char.canonical_name!r}, "
            f"not {op.expected_name!r} — re-record the edit if still wanted"
        )
    if char.canonical_name != op.new_name and char.canonical_name not in char.aliases:
        char.aliases.append(char.canonical_name)  # keep the old name findable as an alias
    char.canonical_name = op.new_name
    return None


def _apply_merge(report: AttributionReport, op: MergeCharacters) -> str | None:
    loser = report.registry.get(op.loser_id)
    winner = report.registry.get(op.winner_id)
    if loser is None or winner is None or op.loser_id == op.winner_id:
        missing = op.loser_id if loser is None else op.winner_id
        return f"merge skipped: character {missing!r} not distinct/present in this attribution"
    if _anchor_mismatch(loser, op.expected_loser_name) or _anchor_mismatch(
        winner, op.expected_winner_name
    ):
        return (
            f"merge skipped: {op.loser_id!r}/{op.winner_id!r} no longer denote "
            f"{op.expected_loser_name!r}/{op.expected_winner_name!r} — re-record the edit"
        )
    for name in [loser.canonical_name, *loser.aliases]:
        if name != winner.canonical_name and name not in winner.aliases:
            winner.aliases.append(name)
    report.registry.characters = [c for c in report.registry.characters if c.id != op.loser_id]
    for chapter in report.chapters:
        chapter.segments = [
            seg.model_copy(update={"speaker": op.winner_id}) if seg.speaker == op.loser_id else seg
            for seg in chapter.segments
        ]
    return None


def _apply_reassign(report: AttributionReport, op: ReassignSegment) -> str | None:
    if op.speaker is not None and report.registry.get(op.speaker) is None:
        return f"reassign skipped: no character {op.speaker!r} in this attribution"
    for chapter in report.chapters:
        in_block = [i for i, s in enumerate(chapter.segments) if s.block_id == op.block_id]
        if not in_block:
            continue
        if op.segment_index >= len(in_block):
            return (
                f"reassign skipped: block {op.block_id!r} has {len(in_block)} segment(s), "
                f"index {op.segment_index} no longer exists"
            )
        i = in_block[op.segment_index]
        seg = chapter.segments[i]
        if op.text_anchor is not None and not _norm_text(seg.text).startswith(op.text_anchor):
            return (
                f"reassign skipped: {op.block_id!r}[{op.segment_index}] is now different "
                f"text than when the edit was recorded — re-record it if still wanted"
            )
        # a manual reassign is ground truth: confidence 1.0, so the review queue drains
        if op.speaker is None:
            update = {"speaker": None, "type": SegmentType.NARRATION, "confidence": 1.0}
        elif seg.type is SegmentType.NARRATION:
            update = {"speaker": op.speaker, "type": SegmentType.DIALOGUE, "confidence": 1.0}
        else:
            update = {"speaker": op.speaker, "confidence": 1.0}  # THOUGHT/DIALOGUE keep type
        chapter.segments[i] = seg.model_copy(update=update)
        return None
    return f"reassign skipped: block {op.block_id!r} not in this attribution"


def anchor_op(report: AttributionReport, op: EditOp) -> EditOp:
    """Validate ``op`` against the CURRENT effective report and fill its content anchors.

    This is what makes durable ops safe to replay: the anchors record what the user was
    actually looking at, so a later attribution run that hands the same id/index to
    different content skips the op instead of silently retargeting. Raises
    ``ServiceError`` when the op doesn't apply cleanly right now."""
    if isinstance(op, RenameCharacter):
        char = report.registry.get(op.character_id)
        if char is None:
            raise ServiceError(f"unknown character {op.character_id!r}")
        return op.model_copy(update={"expected_name": char.canonical_name})
    if isinstance(op, MergeCharacters):
        loser = report.registry.get(op.loser_id)
        winner = report.registry.get(op.winner_id)
        if loser is None:
            raise ServiceError(f"unknown character {op.loser_id!r}")
        if winner is None:
            raise ServiceError(f"unknown character {op.winner_id!r}")
        return op.model_copy(
            update={
                "expected_loser_name": loser.canonical_name,
                "expected_winner_name": winner.canonical_name,
            }
        )
    if op.speaker is not None and report.registry.get(op.speaker) is None:
        raise ServiceError(f"unknown character {op.speaker!r}")
    in_block = [
        seg
        for chapter in report.chapters
        for seg in chapter.segments
        if seg.block_id == op.block_id
    ]
    if not in_block:
        raise ServiceError(f"no block {op.block_id!r} in this attribution")
    if op.segment_index >= len(in_block):
        raise ServiceError(
            f"block {op.block_id!r} has {len(in_block)} segment(s); "
            f"index {op.segment_index} is out of range"
        )
    anchor = _norm_text(in_block[op.segment_index].text)[:_ANCHOR_LEN]
    return op.model_copy(update={"text_anchor": anchor})


def apply_edits(report: AttributionReport, log: EditLog) -> tuple[AttributionReport, list[str]]:
    """Replay the op log over a freshly-loaded report; returns (effective report, warnings).

    The input report is not mutated. Ops apply in order (a rename can precede a merge of
    the renamed character); an op that no longer fits the current attribution produces a
    warning and is skipped.
    """
    if not log.ops:
        return report, []
    effective = report.model_copy(deep=True)
    warnings: list[str] = []
    for op in log.ops:
        if isinstance(op, RenameCharacter):
            problem = _apply_rename(effective, op)
        elif isinstance(op, MergeCharacters):
            problem = _apply_merge(effective, op)
        else:
            problem = _apply_reassign(effective, op)
        if problem:
            warnings.append(problem)
    return effective, warnings
