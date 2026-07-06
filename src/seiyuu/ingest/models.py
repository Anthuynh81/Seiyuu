"""Normalized book models — the documented contract between ingest and all later stages.

Schema (SPEC.md stage 1):
    {book_meta, chapters: [{title, blocks: [{type, id, text}]}]}

Block ids are stable and ordered: ``ch{NNN}_b{NNNN}`` (1-based, chapter-scoped,
zero-padded to at least 3/4 digits — wider when a book exceeds 999 chapters or
9999 blocks, e.g. web novels with 1000+ chapters).
"""

import re
from enum import StrEnum

from pydantic import BaseModel, model_validator

BLOCK_ID_PATTERN = re.compile(r"^ch\d{3,}_b\d{4,}$")


class BlockType(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    # Scene breaks exist ONLY for assembly pause logic. They never become
    # synthesis segments and always carry empty text.
    SCENE_BREAK = "scene_break"


class Block(BaseModel):
    id: str
    type: BlockType
    text: str = ""
    # Half-open [start, end) CODE-POINT offsets of italic runs into ``text``, sorted
    # ascending and non-overlapping. Runs originate from inline <em>/<i> (Phase 1) OR from
    # CSS class / inline style="font-style:italic|oblique" (Phase 2a) — the offset geometry
    # is identical either way, and this schema is provenance-agnostic. Additive metadata
    # (default empty): ingest captures the author's italic signal so a downstream
    # markup-aware pass can emit SegmentType.THOUGHT for interior monologue. Ingest makes NO
    # thought-vs-emphasis judgement here — it only records where the italics are. Old
    # normalized.json without the key validates and yields []; only re-ingest populates it.
    italic_spans: list[tuple[int, int]] = []

    @model_validator(mode="after")
    def _check_invariants(self) -> "Block":
        if not BLOCK_ID_PATTERN.match(self.id):
            raise ValueError(f"block id {self.id!r} does not match ch{{NNN}}_b{{NNNN}}")
        if self.type is BlockType.SCENE_BREAK and self.text:
            raise ValueError(f"scene_break block {self.id} must have empty text")
        if self.type is not BlockType.SCENE_BREAK and not self.text.strip():
            raise ValueError(f"{self.type} block {self.id} must have non-empty text")
        self._check_italic_spans()
        return self

    def _check_italic_spans(self) -> None:
        if self.type is BlockType.SCENE_BREAK and self.italic_spans:
            raise ValueError(f"scene_break block {self.id} must have no italic_spans")
        prev_end = 0
        n = len(self.text)
        for start, end in self.italic_spans:
            if not 0 <= start < end <= n:
                raise ValueError(
                    f"block {self.id} italic span ({start}, {end}) out of range for text len {n}"
                )
            if start < prev_end:
                raise ValueError(
                    f"block {self.id} italic_spans must be sorted and non-overlapping "
                    f"(span ({start}, {end}) follows end {prev_end})"
                )
            prev_end = end

    @property
    def is_speakable(self) -> bool:
        """True if this block produces synthesis segments (scene breaks never do)."""
        return self.type is not BlockType.SCENE_BREAK


class Chapter(BaseModel):
    title: str
    blocks: list[Block]


class BookMeta(BaseModel):
    book_id: str
    title: str
    authors: list[str] = []
    language: str | None = None
    source_path: str
    source_sha256: str


class NormalizedBook(BaseModel):
    book_meta: BookMeta
    chapters: list[Chapter]
