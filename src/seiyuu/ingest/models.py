"""Normalized book models — the documented contract between ingest and all later stages.

Schema (SPEC.md stage 1):
    {book_meta, chapters: [{title, blocks: [{type, id, text}]}]}

Block ids are stable and ordered: ``ch{NNN}_b{NNNN}`` (1-based, chapter-scoped).
"""

import re
from enum import StrEnum

from pydantic import BaseModel, model_validator

BLOCK_ID_PATTERN = re.compile(r"^ch\d{3}_b\d{4}$")


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

    @model_validator(mode="after")
    def _check_invariants(self) -> "Block":
        if not BLOCK_ID_PATTERN.match(self.id):
            raise ValueError(f"block id {self.id!r} does not match ch{{NNN}}_b{{NNNN}}")
        if self.type is BlockType.SCENE_BREAK and self.text:
            raise ValueError(f"scene_break block {self.id} must have empty text")
        if self.type is not BlockType.SCENE_BREAK and not self.text.strip():
            raise ValueError(f"{self.type} block {self.id} must have non-empty text")
        return self

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
