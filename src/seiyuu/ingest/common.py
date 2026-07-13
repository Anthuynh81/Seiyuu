"""Format-agnostic ingest core shared by the EPUB and PDF parsers.

Both parsers reduce their source to an ordered ``RawBlock`` stream, then the same
machinery chapterizes at headings, strips front/back matter heuristically, and
assembles the normalized ``Chapter``/``Block`` models with stable ids.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from seiyuu.ingest.models import Block, BlockType, Chapter, NormalizedBook

# A paragraph made only of decorative symbols (e.g. ***, * * *, ~ ~ ~, ---).
DECORATIVE_PATTERN = re.compile(r"^[\s*~\-—–_=•·.#•◦]+$")

# Heading texts that mark real content sections (vs front/back matter).
CHAPTER_TITLE_PATTERN = re.compile(
    r"^(chapter|part|book|volume|canto|stave|prologue|epilogue|preface|introduction|interlude)\b"
    r"|^[ivxlcdm]+\.?$"
    r"|^\d+\.?$",
    re.IGNORECASE,
)

# Sections whose title doesn't look like a chapter AND that are shorter than
# this many words are treated as front/back matter and dropped (reported).
MATTER_MAX_WORDS = 150


class IngestError(Exception):
    """Loud, actionable ingest failure."""


@dataclass
class RawBlock:
    kind: BlockType
    text: str = ""
    level: int = 0  # heading level, 0 for non-headings
    italic_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ProtoChapter:
    title: str | None
    blocks: list[RawBlock] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)


@dataclass
class IngestResult:
    book: NormalizedBook
    skipped_items: list[str]
    dropped_sections: list[str]
    # Raw bytes of the source's DECLARED cover image (EPUB only today); validated
    # (jpeg/png magic) only when written, by ``extract_cover_art``.
    cover: bytes | None = None


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Single whitespace char, using the SAME class as ``collapse``'s ``\s+`` so flagged
# collapsing produces text identical to ``collapse("".join(chars))``.
_WS_CHAR = re.compile(r"\s")


def collapse_flagged(
    raw_chars: list[str], raw_italic: list[bool]
) -> tuple[str, list[tuple[int, int]]]:
    """Collapse a flagged char stream to (text, italic run offsets) in one pass.

    Mirrors ``re.sub(r"\\s+", " ", ...).strip()`` exactly for the text. A collapsed
    space stays italic ONLY when italic on BOTH sides (interior to an italic run); a
    boundary seam is non-italic, so an italic run never absorbs the whitespace that
    separates it from surrounding prose. Offsets index the FINAL text.
    """
    out_chars: list[str] = []
    out_italic: list[bool] = []
    i = 0
    n = len(raw_chars)
    while i < n:
        if _WS_CHAR.match(raw_chars[i]):
            j = i + 1
            while j < n and _WS_CHAR.match(raw_chars[j]):
                j += 1
            left_italic = raw_italic[i - 1] if i > 0 else False
            right_italic = raw_italic[j] if j < n else False
            out_chars.append(" ")
            out_italic.append(left_italic and right_italic)
            i = j
        else:
            out_chars.append(raw_chars[i])
            out_italic.append(raw_italic[i])
            i += 1

    # Strip the (single, non-italic) leading/trailing collapsed spaces, mirroring ``.strip()``.
    start = 0
    end = len(out_chars)
    while start < end and out_chars[start] == " ":
        start += 1
    while end > start and out_chars[end - 1] == " ":
        end -= 1

    text = "".join(out_chars[start:end])
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        if out_italic[pos]:
            run_start = pos
            while pos < end and out_italic[pos]:
                pos += 1
            spans.append((run_start - start, pos - start))
        else:
            pos += 1
    return text, spans


def chapterize(raws: list[RawBlock], split_level: int) -> list[ProtoChapter]:
    """Split the block stream at chapter-level headings.

    Blocks before the first heading form a (usually dropped) preamble; blocks
    after a heading belong to it even if they came from a later source unit.
    """
    chapters = [ProtoChapter(None)]
    for rb in raws:
        if rb.kind is BlockType.HEADING and rb.level <= split_level:
            chapters.append(ProtoChapter(rb.text, [rb]))
        else:
            chapters[-1].blocks.append(rb)
    return chapters


def is_content(proto: ProtoChapter) -> bool:
    if proto.title and CHAPTER_TITLE_PATTERN.match(proto.title):
        return True
    return proto.word_count >= MATTER_MAX_WORDS


def clean_blocks(raws: list[RawBlock]) -> list[RawBlock]:
    """Collapse scene-break runs and trim them from chapter edges."""
    out: list[RawBlock] = []
    for rb in raws:
        if rb.kind is BlockType.SCENE_BREAK and (not out or out[-1].kind is BlockType.SCENE_BREAK):
            continue
        out.append(rb)
    while out and out[-1].kind is BlockType.SCENE_BREAK:
        out.pop()
    return out


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")
    if len(s) > max_len:
        s = s[:max_len]
        if "-" in s:
            s = s[: s.rfind("-")]
    return s or "book"


def assemble_chapters(
    protos: list[ProtoChapter],
    keep: Callable[[ProtoChapter], bool] = is_content,
) -> tuple[list[Chapter], list[str]]:
    """Matter-strip and number the surviving chapters; report what was dropped."""
    chapters: list[Chapter] = []
    dropped: list[str] = []
    for proto in protos:
        label = proto.title or "(untitled preamble)"
        if not keep(proto):
            if proto.blocks:
                dropped.append(f"{label} ({proto.word_count} words)")
            continue
        blocks = clean_blocks(proto.blocks)
        if not any(b.kind is not BlockType.SCENE_BREAK for b in blocks):
            dropped.append(f"{label} (no speakable blocks)")
            continue
        ci = len(chapters) + 1
        chapters.append(
            Chapter(
                title=proto.title or "Untitled",
                blocks=[
                    Block(
                        id=f"ch{ci:03d}_b{bi:04d}",
                        type=rb.kind,
                        text=rb.text,
                        italic_spans=rb.italic_spans,
                    )
                    for bi, rb in enumerate(blocks, start=1)
                ],
            )
        )
    return chapters, dropped
