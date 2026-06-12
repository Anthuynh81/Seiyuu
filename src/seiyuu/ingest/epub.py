"""EPUB → normalized JSON (ingest stage).

Approach: extract an ordered block stream per spine document, concatenate the
streams (chapters may span file boundaries), then split into chapters at
headings. Front/back matter is stripped heuristically; spine items can be
force-included/excluded from the CLI.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook

# Spine items that are never narration (covers, navigation, wrappers).
SKIP_ITEM_PATTERN = re.compile(r"cover|\btoc\b|nav|titlepage|wrap", re.IGNORECASE)

# Containers whose text is never narration: illustration captions/figures and
# Project Gutenberg boilerplate (header and license footer share the class).
SKIP_CLASS_PATTERN = re.compile(
    r"caption|figcenter|figleft|figright|figure|illus|pg-boilerplate", re.IGNORECASE
)

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

# How many consecutive empty <p> elements count as a deliberate blank-line gap.
BLANK_GAP_RUN = 2


class IngestError(Exception):
    """Loud, actionable ingest failure."""


@dataclass
class RawBlock:
    kind: BlockType
    text: str = ""
    level: int = 0  # heading level, 0 for non-headings


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


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_doc_blocks(html: bytes) -> list[RawBlock]:
    """Ordered blocks from one spine document, with non-narration markup removed."""
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.find_all(class_=SKIP_CLASS_PATTERN):
        el.decompose()
    for el in soup.find_all("nav"):
        el.decompose()

    body = soup.body or soup
    blocks: list[RawBlock] = []
    empty_run = 0
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "hr"]):
        if el.name == "hr":
            empty_run = 0
            blocks.append(RawBlock(BlockType.SCENE_BREAK))
            continue
        text = _collapse(el.get_text())
        if el.name == "p":
            if not text:
                empty_run += 1
                if empty_run == BLANK_GAP_RUN:
                    blocks.append(RawBlock(BlockType.SCENE_BREAK))
                continue
            empty_run = 0
            if DECORATIVE_PATTERN.match(text):
                blocks.append(RawBlock(BlockType.SCENE_BREAK))
            else:
                blocks.append(RawBlock(BlockType.PARAGRAPH, text))
        else:  # heading
            empty_run = 0
            if text:
                blocks.append(RawBlock(BlockType.HEADING, text, level=int(el.name[1])))
    return blocks


def _chapterize(raws: list[RawBlock], split_level: int) -> list[ProtoChapter]:
    """Split the cross-document block stream at chapter-level headings.

    Blocks before the first heading form a (usually dropped) preamble; blocks
    after a heading belong to it even if they came from a later spine file.
    """
    chapters = [ProtoChapter(None)]
    for rb in raws:
        if rb.kind is BlockType.HEADING and rb.level <= split_level:
            chapters.append(ProtoChapter(rb.text, [rb]))
        else:
            chapters[-1].blocks.append(rb)
    return chapters


def _is_content(proto: ProtoChapter) -> bool:
    if proto.title and CHAPTER_TITLE_PATTERN.match(proto.title):
        return True
    return proto.word_count >= MATTER_MAX_WORDS


def _clean_blocks(raws: list[RawBlock]) -> list[RawBlock]:
    """Collapse scene-break runs and trim them from chapter edges."""
    out: list[RawBlock] = []
    for rb in raws:
        if rb.kind is BlockType.SCENE_BREAK and (not out or out[-1].kind is BlockType.SCENE_BREAK):
            continue
        out.append(rb)
    while out and out[-1].kind is BlockType.SCENE_BREAK:
        out.pop()
    return out


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-") or "book"


def _first_meta(book: epub.EpubBook, name: str) -> str | None:
    values = book.get_metadata("DC", name)
    return _collapse(values[0][0]) if values else None


def _matches_any(name: str, idref: str, needles: tuple[str, ...]) -> bool:
    return any(n.lower() in name.lower() or n.lower() in idref.lower() for n in needles)


def parse_epub(
    epub_path: Path,
    include_items: tuple[str, ...] = (),
    exclude_items: tuple[str, ...] = (),
    split_level: int = 2,
) -> IngestResult:
    epub_path = Path(epub_path).resolve()
    if not epub_path.is_file():
        raise IngestError(f"EPUB not found: {epub_path}")
    try:
        book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    except Exception as exc:
        raise IngestError(f"failed to read EPUB {epub_path}: {exc}") from exc

    title = _first_meta(book, "title") or epub_path.stem
    authors = [_collapse(v) for v, _ in book.get_metadata("DC", "creator")]
    language = _first_meta(book, "language")
    sha = hashlib.sha256(epub_path.read_bytes()).hexdigest()
    book_id = f"{_slug(title)}-{sha[:8]}"

    raws: list[RawBlock] = []
    skipped: list[str] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        name = item.get_name()
        if _matches_any(name, idref, exclude_items):
            skipped.append(name)
            continue
        if (SKIP_ITEM_PATTERN.search(name) or SKIP_ITEM_PATTERN.search(idref)) and not (
            _matches_any(name, idref, include_items)
        ):
            skipped.append(name)
            continue
        raws.extend(_extract_doc_blocks(item.get_content()))

    if not raws:
        raise IngestError(f"no narratable content found in {epub_path} (all items skipped?)")

    chapters: list[Chapter] = []
    dropped: list[str] = []
    for proto in _chapterize(raws, split_level):
        label = proto.title or "(untitled preamble)"
        if not _is_content(proto):
            if proto.blocks:
                dropped.append(f"{label} ({proto.word_count} words)")
            continue
        blocks = _clean_blocks(proto.blocks)
        if not any(b.kind is not BlockType.SCENE_BREAK for b in blocks):
            dropped.append(f"{label} (no speakable blocks)")
            continue
        ci = len(chapters) + 1
        chapters.append(
            Chapter(
                title=proto.title or "Untitled",
                blocks=[
                    Block(id=f"ch{ci:03d}_b{bi:04d}", type=rb.kind, text=rb.text)
                    for bi, rb in enumerate(blocks, start=1)
                ],
            )
        )

    if not chapters:
        raise IngestError(
            f"no chapters survived matter-stripping for {epub_path}; "
            f"dropped: {dropped}. Use --include-item or --split-level to adjust."
        )

    meta = BookMeta(
        book_id=book_id,
        title=title,
        authors=authors,
        language=language,
        source_path=str(epub_path),
        source_sha256=sha,
    )
    return IngestResult(
        book=NormalizedBook(book_meta=meta, chapters=chapters),
        skipped_items=skipped,
        dropped_sections=dropped,
    )


def write_normalized(book: NormalizedBook, books_dir: Path) -> Path:
    out_dir = Path(books_dir) / book.book_meta.book_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "normalized.json"
    out_path.write_text(book.model_dump_json(indent=2), encoding="utf-8")
    return out_path
