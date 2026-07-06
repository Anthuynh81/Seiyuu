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

from bs4 import BeautifulSoup, CData, NavigableString
from ebooklib import ITEM_DOCUMENT, ITEM_STYLE, epub

from seiyuu.ingest.css_italics import (
    EMPTY_ITALIC_MAP,
    INLINE_ITALIC,
    INLINE_NORMAL,
    ItalicStyleMap,
    parse_css_italics,
)
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.repository import atomic_write_text

# Spine items that are never narration (covers, navigation, wrappers). Matched
# as whole name tokens — substring matching falsely skipped chapters like
# "Azik's Discovery" (cover) and "Wrapping Up Work" (wrap). A name match alone
# is not enough: the item must also be nearly textless (see MATTER_DOC_MAX_WORDS).
SKIP_NAME_TOKENS = {"cover", "coverpage", "toc", "nav", "titlepage", "wrap", "wrapper"}

# A name-flagged spine item with more words than this is real content and kept.
MATTER_DOC_MAX_WORDS = 50

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


# Inline tags whose text is the standard unspoken-thought / emphasis signal. Kept as the
# highest-priority per-element signal so the tag italic is byte-identical to Phase 1 even
# when CSS restyles <em>/<i> to normal (see ItalicStyleMap.em_forced_normal, not acted on).
_ITALIC_TAGS = {"em", "i"}


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


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Single whitespace char, using the SAME class as ``_collapse``'s ``\s+`` so the italic
# walker collapses identically to ``_collapse(el.get_text())`` for the text itself.
_WS_CHAR = re.compile(r"\s")


def _element_font_style(el, italic_map: ItalicStyleMap) -> bool | None:
    """Font-style DECLARED by this single element: ``True`` italic, ``False`` normal-forced,
    ``None`` undeclared. Priority mirrors the resolution the walker needs, not raw CSS
    specificity: the ``<em>``/``<i>`` tag signal is kept first (byte-identical to Phase 1),
    then the inline ``style`` (nearer normal cancels), then a class token in the italic set.
    """
    if el.name in _ITALIC_TAGS:
        return True
    style = el.get("style")  # a STRING, or None if absent
    if style:
        if INLINE_ITALIC.search(style):
            return True
        if INLINE_NORMAL.search(style):
            return False
    classes = el.get("class")  # a LIST, or None if absent
    if classes:
        for token in classes:
            if token in italic_map.italic_classes:
                return True
    return None


def _collapse_with_italics(
    el, italic_map: ItalicStyleMap = EMPTY_ITALIC_MAP
) -> tuple[str, list[tuple[int, int]]]:
    """Collapsed text of ``el`` AND its italic run offsets, in a single pass.

    The offsets index the FINAL, post-collapse/strip text — never the raw ``get_text()``.
    ``_collapse`` rewrites every whitespace run to one space and strips the ends, which
    shifts positions; a "find in get_text() then collapse" shortcut would silently
    mis-slice. So we gather each descendant string with a flag for whether it is italic
    (bounded by ``el``), collapse the char stream once, and read the italic runs straight
    off the collapsed flags. The text half is identical to ``_collapse(el.get_text())``;
    scene-break/decorative handling is unchanged upstream.

    Italic is resolved inheritance-correctly (Phase 2a): for each string node we climb its
    ancestors up to (but excluding) ``el`` and take the NEAREST ancestor that declares a
    font-style — an ``<em>``/``<i>`` tag, an inline ``style="font-style:italic|oblique"``,
    or a class in ``italic_map.italic_classes``. font-style inherits, so a class on a
    wrapper italicizes its descendants; a nearer ``style="font-style:normal"`` (or a
    normal-forced element) cancels it. ``el``'s OWN font-style is deliberately invisible —
    a fully-italic paragraph (letter/telegraph) must not become a mid-prose thought.
    """
    raw_chars: list[str] = []
    raw_italic: list[bool] = []
    for node in el.descendants:
        # Match ``get_text()`` semantics EXACTLY: it yields only the base string
        # types (NavigableString, CData). ``isinstance`` would also catch NavigableString
        # subclasses — Comment, Stylesheet, Script, Declaration, ProcessingInstruction —
        # which ``get_text()`` excludes, injecting comment/script/CSS text into narration
        # and breaking the thought-off byte-identity invariant.
        if type(node) not in (NavigableString, CData):
            continue
        italic = False
        for parent in node.parents:
            if parent is el:
                break
            decl = _element_font_style(parent, italic_map)
            if decl is not None:  # nearest declaring ancestor wins (normal cancels italic)
                italic = decl
                break
        for ch in str(node):
            raw_chars.append(ch)
            raw_italic.append(italic)

    # Collapse each maximal whitespace run to a single space, mirroring
    # ``re.sub(r"\s+", " ", ...)``. The collapsed space stays italic ONLY when italic on BOTH
    # sides (interior to an italic run); a boundary seam is non-italic, so an italic run never
    # absorbs the whitespace that separates it from surrounding prose.
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


def _extract_doc_blocks(
    html: bytes, italic_map: ItalicStyleMap = EMPTY_ITALIC_MAP
) -> list[RawBlock]:
    """Ordered blocks from one spine document, with non-narration markup removed.

    ``italic_map`` widens the italic signal beyond inline ``<em>``/``<i>`` to CSS-class and
    inline-style italics (Phase 2a). The default empty map reduces the walker to Phase-1
    tag-only behavior exactly (byte-identity).
    """
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
        text, italic_spans = _collapse_with_italics(el, italic_map)
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
                blocks.append(RawBlock(BlockType.PARAGRAPH, text, italic_spans=italic_spans))
        else:  # heading
            empty_run = 0
            if text:
                blocks.append(
                    RawBlock(
                        BlockType.HEADING, text, level=int(el.name[1]), italic_spans=italic_spans
                    )
                )
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


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")
    if len(s) > max_len:
        s = s[:max_len]
        if "-" in s:
            s = s[: s.rfind("-")]
    return s or "book"


def _name_tokens(*names: str) -> set[str]:
    tokens: set[str] = set()
    for name in names:
        tokens.update(re.findall(r"[a-z]+|\d+", name.lower()))
    return tokens


def _first_meta(book: epub.EpubBook, name: str) -> str | None:
    values = book.get_metadata("DC", name)
    return _collapse(values[0][0]) if values else None


def _matches_any(name: str, idref: str, needles: tuple[str, ...]) -> bool:
    return any(n.lower() in name.lower() or n.lower() in idref.lower() for n in needles)


def _build_italic_style_map(book: epub.EpubBook) -> ItalicStyleMap:
    """Once-per-book union of every italic CSS class from external sheets AND in-document
    ``<style>`` blocks. External sheets are ITEM_STYLE items; in-document ``<style>`` is not
    ITEM_STYLE and its text is (correctly) excluded from narration, so it is read here in a
    SEPARATE pass — the narration string filter in ``_collapse_with_italics`` is untouched.
    """
    sheets: list[str] = []
    for item in book.get_items_of_type(ITEM_STYLE):
        sheets.append(item.get_content().decode("utf-8", errors="replace"))
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for style in soup.find_all("style"):
            sheets.append(style.get_text())
    return parse_css_italics(sheets)


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

    # Book-global italic class set (external sheets + in-document <style>), built BEFORE the
    # spine content loop so every document resolves CSS italics against the same map.
    italic_map = _build_italic_style_map(book)

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
        doc_blocks = _extract_doc_blocks(item.get_content(), italic_map)
        doc_words = sum(len(b.text.split()) for b in doc_blocks)
        if (
            (_name_tokens(name, idref) & SKIP_NAME_TOKENS)
            and doc_words <= MATTER_DOC_MAX_WORDS
            and not _matches_any(name, idref, include_items)
        ):
            skipped.append(name)
            continue
        raws.extend(doc_blocks)

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
    out_path = Path(books_dir) / book.book_meta.book_id / "normalized.json"
    return atomic_write_text(out_path, book.model_dump_json(indent=2))
