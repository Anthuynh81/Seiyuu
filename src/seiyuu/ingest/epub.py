"""EPUB → normalized JSON (ingest stage).

Approach: extract an ordered block stream per spine document, concatenate the
streams (chapters may span file boundaries), then split into chapters at
headings. Front/back matter is stripped heuristically; spine items can be
force-included/excluded from the CLI.
"""

import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup, CData, NavigableString
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_STYLE, epub

from seiyuu.ingest.common import (
    CHAPTER_TITLE_PATTERN,
    DECORATIVE_PATTERN,
    MATTER_MAX_WORDS,
    IngestError,
    IngestResult,
    ProtoChapter,
    RawBlock,
    assemble_chapters,
    collapse_flagged,
)
from seiyuu.ingest.common import (
    chapterize as _chapterize,
)
from seiyuu.ingest.common import (
    collapse as _collapse,
)
from seiyuu.ingest.common import (
    slug as _slug,
)
from seiyuu.ingest.css_italics import (
    EMPTY_ITALIC_MAP,
    INLINE_ITALIC,
    INLINE_NORMAL,
    ItalicStyleMap,
    parse_css_italics,
)
from seiyuu.ingest.models import BlockType, BookMeta, NormalizedBook
from seiyuu.repository import atomic_write_text

__all__ = [
    "CHAPTER_TITLE_PATTERN",
    "DECORATIVE_PATTERN",
    "MATTER_MAX_WORDS",
    "IngestError",
    "IngestResult",
    "ProtoChapter",
    "RawBlock",
    "parse_epub",
    "write_normalized",
]

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

# How many consecutive empty <p> elements count as a deliberate blank-line gap.
BLANK_GAP_RUN = 2

# Inline tags whose text is the standard unspoken-thought / emphasis signal. Kept as the
# highest-priority per-element signal so the tag italic is byte-identical to Phase 1 even
# when CSS restyles <em>/<i> to normal (see ItalicStyleMap.em_forced_normal, not acted on).
_ITALIC_TAGS = {"em", "i"}


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

    return collapse_flagged(raw_chars, raw_italic)


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


def _find_cover(book: epub.EpubBook) -> bytes | None:
    """Bytes of the book's DECLARED cover image, or None when it declares none.

    EPUB3 marks the image itself with the ``cover-image`` manifest property (ebooklib
    types it ITEM_COVER); EPUB2 instead points ``<meta name="cover" content="..."/>`` at
    a manifest item id. Declared covers only — no filename guessing; jpeg/png validation
    happens at write time (``extract_cover_art``), not here.
    """
    for item in book.get_items_of_type(ITEM_COVER):
        content = item.get_content()
        if content:
            return content
    # The meta tag inherits the OPF default namespace in well-formed books; ancient
    # files without one land under the None key. get_metadata raises KeyError (not [])
    # when the whole namespace is absent from the book.
    for namespace in ("OPF", None):
        try:
            metas = book.get_metadata(namespace, "meta")
        except KeyError:
            continue
        for _value, attrs in metas:
            if attrs.get("name") != "cover" or not attrs.get("content"):
                continue
            item = book.get_item_with_id(attrs["content"])
            content = item.get_content() if item is not None else None
            if content:
                return content
    return None


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

    chapters, dropped = assemble_chapters(_chapterize(raws, split_level))

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
        cover=_find_cover(book),
    )


def write_normalized(book: NormalizedBook, books_dir: Path) -> Path:
    out_path = Path(books_dir) / book.book_meta.book_id / "normalized.json"
    return atomic_write_text(out_path, book.model_dump_json(indent=2))
