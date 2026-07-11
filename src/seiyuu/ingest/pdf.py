"""PDF → normalized JSON (ingest stage, M8).

Approach: extract span-level text per page (pymupdf), drop running headers/footers
and bare page numbers, reflow lines into paragraphs (de-hyphenating line-break
hyphens, preserving span italics for the thought signal), then chapterize — by the
PDF outline when present, by a conservative font-size heading heuristic otherwise;
a book with neither becomes one chapter. Matter stripping and the normalized schema
are shared with the EPUB parser (``ingest.common``). Scanned/image-only PDFs are
refused loudly: OCR is out of scope.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from seiyuu.ingest.common import (
    CHAPTER_TITLE_PATTERN,
    DECORATIVE_PATTERN,
    IngestError,
    IngestResult,
    ProtoChapter,
    RawBlock,
    assemble_chapters,
    chapterize,
    collapse,
    collapse_flagged,
    is_content,
    slug,
)
from seiyuu.ingest.models import BlockType, BookMeta, NormalizedBook

# pymupdf span flag bits (TextPage "dict" spans).
_ITALIC_FLAG = 1 << 1
_BOLD_FLAG = 1 << 4

# A line that is just a page number (arabic or roman numerals).
_PAGE_NUMBER = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,7})$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")

# Top/bottom fraction of the page height where running headers/footers live.
_HF_BAND = 0.08
# A banded line whose digit-normalized text repeats on at least this fraction of
# pages (min 3) is a running header/footer.
_HF_MIN_FRACTION = 0.3

# Fewer extracted words than this across the whole document = no usable text layer.
_MIN_TEXT_WORDS = 25

# First-line indent (pt) past the modal left margin that starts a new paragraph.
_INDENT_PT = 6.0
# Font-size jump (pt) between consecutive lines that forces a paragraph boundary.
_SIZE_JUMP = 1.5
# Same-page vertical gap: more than max(3pt, this fraction of the previous line's
# height) of clear space reads as paragraph spacing.
_GAP_FRACTION = 0.45

# Heading heuristic (used only when the PDF has no usable outline): a candidate is
# short, and either clearly larger than the body face or bold with a chapter-like
# title. Fewer than 2 detected chapter headings = no confidence, single chapter.
_HEADING_MAX_WORDS = 12
_HEADING_MAX_CHARS = 80
_HEADING_MIN_DELTA = 1.5
_H1_DELTA = 4.0
_MIN_DETECTED_HEADINGS = 2

# Sentence-final punctuation that lets a paragraph end at a page boundary.
_TERMINAL = (".", "?", "!", ":", ";", '"', "'", "’", "”", "…")


@dataclass
class _Line:
    chars: list[str]
    italic: list[bool]
    text: str
    size: float
    bold: bool
    x0: float
    y0: float
    y1: float
    page: int
    page_height: float


@dataclass
class _Para:
    chars: list[str] = field(default_factory=list)
    italic: list[bool] = field(default_factory=list)
    size: float = 0.0  # largest span size seen
    bold_chars: int = 0
    total_chars: int = 0
    line_count: int = 0
    page: int = 0  # page of the first line

    def add_line(self, line: _Line) -> None:
        if self.line_count == 0:
            self.page = line.page
        elif self.chars and self.chars[-1] == "-" and line.chars and line.chars[0].islower():
            # line-break hyphenation: "exam-" + "ple" → "example"
            self.chars.pop()
            self.italic.pop()
        else:
            self.chars.append("\n")  # seam whitespace; collapse_flagged handles italics
            self.italic.append(False)
        self.chars.extend(line.chars)
        self.italic.extend(line.italic)
        self.size = max(self.size, line.size)
        if line.bold:
            self.bold_chars += len(line.chars)
        self.total_chars += len(line.chars)
        self.line_count += 1

    @property
    def mostly_bold(self) -> bool:
        return self.total_chars > 0 and self.bold_chars > self.total_chars / 2


def _extract_lines(doc: pymupdf.Document) -> list[_Line]:
    """Ordered text lines with per-char italic flags; images are ignored."""
    lines: list[_Line] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        d = page.get_text("dict", sort=True)
        page_h = float(d.get("height") or page.rect.height)
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:  # 0 = text, 1 = image
                continue
            for ln in block.get("lines", []):
                chars: list[str] = []
                italics: list[bool] = []
                sizes: list[float] = []
                bold = 0
                total = 0
                for span in ln.get("spans", []):
                    t = span.get("text", "")
                    if not t:
                        continue
                    flags = int(span.get("flags", 0))
                    it = bool(flags & _ITALIC_FLAG)
                    chars.extend(t)
                    italics.extend([it] * len(t))
                    sizes.append(float(span.get("size", 0.0)))
                    if flags & _BOLD_FLAG:
                        bold += len(t)
                    total += len(t)
                text = "".join(chars).strip()
                if not text:
                    continue
                x0, y0, _x1, y1 = ln["bbox"]
                lines.append(
                    _Line(
                        chars=chars,
                        italic=italics,
                        text=text,
                        size=max(sizes),
                        bold=bold > total / 2,
                        x0=float(x0),
                        y0=float(y0),
                        y1=float(y1),
                        page=pno,
                        page_height=page_h,
                    )
                )
    return lines


def _banded(line: _Line) -> bool:
    return line.y0 < line.page_height * _HF_BAND or line.y1 > line.page_height * (1 - _HF_BAND)


def _furniture_key(line: _Line) -> str:
    return _DIGITS.sub("#", collapse(line.text)).lower()


def _strip_furniture(lines: list[_Line], page_count: int) -> list[_Line]:
    """Drop running headers/footers (repeating banded lines) and bare page numbers."""
    pages_seen: dict[str, set[int]] = {}
    for line in lines:
        if _banded(line):
            pages_seen.setdefault(_furniture_key(line), set()).add(line.page)
    threshold = max(3, int(page_count * _HF_MIN_FRACTION))
    out: list[_Line] = []
    for line in lines:
        if _banded(line):
            if _PAGE_NUMBER.match(line.text):
                continue
            if len(pages_seen.get(_furniture_key(line), ())) >= threshold:
                continue
        out.append(line)
    return out


def _body_size(lines: list[_Line]) -> float:
    """The dominant font size by character count — the running-text face."""
    weights: dict[float, int] = {}
    for line in lines:
        key = round(line.size * 2) / 2
        weights[key] = weights.get(key, 0) + len(line.chars)
    return max(weights.items(), key=lambda kv: kv[1])[0] if weights else 12.0


def _left_margin(lines: list[_Line], body: float) -> float:
    """Modal left edge of body-sized lines — the paragraph continuation margin."""
    counts: dict[int, int] = {}
    for line in lines:
        if abs(line.size - body) <= 1.0:
            key = round(line.x0)
            counts[key] = counts.get(key, 0) + 1
    return float(max(counts.items(), key=lambda kv: kv[1])[0]) if counts else 0.0


def _reflow(lines: list[_Line], left_margin: float) -> list[_Para]:
    """Merge lines into paragraphs.

    A new paragraph starts on: a font-size jump (heading boundary), a first-line
    indent, a vertical gap larger than paragraph spacing, or a page turn where the
    previous line already ended a sentence. Anything else continues the paragraph —
    including across pages, so a paragraph split by a page break is rejoined.
    """
    paras: list[_Para] = []
    prev: _Line | None = None
    for line in lines:
        new = prev is None
        if prev is not None:
            if abs(line.size - prev.size) > _SIZE_JUMP:
                new = True
            elif line.page != prev.page:
                new = prev.text.endswith(_TERMINAL) or line.x0 - left_margin > _INDENT_PT
            else:
                gap = line.y0 - prev.y1
                prev_h = max(prev.y1 - prev.y0, 1.0)
                new = gap > max(3.0, _GAP_FRACTION * prev_h) or line.x0 - left_margin > _INDENT_PT
        if new:
            paras.append(_Para())
        paras[-1].add_line(line)
        prev = line
    return paras


def _paras_to_raws(paras: list[_Para]) -> list[tuple[RawBlock, _Para]]:
    out: list[tuple[RawBlock, _Para]] = []
    for p in paras:
        text, spans = collapse_flagged(p.chars, p.italic)
        if not text:
            continue
        if DECORATIVE_PATTERN.match(text):
            out.append((RawBlock(BlockType.SCENE_BREAK), p))
            continue
        # Mirror the EPUB rule: a fully-italic paragraph (letter, telegram) is styling,
        # not a mid-prose thought — the paragraph-level italic signal is invisible.
        if spans == [(0, len(text))]:
            spans = []
        out.append((RawBlock(BlockType.PARAGRAPH, text, italic_spans=spans), p))
    return out


def _toc_entries(doc: pymupdf.Document, split_level: int) -> list[tuple[int, str, int]]:
    """Usable outline entries as (level, title, 0-based page); [] when no real TOC."""
    try:
        toc = doc.get_toc(simple=True) or []
    except Exception:
        return []
    entries = [
        (int(lvl), collapse(title), int(page) - 1)
        for lvl, title, page in toc
        if collapse(title) and int(lvl) <= split_level and int(page) >= 1
    ]
    return entries if len(entries) >= _MIN_DETECTED_HEADINGS else []


def _apply_toc(
    para_raws: list[tuple[RawBlock, _Para]], entries: list[tuple[int, str, int]]
) -> list[RawBlock]:
    """Insert outline headings into the paragraph stream.

    Each entry lands before the first paragraph on its page; when a paragraph on
    that page (or the next — outlines often point at a title page) IS the title
    text, it is consumed into the heading instead of duplicating it.
    """
    raws = [rb for rb, _ in para_raws]
    pages = [p.page for _, p in para_raws]
    result: list[RawBlock] = []
    idx = 0
    for lvl, title, page in entries:
        target = idx
        while target < len(raws) and pages[target] < page:
            target += 1
        result.extend(raws[idx:target])
        idx = target
        title_lower = title.lower()
        match = None
        j = idx
        while j < len(raws) and pages[j] <= page + 1:
            para_lower = raws[j].text.lower()
            if para_lower and (
                para_lower.startswith(title_lower[:60]) or title_lower.startswith(para_lower[:60])
            ):
                match = j
                break
            j += 1
        if match is not None:
            result.extend(raws[idx:match])
            result.append(RawBlock(BlockType.HEADING, raws[match].text, level=lvl))
            idx = match + 1
        else:
            result.append(RawBlock(BlockType.HEADING, title, level=lvl))
    result.extend(raws[idx:])
    return result


def _heading_marks(para_raws: list[tuple[RawBlock, _Para]], body: float) -> list[tuple[int, int]]:
    """Conservative no-TOC heading detection → (index, level) candidates.

    A heading is short and either clearly larger than the body face, or bold with a
    chapter-like title (the shared CHAPTER_TITLE_PATTERN).
    """
    marks: list[tuple[int, int]] = []
    for i, (rb, p) in enumerate(para_raws):
        if rb.kind is not BlockType.PARAGRAPH:
            continue
        text = rb.text
        if len(text) > _HEADING_MAX_CHARS or len(text.split()) > _HEADING_MAX_WORDS:
            continue
        big = p.size >= body + _HEADING_MIN_DELTA
        titled = bool(CHAPTER_TITLE_PATTERN.match(text))
        if big or (p.mostly_bold and titled):
            level = 1 if p.size >= body + _H1_DELTA else 2
            marks.append((i, level))
    return marks


def _matches_title(title: str, needles: tuple[str, ...]) -> bool:
    return any(n.lower() in title.lower() for n in needles)


def parse_pdf(
    pdf_path: Path,
    include_items: tuple[str, ...] = (),
    exclude_items: tuple[str, ...] = (),
    split_level: int = 2,
) -> IngestResult:
    """Parse a text-layer PDF into the normalized book.

    ``include_items``/``exclude_items`` match against chapter TITLES (a PDF has no
    spine): exclude drops a chapter outright; include keeps one the matter-stripping
    heuristics would drop.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise IngestError(f"PDF not found: {pdf_path}")
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise IngestError(f"failed to read PDF {pdf_path}: {exc}") from exc
    try:
        if doc.needs_pass:
            raise IngestError(f"{pdf_path} is password-protected — remove the password first")

        meta_raw = doc.metadata or {}
        title = collapse(meta_raw.get("title") or "") or pdf_path.stem
        author_raw = collapse(meta_raw.get("author") or "")
        authors = [a for a in (collapse(x) for x in re.split(r"[;&]", author_raw)) if a]
        language = getattr(doc, "language", None) or None
        sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        book_id = f"{slug(title)}-{sha[:8]}"

        lines = _extract_lines(doc)
        total_words = sum(len(line.text.split()) for line in lines)
        if total_words < _MIN_TEXT_WORDS:
            raise IngestError(
                f"{pdf_path} has no usable text layer ({total_words} words across "
                f"{doc.page_count} pages) — a scanned/image-only PDF needs OCR, "
                "which is not supported"
            )

        lines = _strip_furniture(lines, doc.page_count)
        body = _body_size(lines)
        para_raws = _paras_to_raws(_reflow(lines, _left_margin(lines, body)))
        if not para_raws:
            raise IngestError(f"no narratable content found in {pdf_path}")

        entries = _toc_entries(doc, split_level)
        if entries:
            raws = _apply_toc(para_raws, entries)
        else:
            marks = _heading_marks(para_raws, body)
            if sum(1 for _, lvl in marks if lvl <= split_level) >= _MIN_DETECTED_HEADINGS:
                for i, lvl in marks:
                    rb, p = para_raws[i]
                    para_raws[i] = (RawBlock(BlockType.HEADING, rb.text, level=lvl), p)
            raws = [rb for rb, _ in para_raws]

        skipped: list[str] = []
        has_headings = any(rb.kind is BlockType.HEADING and rb.level <= split_level for rb in raws)
        if has_headings:
            protos: list[ProtoChapter] = []
            for proto in chapterize(raws, split_level):
                if proto.title and _matches_title(proto.title, exclude_items):
                    skipped.append(proto.title)
                    continue
                protos.append(proto)

            def keep(proto: ProtoChapter) -> bool:
                if proto.title and _matches_title(proto.title, include_items):
                    return True
                return is_content(proto)

            chapters, dropped = assemble_chapters(protos, keep=keep)
        else:
            # No outline and no confident headings: the whole book is one chapter.
            # The text-layer check above already guaranteed real content, so the
            # matter-stripping word threshold does not apply here.
            protos = [ProtoChapter(title, list(raws))]
            chapters, dropped = assemble_chapters(protos, keep=lambda _proto: True)

        if not chapters:
            raise IngestError(
                f"no chapters survived matter-stripping for {pdf_path}; "
                f"dropped: {dropped}. Use --include-item or --split-level to adjust."
            )

        meta = BookMeta(
            book_id=book_id,
            title=title,
            authors=authors,
            language=language,
            source_path=str(pdf_path),
            source_sha256=sha,
        )
        return IngestResult(
            book=NormalizedBook(book_meta=meta, chapters=chapters),
            skipped_items=skipped,
            dropped_sections=dropped,
        )
    finally:
        doc.close()
