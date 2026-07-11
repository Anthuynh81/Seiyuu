"""PDF ingestion (M8): synthetic fixtures generated with pymupdf at test time.

No binary fixtures in the repo — each test builds exactly the PDF geometry it
needs (fonts, positions, outline) so every heuristic is exercised deliberately:
TOC/heading/single-chapter chapterization, hyphenation repair, running header
and page-number stripping, scene breaks, italic spans, and the scanned refusal.
"""

from pathlib import Path

import pymupdf
import pytest

from seiyuu.ingest import BlockType, IngestError, parse_book, parse_pdf
from seiyuu.ingest.pdf import _MIN_TEXT_WORDS

BODY = 11.0
HEAD = 16.0
SERIF = "tiro"  # Times-Roman base-14 alias
SERIF_ITALIC = "tiit"
LINE = 14  # body leading (pt)


class PdfBuilder:
    """Minimal typesetter: absolute-positioned lines so tests control geometry."""

    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page = None
        self.y = 0.0

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=400, height=600)
        self.y = 80.0

    def line(
        self,
        text: str,
        *,
        size: float = BODY,
        x: float = 50.0,
        y: float | None = None,
        font: str = SERIF,
        gap: float = 0.0,
    ) -> None:
        if self.page is None:
            self.new_page()
        if y is not None:
            self.y = y
        self.y += gap
        self.page.insert_text((x, self.y), text, fontsize=size, fontname=font)
        self.y += LINE

    def para(self, text: str, *, width_chars: int = 55, indent: float = 12.0) -> None:
        """A body paragraph: indented first line, wrapped continuation lines."""
        words = text.split()
        lines: list[str] = [""]
        for w in words:
            candidate = f"{lines[-1]} {w}".strip()
            if len(candidate) > width_chars and lines[-1]:
                lines.append(w)
            else:
                lines[-1] = candidate
        for i, ln in enumerate(lines):
            self.line(ln, x=50.0 + (indent if i == 0 else 0.0))

    def save(self, path: Path, toc: list[list] | None = None) -> Path:
        if toc:
            self.doc.set_toc(toc)
        self.doc.save(str(path))
        self.doc.close()
        return path


PROSE = (
    "The studio smelled of dust and warm tape, and nobody had touched the faders "
    "in years. She counted the reels twice before she trusted herself to speak, "
    "and even then the words came out smaller than she meant them to be."
)


def _chapter_pages(b: PdfBuilder, titles: list[str], paras_per_chapter: int = 3) -> None:
    for title in titles:
        b.new_page()
        b.line(title, size=HEAD)
        b.y += 6
        for _ in range(paras_per_chapter):
            b.para(PROSE)


# ---------------------------------------------------------------- chapterization


def test_toc_chapterization(tmp_path: Path) -> None:
    b = PdfBuilder()
    _chapter_pages(b, ["Chapter 1 The Reels", "Chapter 2 The Splice"])
    pdf = b.save(
        tmp_path / "toc.pdf",
        toc=[[1, "Chapter 1 The Reels", 1], [1, "Chapter 2 The Splice", 2]],
    )
    book = parse_pdf(pdf).book
    assert [c.title for c in book.chapters] == ["Chapter 1 The Reels", "Chapter 2 The Splice"]
    # the on-page title line was consumed into the heading, not duplicated
    ch1 = book.chapters[0]
    assert ch1.blocks[0].type is BlockType.HEADING
    assert sum(1 for blk in ch1.blocks if "Chapter 1" in blk.text) == 1


def test_heading_heuristic_without_toc(tmp_path: Path) -> None:
    b = PdfBuilder()
    _chapter_pages(b, ["Chapter 1", "Chapter 2", "Chapter 3"])
    book = parse_pdf(b.save(tmp_path / "heads.pdf")).book
    assert [c.title for c in book.chapters] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert all(c.blocks[0].type is BlockType.HEADING for c in book.chapters)


def test_single_chapter_fallback(tmp_path: Path) -> None:
    b = PdfBuilder()
    b.new_page()
    for _ in range(4):
        b.para(PROSE)
    b.doc.set_metadata({"title": "Flat Novella", "author": "A. Uthor; B. Uthor"})
    book = parse_pdf(b.save(tmp_path / "flat.pdf")).book
    assert len(book.chapters) == 1
    assert book.chapters[0].title == "Flat Novella"
    assert book.book_meta.authors == ["A. Uthor", "B. Uthor"]
    assert book.book_meta.book_id.startswith("flat-novella-")


# ---------------------------------------------------------------- text mechanics


def test_paragraph_reflow_and_hyphenation(tmp_path: Path) -> None:
    b = PdfBuilder()
    b.new_page()
    b.line("The reel spun in a slow, delib-", x=62.0)
    b.line("erate circle before it stopped.")
    b.line("A second paragraph starts here after the first one closed.", x=62.0)
    for _ in range(3):
        b.para(PROSE)
    book = parse_pdf(b.save(tmp_path / "hyph.pdf")).book
    texts = [blk.text for c in book.chapters for blk in c.blocks]
    assert "The reel spun in a slow, deliberate circle before it stopped." in texts
    assert "A second paragraph starts here after the first one closed." in texts


def test_paragraph_joins_across_pages(tmp_path: Path) -> None:
    b = PdfBuilder()
    b.new_page()
    for _ in range(2):
        b.para(PROSE)
    b.line("The last line of the page ended mid", y=560.0)
    b.new_page()
    b.line("sentence and continued overleaf.")
    for _ in range(2):
        b.para(PROSE)
    book = parse_pdf(b.save(tmp_path / "join.pdf")).book
    texts = [blk.text for c in book.chapters for blk in c.blocks]
    assert "The last line of the page ended mid sentence and continued overleaf." in texts


def test_headers_footers_and_page_numbers_dropped(tmp_path: Path) -> None:
    b = PdfBuilder()
    for pno in range(4):
        b.new_page()
        b.line("THE LONG REHEARSAL", y=20.0, size=9.0)  # running header
        for _ in range(2):
            b.para(PROSE)
        b.line(str(pno + 1), y=585.0, size=9.0)  # page number
    book = parse_pdf(b.save(tmp_path / "furniture.pdf")).book
    all_text = " ".join(blk.text for c in book.chapters for blk in c.blocks)
    assert "THE LONG REHEARSAL" not in all_text
    assert " 3 " not in f" {all_text} "


def test_scene_break_marker(tmp_path: Path) -> None:
    b = PdfBuilder()
    b.new_page()
    for _ in range(2):
        b.para(PROSE)
    b.line("* * *", x=170.0, gap=10.0)
    b.y += 10.0
    for _ in range(2):
        b.para(PROSE)
    book = parse_pdf(b.save(tmp_path / "scene.pdf")).book
    kinds = [blk.type for c in book.chapters for blk in c.blocks]
    assert BlockType.SCENE_BREAK in kinds


def test_italic_spans_captured(tmp_path: Path) -> None:
    b = PdfBuilder()
    b.new_page()
    for _ in range(2):
        b.para(PROSE)
    # one line with an interior italic word, hand-positioned span by span
    y = b.y + LINE
    page = b.page
    x = 50.0
    runs = [("She thought ", SERIF), ("never again", SERIF_ITALIC), (" and smiled.", SERIF)]
    for text, font in runs:
        page.insert_text((x, y), text, fontsize=BODY, fontname=font)
        x += pymupdf.get_text_length(text, fontname=font, fontsize=BODY)
    b.y = y + LINE + 20  # keep the builder's cursor clear of the manual line
    for _ in range(2):
        b.para(PROSE)
    book = parse_pdf(b.save(tmp_path / "ital.pdf")).book
    target = next(
        blk for c in book.chapters for blk in c.blocks if blk.text.startswith("She thought")
    )
    assert len(target.italic_spans) == 1
    start, end = target.italic_spans[0]
    assert target.text[start:end] == "never again"


def test_fully_italic_paragraph_has_no_spans(tmp_path: Path) -> None:
    b = PdfBuilder()
    b.new_page()
    for _ in range(2):
        b.para(PROSE)
    b.line("The whole letter was set in italics.", font=SERIF_ITALIC, gap=8.0)
    for _ in range(2):
        b.para(PROSE)
    book = parse_pdf(b.save(tmp_path / "letter.pdf")).book
    target = next(
        blk for c in book.chapters for blk in c.blocks if blk.text.startswith("The whole letter")
    )
    assert target.italic_spans == []


# ---------------------------------------------------------------- refusals & dispatch


def test_scanned_pdf_refused(tmp_path: Path) -> None:
    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=400, height=600)
        page.draw_rect(pymupdf.Rect(50, 50, 350, 550), fill=(0.9, 0.9, 0.9))
    path = tmp_path / "scan.pdf"
    doc.save(str(path))
    doc.close()
    with pytest.raises(IngestError, match="OCR"):
        parse_pdf(path)
    assert _MIN_TEXT_WORDS > 0  # the refusal threshold exists and is positive


def test_parse_book_dispatch_and_unknown_suffix(tmp_path: Path) -> None:
    b = PdfBuilder()
    b.new_page()
    for _ in range(3):
        b.para(PROSE)
    pdf = b.save(tmp_path / "via-dispatch.pdf")
    assert parse_book(pdf).book.chapters  # .pdf routes to parse_pdf

    stray = tmp_path / "notes.txt"
    stray.write_text("not a book")
    with pytest.raises(IngestError, match="unsupported book format"):
        parse_book(stray)


def test_idempotent_book_id(tmp_path: Path) -> None:
    b = PdfBuilder()
    b.new_page()
    for _ in range(3):
        b.para(PROSE)
    b.doc.set_metadata({"title": "Same Bytes"})
    pdf = b.save(tmp_path / "same.pdf")
    first = parse_pdf(pdf).book.book_meta.book_id
    assert parse_pdf(pdf).book.book_meta.book_id == first


def test_exclude_item_matches_chapter_title(tmp_path: Path) -> None:
    b = PdfBuilder()
    _chapter_pages(b, ["Chapter 1 Keep", "Chapter 2 Drop Me"])
    pdf = b.save(
        tmp_path / "excl.pdf",
        toc=[[1, "Chapter 1 Keep", 1], [1, "Chapter 2 Drop Me", 2]],
    )
    result = parse_pdf(pdf, exclude_items=("drop me",))
    assert [c.title for c in result.book.chapters] == ["Chapter 1 Keep"]
    assert result.skipped_items == ["Chapter 2 Drop Me"]
