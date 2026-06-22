"""Snapshot/structure tests against the real Project Gutenberg fixture.

Known edition quirk, pinned deliberately: chapter 1 is the title page +
Saintsbury preface (~4300 words). Its section headings are images in this
edition, so it cannot be split or classified further; the 61 novel chapters
follow as chapters 2-62.
"""

import json
from pathlib import Path

import pytest

from seiyuu.ingest import BlockType, parse_epub

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def pnp_result(pnp_epub: Path):
    return parse_epub(pnp_epub)


def test_chapter_structure_matches_snapshot(pnp_result) -> None:
    snapshot = json.loads((FIXTURES_DIR / "pnp_summary.json").read_text(encoding="utf-8"))
    book = pnp_result.book
    assert book.book_meta.book_id == snapshot["book_id"]
    actual = [
        {
            "title": c.title,
            "blocks": len(c.blocks),
            "words": sum(len(b.text.split()) for b in c.blocks),
        }
        for c in book.chapters
    ]
    assert actual == snapshot["chapters"]


def test_sixty_one_novel_chapters_plus_front_section(pnp_result) -> None:
    chapters = pnp_result.book.chapters
    assert len(chapters) == 62
    assert chapters[0].title == "PRIDE. and PREJUDICE"  # title page + preface
    assert chapters[1].title == "Chapter I."
    assert chapters[-1].title == "CHAPTER LXI."


def test_famous_first_line(pnp_result) -> None:
    ch1 = pnp_result.book.chapters[1]  # "Chapter I."
    first_para = next(b for b in ch1.blocks if b.type is BlockType.PARAGRAPH)
    # this edition renders the drop-cap first word as "IT" — compare case-insensitively
    assert "it is a truth universally acknowledged" in first_para.text.lower()


def test_no_boilerplate_or_captions_leak(pnp_result) -> None:
    text = " ".join(b.text for c in pnp_result.book.chapters for b in c.blocks)
    assert "Project Gutenberg" not in text
    assert "He rode a black horse" not in text  # illustration caption


def test_metadata(pnp_result) -> None:
    meta = pnp_result.book.book_meta
    assert meta.title == "Pride and Prejudice"
    assert meta.authors == ["Jane Austen"]
    assert meta.language == "en"


def test_block_ids_ordered(pnp_result) -> None:
    for ci, chapter in enumerate(pnp_result.book.chapters, start=1):
        for bi, block in enumerate(chapter.blocks, start=1):
            assert block.id == f"ch{ci:03d}_b{bi:04d}"
