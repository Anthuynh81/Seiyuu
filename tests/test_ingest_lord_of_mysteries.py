"""Structure tests against a large web-novel EPUB (1432 chapters, ~2.7M words).

Regression coverage for two bugs this fixture exposed:
- block ids must widen past 3 digits for 1000+ chapter books;
- spine-item skipping must not substring-match chapter names
  ("Azik's Discovery" -> cover, "Wrapping Up Work" -> wrap).
"""

from pathlib import Path

import pytest

from seiyuu.ingest import parse_epub

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "Lord_of_Mysteries.epub"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="large fixture not present")


@pytest.fixture(scope="module")
def lom_result():
    return parse_epub(FIXTURE)


def test_all_chapters_present(lom_result) -> None:
    chapters = lom_result.book.chapters
    assert len(chapters) == 1432
    assert chapters[0].title == "Chapter 1: Crimson"
    assert chapters[-1].title == "Chapter 1432: Bonus : That Corner (2)"


def test_no_false_positive_skips(lom_result) -> None:
    # Only the true cover page may be skipped; these real chapters were once
    # skipped by substring matching and must stay present.
    assert lom_result.skipped_items == ["Text/Cover.xhtml"]
    titles = [c.title for c in lom_result.book.chapters]
    assert "Chapter 124: Wrapping Up Work" in titles
    assert any(t.startswith("Chapter 150:") for t in titles)
    assert any(t.startswith("Chapter 607:") for t in titles)


def test_wide_block_ids(lom_result) -> None:
    last = lom_result.book.chapters[1431]
    assert last.blocks[0].id == "ch1432_b0001"


def test_slug_capped(lom_result) -> None:
    book_id = lom_result.book.book_meta.book_id
    slug, sha = book_id.rsplit("-", 1)
    assert len(slug) <= 40
    assert len(sha) == 8
