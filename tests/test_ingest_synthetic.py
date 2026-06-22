import json
from pathlib import Path

from seiyuu.ingest import BlockType, NormalizedBook, parse_epub, write_normalized


def all_text(book: NormalizedBook) -> str:
    return " ".join(b.text for c in book.chapters for b in c.blocks)


def test_chapter_split_and_titles(synthetic_epub: Path) -> None:
    result = parse_epub(synthetic_epub)
    assert [c.title for c in result.book.chapters] == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_matter_stripping(synthetic_epub: Path) -> None:
    result = parse_epub(synthetic_epub)
    text = all_text(result.book)
    assert "COVER PAGE TEXT" not in text  # cover skipped by spine-item name
    assert "Copyright 2026" not in text  # short untitled preamble dropped
    assert "A very short bio" not in text  # short back matter dropped
    assert any("cover" in name.lower() for name in result.skipped_items)
    assert any("About the Author" in s for s in result.dropped_sections)


def test_all_three_scene_break_forms(synthetic_epub: Path) -> None:
    result = parse_epub(synthetic_epub)
    ch1 = result.book.chapters[0]
    kinds = [b.type for b in ch1.blocks]
    # heading, para, break(***), para, break(hr), para, break(blank gap), para
    assert kinds == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.SCENE_BREAK,
        BlockType.PARAGRAPH,
        BlockType.SCENE_BREAK,
        BlockType.PARAGRAPH,
        BlockType.SCENE_BREAK,
        BlockType.PARAGRAPH,
    ]
    assert all(b.text == "" for b in ch1.blocks if b.type is BlockType.SCENE_BREAK)
    assert all(not b.is_speakable for b in ch1.blocks if b.type is BlockType.SCENE_BREAK)


def test_cross_file_chapter_continuation(synthetic_epub: Path) -> None:
    result = parse_epub(synthetic_epub)
    ch2_texts = [b.text for b in result.book.chapters[1].blocks]
    assert "Chapter two begins here." in ch2_texts
    assert "Continuation paragraph that still belongs to chapter two." in ch2_texts


def test_captions_stripped(synthetic_epub: Path) -> None:
    result = parse_epub(synthetic_epub)
    assert "CAPTION TEXT MUST NOT APPEAR" not in all_text(result.book)


def test_block_ids_stable_and_ordered(synthetic_epub: Path) -> None:
    result = parse_epub(synthetic_epub)
    for ci, chapter in enumerate(result.book.chapters, start=1):
        for bi, block in enumerate(chapter.blocks, start=1):
            assert block.id == f"ch{ci:03d}_b{bi:04d}"


def test_write_and_round_trip(synthetic_epub: Path, tmp_path: Path) -> None:
    result = parse_epub(synthetic_epub)
    out_path = write_normalized(result.book, tmp_path / "books")
    assert out_path.name == "normalized.json"
    assert out_path.parent.name == result.book.book_meta.book_id
    reloaded = NormalizedBook.model_validate(json.loads(out_path.read_text(encoding="utf-8")))
    assert reloaded == result.book


def test_book_meta(synthetic_epub: Path) -> None:
    meta = parse_epub(synthetic_epub).book.book_meta
    assert meta.title == "Synthetic Test Book"
    assert meta.authors == ["Test Author"]
    assert meta.language == "en"
    assert meta.book_id.startswith("synthetic-test-book-")
    assert len(meta.source_sha256) == 64
