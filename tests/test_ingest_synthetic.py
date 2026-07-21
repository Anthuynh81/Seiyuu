import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from ebooklib import epub

from conftest import build_synthetic_epub
from seiyuu.ingest import (
    BlockType,
    IngestError,
    NormalizedBook,
    extract_cover_art,
    parse_epub,
    write_normalized,
)
from seiyuu.ingest.epub import _collapse, _collapse_with_italics


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


# -- embedded cover extraction --------------------------------------------------------------

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"png-payload"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"jpeg-payload"


def _epub2_meta_cover_book(path: Path, cover: bytes) -> Path:
    """EPUB2-style declared cover: a plain manifest image plus ``<meta name="cover">``
    pointing at its id — no EPUB3 ``cover-image`` property anywhere."""
    book = epub.EpubBook()
    book.set_identifier("epub2-cover-001")
    book.set_title("Epub2 Cover Book")
    book.set_language("en")
    ch = epub.EpubHtml(title="Chapter 1", file_name="c1.xhtml", lang="en")
    ch.set_content("<html><body><h2>Chapter 1</h2><p>Some chapter text.</p></body></html>")
    book.add_item(ch)
    book.add_item(
        epub.EpubImage(
            uid="cover-img", file_name="images/cover.jpg", media_type="image/jpeg", content=cover
        )
    )
    book.add_metadata(None, "meta", "", {"name": "cover", "content": "cover-img"})
    book.spine = [ch]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


def test_declared_cover_extracted_to_output_dir(tmp_path: Path) -> None:
    src = build_synthetic_epub(tmp_path / "with-cover.epub", cover_image=PNG_BYTES)
    result = parse_epub(src)
    assert result.cover == PNG_BYTES
    out = extract_cover_art(result, tmp_path / "output")
    book_dir = tmp_path / "output" / result.book.book_meta.book_id
    assert out == book_dir / "cover.png"
    assert out.read_bytes() == PNG_BYTES
    assert not (book_dir / "cover.jpg").exists()


def test_epub2_meta_cover_extracted(tmp_path: Path) -> None:
    src = _epub2_meta_cover_book(tmp_path / "epub2.epub", JPEG_BYTES)
    result = parse_epub(src)
    assert result.cover == JPEG_BYTES
    out = extract_cover_art(result, tmp_path / "output")
    assert out is not None
    assert out.name == "cover.jpg"
    assert out.read_bytes() == JPEG_BYTES


def test_no_declared_cover_is_skipped(synthetic_epub: Path, tmp_path: Path) -> None:
    # the fixture has a cover PAGE in the spine, but no declared cover IMAGE
    result = parse_epub(synthetic_epub)
    assert result.cover is None
    assert extract_cover_art(result, tmp_path / "output") is None
    assert not (tmp_path / "output" / result.book.book_meta.book_id).exists()


def test_corrupt_declared_cover_skipped_silently(tmp_path: Path) -> None:
    src = build_synthetic_epub(tmp_path / "bad-cover.epub", cover_image=b"GIF89a not jpeg/png")
    result = parse_epub(src)
    assert result.cover is not None  # declared, but fails jpeg/png validation
    assert extract_cover_art(result, tmp_path / "output") is None
    assert not (tmp_path / "output" / result.book.book_meta.book_id).exists()


def test_existing_cover_survives_reingest(tmp_path: Path) -> None:
    src = build_synthetic_epub(tmp_path / "with-cover.epub", cover_image=PNG_BYTES)
    result = parse_epub(src)
    book_dir = tmp_path / "output" / result.book.book_meta.book_id
    book_dir.mkdir(parents=True)
    user_cover = book_dir / "cover.jpg"
    user_cover.write_bytes(JPEG_BYTES)  # a user-uploaded cover already on disk wins
    assert extract_cover_art(result, tmp_path / "output") is None
    assert user_cover.read_bytes() == JPEG_BYTES
    assert not (book_dir / "cover.png").exists()


def test_inline_comment_excluded_matches_get_text() -> None:
    # The italic-aware walker must reproduce ``_collapse(el.get_text())`` EXACTLY on the
    # default thought-off path. HTML comments (and other NavigableString subclasses like
    # <script>/<style>) are excluded by get_text(); the walker must exclude them too, or
    # comment/script text leaks into narration and breaks thought-off byte identity.
    el = BeautifulSoup(
        "<p>Hello <!-- secret --> world<script>evil()</script><style>.x{}</style></p>",
        "html.parser",
    ).p
    text, spans = _collapse_with_italics(el)
    assert text == "Hello world"
    assert text == _collapse(el.get_text())
    assert spans == []


def test_declared_decompression_cap(synthetic_epub: Path, monkeypatch) -> None:
    # The upload cap bounds only the COMPRESSED file; a zip bomb must be refused from
    # the central directory's declared sizes, before ebooklib materializes members.
    monkeypatch.setattr("seiyuu.ingest.epub.MAX_DECOMPRESSED_BYTES", 64)
    with pytest.raises(IngestError, match="uncompressed"):
        parse_epub(synthetic_epub)
