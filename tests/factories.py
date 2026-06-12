"""Shared test object builders."""

from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook


def make_book() -> NormalizedBook:
    return NormalizedBook(
        book_meta=BookMeta(
            book_id="test-book-00000000",
            title="Test Book",
            source_path="test.epub",
            source_sha256="0" * 64,
        ),
        chapters=[
            Chapter(
                title="Chapter 1",
                blocks=[
                    Block(id="ch001_b0001", type=BlockType.HEADING, text="Chapter 1"),
                    Block(id="ch001_b0002", type=BlockType.PARAGRAPH, text="Hello world."),
                    Block(id="ch001_b0003", type=BlockType.SCENE_BREAK),
                    Block(id="ch001_b0004", type=BlockType.PARAGRAPH, text="After the break."),
                ],
            ),
            Chapter(
                title="Chapter 2",
                blocks=[
                    Block(id="ch002_b0001", type=BlockType.HEADING, text="Chapter 2"),
                    Block(id="ch002_b0002", type=BlockType.PARAGRAPH, text="Second chapter."),
                ],
            ),
        ],
    )
