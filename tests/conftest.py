from collections.abc import Iterator
from pathlib import Path

import pytest
from ebooklib import epub

from seiyuu.gpu import get_gpu_manager
from seiyuu.settings import get_settings

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test gets its own settings data_dir — and therefore its own gpu.lock.

    The get_gpu_manager() singleton arms a cross-PROCESS file lock at
    data_dir/gpu.lock. Pointed at the real data/ dir, parallel pytest-xdist
    workers (or a live API server on this machine) contend on that one OS lock
    and unrelated tests fail with GpuBusyError. Per-test isolation keeps the
    suite parallel-safe and off the real data/ directory entirely.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    get_gpu_manager.cache_clear()
    yield
    get_settings.cache_clear()
    get_gpu_manager.cache_clear()


CH12_HTML = """<html><body>
<h2>Chapter 1</h2>
<p>First paragraph of chapter one.</p>
<p>* * *</p>
<p>After the decorative break.</p>
<hr/>
<p>After the horizontal rule.</p>
<p></p>
<p></p>
<p>After the blank gap.</p>
<div class="figcenter"><img src="x.jpg"/>
<div class="caption"><p>CAPTION TEXT MUST NOT APPEAR</p></div></div>
<h2>Chapter 2</h2>
<p>Chapter two begins here.</p>
</body></html>"""

CH3_HTML = """<html><body>
<p>Continuation paragraph that still belongs to chapter two.</p>
<h2>Chapter 3</h2>
<p>Chapter three text, short and sweet.</p>
</body></html>"""


def build_synthetic_epub(path: Path, cover_image: bytes | None = None) -> Path:
    """A tiny EPUB exercising every ingest heuristic: skippable cover, short
    front/back matter, multi-chapter files, cross-file chapter continuation,
    all three scene-break forms, and caption stripping. ``cover_image`` adds a
    DECLARED cover image (EPUB3 ``cover-image`` property + EPUB2 meta)."""
    book = epub.EpubBook()
    book.set_identifier("synthetic-test-001")
    book.set_title("Synthetic Test Book")
    book.set_language("en")
    book.add_author("Test Author")
    if cover_image is not None:
        book.set_cover("cover-image.png", cover_image, create_page=False)

    def page(title: str, file_name: str, content: str) -> epub.EpubHtml:
        item = epub.EpubHtml(title=title, file_name=file_name, lang="en")
        item.set_content(content)
        book.add_item(item)
        return item

    cover = page("Cover", "cover.xhtml", "<html><body><p>COVER PAGE TEXT</p></body></html>")
    front = page(
        "Copyright",
        "front.xhtml",
        "<html><body><p>Copyright 2026 Test Author. All rights reserved.</p></body></html>",
    )
    ch12 = page("Chapters 1-2", "text1.xhtml", CH12_HTML)
    ch3 = page("Chapter 3", "text2.xhtml", CH3_HTML)
    back = page(
        "About",
        "back.xhtml",
        "<html><body><h2>About the Author</h2><p>A very short bio.</p></body></html>",
    )

    book.spine = [cover, front, ch12, ch3, back]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


@pytest.fixture
def synthetic_epub(tmp_path: Path) -> Path:
    return build_synthetic_epub(tmp_path / "synthetic.epub")


@pytest.fixture(scope="session")
def pnp_epub() -> Path:
    return FIXTURES_DIR / "PrideAndPrejudice.epub"
