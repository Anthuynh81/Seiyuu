"""CLI: `seiyuu attribute` writes attribution.json; `seiyuu characters` reports it.

The provider is faked (no live LLM); the pipeline, cache, and report I/O are real.
"""

import pytest
from click.testing import CliRunner

from factories import make_book
from fake_provider import FakeProvider
from seiyuu.attribute.models import (
    CharacterMention,
    ChunkAttribution,
    Segment,
    SegmentType,
)
from seiyuu.cli import main


def _alice(chunk, registry, attempt):
    return ChunkAttribution(
        segments=[
            Segment(block_id=b.id, type=SegmentType.DIALOGUE, text=b.text, speaker="Alice")
            for b in chunk.owned_blocks
        ],
        characters=[CharacterMention(name="Alice", gender="female")],
    )


@pytest.fixture
def books_dir(tmp_path):
    book = make_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    (book_dir / "normalized.json").write_text(book.model_dump_json(), encoding="utf-8")
    return tmp_path / "books"


@pytest.fixture
def fake_provider(monkeypatch):
    import seiyuu.attribute.providers

    provider = FakeProvider(_alice)
    monkeypatch.setattr(seiyuu.attribute.providers, "get_provider", lambda *a, **k: provider)
    return provider


def test_attribute_writes_report(books_dir, fake_provider):
    result = CliRunner().invoke(main, ["attribute", "test-book", "--books-dir", str(books_dir)])
    assert result.exit_code == 0, result.output
    assert "1 characters" in result.output
    report = next(books_dir.glob("*/attribution.json"))
    assert report.is_file()
    assert (report.parent / "attribution.db").is_file()


def test_characters_report_lists_speaker_and_samples(books_dir, fake_provider):
    runner = CliRunner()
    assert (
        runner.invoke(main, ["attribute", "test-book", "--books-dir", str(books_dir)]).exit_code
        == 0
    )
    result = runner.invoke(main, ["characters", "test-book", "--books-dir", str(books_dir)])
    assert result.exit_code == 0, result.output
    assert "Alice [alice]" in result.output
    assert "Hello world." in result.output  # a sample dialogue line
    assert "narration segments:" in result.output


def test_attribute_unknown_book_is_actionable(tmp_path, fake_provider):
    result = CliRunner().invoke(main, ["attribute", "nope", "--books-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "seiyuu ingest" in result.output


def test_attribute_cross_process_gpu_busy_is_a_clean_error(
    tmp_path, books_dir, fake_provider, monkeypatch
):
    # Another PROCESS holds gpu.lock (a second manager instance — the OS lock is
    # per-descriptor): the CLI must print the actionable refusal, not a traceback.
    import seiyuu.services.attribution as attribution_service
    from seiyuu.gpu import GpuBusyError, GpuResourceManager

    class _Consumer:
        def unload(self) -> None: ...

    lock = tmp_path / "contended-gpu.lock"
    holder = GpuResourceManager(lock_path=lock)
    with holder.acquire(_Consumer(), "engine:kokoro"):
        pass  # lazy residency keeps the card claimed (simulates the API server process)
    monkeypatch.setattr(
        attribution_service, "get_gpu_manager", lambda: GpuResourceManager(lock_path=lock)
    )

    result = CliRunner().invoke(main, ["attribute", "test-book", "--books-dir", str(books_dir)])
    assert result.exit_code != 0
    assert "another seiyuu process holds the GPU" in result.output  # ClickException, printed
    assert not isinstance(result.exception, GpuBusyError)  # no raw traceback
