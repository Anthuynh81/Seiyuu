"""CLI: `seiyuu validate` reports whisper validation results recorded in a render manifest."""

from click.testing import CliRunner

from seiyuu.cli import main
from seiyuu.ingest.models import BlockType
from seiyuu.render import RenderedChapter, RenderedSegment, RenderManifest
from seiyuu.validate import ValidationResult


def _write_manifest(out_dir, *segments):
    book_dir = out_dir / "test-book-00000000"
    book_dir.mkdir(parents=True, exist_ok=True)
    manifest = RenderManifest(
        book_id="test-book-00000000",
        book_title="V",
        chapters=[RenderedChapter(index=1, title="C1", segments=list(segments))],
        validation_failures=sum(1 for s in segments if s.validation and not s.validation.ok),
    )
    (book_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")


def test_validate_reports_failures(tmp_path):
    out = tmp_path / "out"
    ok = RenderedSegment(
        block_id="ch001_b0001", type=BlockType.PARAGRAPH, wav="cache/a.wav", voice_id="v",
        validation=ValidationResult(ok=True, score=0.95, transcript="hi", expected="hi"),
    )  # fmt: skip
    bad = RenderedSegment(
        block_id="ch001_b0002", type=BlockType.PARAGRAPH, wav="cache/b.wav", voice_id="v",
        validation=ValidationResult(ok=False, score=0.3, transcript="x", expected="hello there"),
    )  # fmt: skip
    _write_manifest(out, ok, bad)
    result = CliRunner().invoke(main, ["validate", "test-book", "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert "validated segments: 2, failures: 1" in result.output
    assert "ch001_b0002" in result.output and "FAIL" in result.output
    assert "hello there" in result.output  # expected text for the failure
    assert "ch001_b0001" not in result.output  # passing segment hidden by default


def test_validate_all_lists_passing_segments(tmp_path):
    out = tmp_path / "out"
    ok = RenderedSegment(
        block_id="ch001_b0001", type=BlockType.PARAGRAPH, wav="cache/a.wav", voice_id="v",
        validation=ValidationResult(ok=True, score=0.95, transcript="hi", expected="hi"),
    )  # fmt: skip
    _write_manifest(out, ok)
    result = CliRunner().invoke(main, ["validate", "test-book", "--all", "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert "ch001_b0001" in result.output


def test_validate_no_validation_data(tmp_path):
    out = tmp_path / "out"
    seg = RenderedSegment(
        block_id="ch001_b0001", type=BlockType.PARAGRAPH, wav="cache/a.wav", voice_id="v"
    )
    _write_manifest(out, seg)
    result = CliRunner().invoke(main, ["validate", "test-book", "--output-dir", str(out)])
    assert result.exit_code == 0
    assert "no validated segments" in result.output
