"""Duration estimation + target-duration tempo math."""

from click.testing import CliRunner

from factories import make_book
from seiyuu.cli import main
from seiyuu.duration import estimate_runtime_seconds, format_hms, tempo_for_target


def test_estimate_scales_with_pace():
    book = make_book()
    base = estimate_runtime_seconds(book, wpm=150)
    assert base > 0
    assert estimate_runtime_seconds(book, wpm=75) == base * 2  # half the pace, twice the time


def test_estimate_chapter_subset_is_additive():
    book = make_book()
    whole = estimate_runtime_seconds(book, wpm=150)
    ch1 = estimate_runtime_seconds(book, wpm=150, chapters=(1,))
    ch2 = estimate_runtime_seconds(book, wpm=150, chapters=(2,))
    assert ch1 > 0 and ch2 > 0
    assert ch1 + ch2 == whole


def test_tempo_for_target_speeds_up_and_slows_down():
    assert tempo_for_target(600, 500) == 1.2  # too long -> speed up
    assert round(tempo_for_target(600, 640), 4) == round(600 / 640, 4)  # too short -> slow down


def test_tempo_for_target_is_clamped():
    assert tempo_for_target(600, 300) == 1.3  # would be 2.0
    assert tempo_for_target(600, 1200) == 0.85  # would be 0.5
    assert tempo_for_target(600, 0) == 1.0  # no target
    assert tempo_for_target(0, 600) == 1.0


def test_format_hms():
    assert format_hms(3720) == "1h 2m"
    assert format_hms(600) == "10m"
    assert format_hms(0) == "0m"


def test_estimate_cli(tmp_path):
    from seiyuu.ingest import write_normalized

    write_normalized(make_book(), tmp_path / "books")
    result = CliRunner().invoke(
        main, ["estimate", "test-book", "--books-dir", str(tmp_path / "books")]
    )
    assert result.exit_code == 0, result.output
    assert "estimated runtime" in result.output
    assert "wpm" in result.output
