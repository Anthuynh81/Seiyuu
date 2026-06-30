"""Duration estimation + target-duration tempo (M4).

A pre-render runtime estimate comes from speakable word count and a narration pace (wpm). The
target-duration mode nudges the FINAL audiobook toward a requested length with a single pitch-
preserving `atempo`, clamped to a sane range so speech never sounds chipmunked or dragged.
"""

DEFAULT_WPM = 150.0  # typical audiobook narration pace
TEMPO_MIN = 0.85
TEMPO_MAX = 1.3


def estimate_runtime_seconds(
    book, *, wpm: float = DEFAULT_WPM, chapters: tuple[int, ...] = ()
) -> float:
    """Estimated narration seconds for the book (or a 1-based chapter subset) at `wpm`."""
    wanted = set(chapters)
    words = sum(
        len(block.text.split())
        for ci, chapter in enumerate(book.chapters, start=1)
        for block in chapter.blocks
        if block.is_speakable and (not wanted or ci in wanted)
    )
    return words / wpm * 60 if wpm > 0 else 0.0


def tempo_for_target(
    actual_seconds: float,
    target_seconds: float,
    *,
    lo: float = TEMPO_MIN,
    hi: float = TEMPO_MAX,
) -> float:
    """atempo factor to move `actual` toward `target`, clamped to [lo, hi] (1.0 == no change).

    >1 speeds up (shorter); <1 slows down. Clamped so the result stays natural even when the
    requested target is far from what was rendered.
    """
    if target_seconds <= 0 or actual_seconds <= 0:
        return 1.0
    return max(lo, min(hi, actual_seconds / target_seconds))


def format_hms(seconds: float) -> str:
    """Human runtime like '8h 12m' or '47m'."""
    total = int(round(seconds))
    hours, minutes = total // 3600, (total % 3600) // 60
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"
