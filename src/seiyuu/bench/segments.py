"""Benchmark inputs: a bundled fixed text set, or real blocks from a normalized book.

The bundled set is FIXED so runs on different cards/dates are comparable: five short dialogue
lines (where per-call overhead dominates) interleaved with five long narration paragraphs (where
steady-state throughput dominates) — the mix a real audiobook render sees.
"""

from pathlib import Path

from seiyuu.bench.models import BenchSegment, SegmentKind

_D, _N = SegmentKind.DIALOGUE, SegmentKind.NARRATION

_BUNDLED: list[tuple[SegmentKind, str]] = [
    (_D, '"You\'re late," she said, without looking up from the map.'),
    (
        _N,
        "The lighthouse had not been lit in forty years, yet every evening at dusk the old keeper "
        "climbed its two hundred and twelve steps, wiped the salt from the great lens, and stood "
        "watching the horizon until the last of the fishing boats had found the harbor. Nobody in "
        "the village could remember who had first asked him to do it, and nobody had ever asked "
        "him to stop.",
    ),
    (_D, '"Then we leave at first light. Pack only what you can carry."'),
    (
        _N,
        "Rain arrived in the afternoon, first as a soft hiss against the windows and then as a "
        "steady drumming on the tin roof of the workshop. Mara set down her tools, lit the lamp, "
        "and spread the stolen blueprints across the bench. The ink had run in places, but the "
        "shape of the machine was unmistakable: a clockwork heart, no larger than a fist, built "
        "to keep time for something that was never meant to sleep.",
    ),
    (_D, '"Did you hear that?" he whispered. "Something is moving downstairs."'),
    (
        _N,
        "By the third day of the march the column had stretched to nearly a mile, and the "
        "stragglers at its tail could no longer see the banners at its head. The quartermaster "
        "counted the wagons twice each morning and came up short both times. On the fourth day "
        "he stopped counting, because the answer frightened the men more than the silence did.",
    ),
    (_D, '"I never wanted the crown. I only wanted my brother back."'),
    (
        _N,
        "The library occupied the entire east wing of the house, three stories of oak shelving "
        "joined by narrow iron galleries and spiral stairs. Dust lay thick on the upper levels, "
        "where the oldest volumes were kept, and the air there smelled of leather, candle smoke, "
        "and something faintly sweet that Elena could never quite name. She had been forbidden to "
        "climb above the second gallery since she was six years old.",
    ),
    (_D, '"Ha! Twelve thousand credits, and not a coin less, Captain."'),
    (
        _N,
        "When the storm finally broke, the valley below was unrecognizable. The river had carved "
        "a new channel through the orchard, uprooting trees that had stood for a century, and the "
        "old stone bridge lay in pieces along the bank like the bones of some enormous animal. "
        "Still, the mill wheel turned, slow and patient, as though nothing at all had happened.",
    ),
]


def bundled_segments() -> list[BenchSegment]:
    """The fixed default set (~10 segments, mixed short dialogue and long narration)."""
    return [BenchSegment(index=i, kind=kind, text=text) for i, (kind, text) in enumerate(_BUNDLED)]


_QUOTES = ('"', "“", "”")


def classify(text: str) -> SegmentKind:
    """Crude but stable: a block containing a quotation mark is dialogue-bearing."""
    return SegmentKind.DIALOGUE if any(q in text for q in _QUOTES) else SegmentKind.NARRATION


class BenchInputError(Exception):
    """The requested benchmark input (book/chapter) can't be loaded — clean, actionable."""


def book_segments(book_dir: Path, chapter: int, *, limit: int = 10) -> list[BenchSegment]:
    """Up to ``limit`` speakable blocks of 1-based ``chapter`` from ``book_dir/normalized.json``.

    Block-level text, as single-voice render synthesizes it (scene breaks skipped)."""
    from seiyuu.ingest.models import NormalizedBook

    path = Path(book_dir) / "normalized.json"
    if not path.is_file():
        raise BenchInputError(f"no normalized.json at {path}; run `seiyuu ingest` first")
    book = NormalizedBook.model_validate_json(path.read_text(encoding="utf-8"))
    if not 1 <= chapter <= len(book.chapters):
        raise BenchInputError(
            f"{book.book_meta.book_id}: no chapter {chapter} (book has {len(book.chapters)})"
        )
    if limit < 1:
        raise BenchInputError(f"--limit must be >= 1, got {limit}")
    blocks = [b for b in book.chapters[chapter - 1].blocks if b.is_speakable]
    if not blocks:
        raise BenchInputError(f"{book.book_meta.book_id}: chapter {chapter} has no speakable text")
    return [
        BenchSegment(index=i, kind=classify(b.text), text=b.text, source_id=b.id)
        for i, b in enumerate(blocks[:limit])
    ]
