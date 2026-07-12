"""UK single-quote dialogue convention: detection, guarded splitting, cache separation.

A book using ‘single curly quotes’ for all dialogue used to split to pure narration with no
warning anywhere. These tests pin the three-part fix: book-level convention detection, the
guarded single-curly splitter (apostrophes never open/close a run), and the "-sq"
prompt_version cache-key suffix that keeps pre-fix all-narration cache rows from replaying.
Double-quote behavior must stay byte-identical throughout.
"""

import json
import re
import types

import pytest
from click.testing import CliRunner

from fake_provider import FakeProvider
from seiyuu.attribute import AttributionCache, attribute_book
from seiyuu.attribute.cache import ChunkCacheKey
from seiyuu.attribute.models import CharacterRegistry, ChunkAttribution, Segment, SegmentType
from seiyuu.attribute.providers.base import (
    SINGLE_QUOTE_KEY_SUFFIX,
    AttributionLLM,
    base_prompt_version,
)
from seiyuu.attribute.providers.local import OllamaProvider
from seiyuu.attribute.spans import (
    DialogueConvention,
    detect_dialogue_convention,
    is_quoted_span,
    is_unattributed_quote,
    split_block_spans,
    thought_candidate_spans,
)
from seiyuu.cli import main
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.settings import get_settings

PROMPTS_DIR = get_settings().prompts_dir


# ---------------------------------------------------------------------------------------
# Detection heuristics
# ---------------------------------------------------------------------------------------


def _uk_text(lines: int = 25, doubles: int = 0) -> str:
    uk = " ".join(f"‘Line {i} here, don’t you think,’ said Tom." for i in range(lines))
    nested = " ".join(f'"Nested {i}," she read aloud.' for i in range(doubles))
    return f"{uk} {nested}".strip()


def test_detects_single_curly_book():
    detection = detect_dialogue_convention(_uk_text())
    assert detection.convention is DialogueConvention.SINGLE_CURLY
    assert detection.single_curly_runs >= 25 and detection.double_runs == 0


def test_single_curly_survives_nested_double_quotes():
    # UK books nest reported speech in doubles; a couple of them must not flip the verdict.
    detection = detect_dialogue_convention(_uk_text(lines=25, doubles=2))
    assert detection.convention is DialogueConvention.SINGLE_CURLY


def test_double_book_with_decorative_singles_stays_double():
    text = " ".join(f'"Line {i}," said Tom.' for i in range(50))
    text += " He liked the sign ‘fancy, decorative,’ and its ‘odd, curly,’ marks."
    assert detect_dialogue_convention(text).convention is DialogueConvention.DOUBLE


def test_detects_single_straight_book():
    text = " ".join(f"'Line {i} here, don't stop,' said Tom." for i in range(25))
    assert detect_dialogue_convention(text).convention is DialogueConvention.SINGLE_STRAIGHT


def test_no_dialogue_defaults_to_double():
    assert (
        detect_dialogue_convention("Just prose. No dialogue at all.").convention
        is DialogueConvention.DOUBLE
    )


def test_below_floor_singles_default_to_double():
    # A handful of single runs (below the absolute floor) must never flip a book.
    text = " ".join(f"‘Line {i} here,’ said Tom." for i in range(5))
    assert detect_dialogue_convention(text).convention is DialogueConvention.DOUBLE


def test_scattered_single_mix_without_doubles_is_unknown():
    curly = " ".join(f"‘Odd {i},’ he said." for i in range(12))
    straight = " ".join(f"'Odder {i},' she said." for i in range(10))
    detection = detect_dialogue_convention(f"{curly} {straight}")
    assert detection.convention is DialogueConvention.UNKNOWN


# ---------------------------------------------------------------------------------------
# Guarded single-curly splitting (adversarial apostrophes)
# ---------------------------------------------------------------------------------------

SINGLE_CASES = [
    (
        "‘Don’t worry, it’s fine,’ she said.",
        ["‘Don’t worry, it’s fine,’", " she said."],
    ),
    (
        # Possessive plural + o’clock INSIDE the quote must not close it early.
        "‘We dined at the Bennets’ house at six o’clock,’ said Jane.",
        ["‘We dined at the Bennets’ house at six o’clock,’", " said Jane."],
    ),
    (
        # Elision at the start of the quoted speech (U+2019 right after the opener).
        "‘’Tis nothing,’ said he.",
        ["‘’Tis nothing,’", " said he."],
    ),
    (
        # UK nesting: double quotes INSIDE the single-quoted run stay part of the dialogue.
        "‘She said “inner quote” to me,’ he replied.",
        ["‘She said “inner quote” to me,’", " he replied."],
    ),
    (
        # Apostrophe in the narration BEFORE the quote never opens a run.
        "It was six o’clock. ‘Late again,’ said Tom.",
        ["It was six o’clock. ", "‘Late again,’", " said Tom."],
    ),
    (
        # Possessive in narration AFTER the quote never re-opens/extends it.
        "‘Hello,’ she said. It was the Bennets’ house.",
        ["‘Hello,’", " she said. It was the Bennets’ house."],
    ),
    ("‘Where?’ ‘Here.’", ["‘Where?’", " ", "‘Here.’"]),
    ("‘Stop!’—she gasped.", ["‘Stop!’", "—she gasped."]),
    # Unclosed quote degrades to a single span (graceful, like the double pattern).
    ("‘He never finished the", ["‘He never finished the"]),
    # Pure apostrophes, no dialogue -> one prose span.
    ("Bennet’s house wasn’t far.", ["Bennet’s house wasn’t far."]),
]


@pytest.mark.parametrize("text,expected", SINGLE_CASES)
def test_single_curly_split_cases(text, expected):
    assert split_block_spans(text, DialogueConvention.SINGLE_CURLY) == expected


@pytest.mark.parametrize("text,_", SINGLE_CASES)
def test_single_curly_concatenation_reproduces_source(text, _):
    # Reconstruction-by-construction holds in the new mode too.
    assert "".join(split_block_spans(text, DialogueConvention.SINGLE_CURLY)) == text


@pytest.mark.parametrize("text,_", SINGLE_CASES)
def test_default_double_mode_leaves_single_quotes_alone(text, _):
    # Without the convention argument the split is the pre-fix double-only behavior:
    # single-curly quotes are never dialogue boundaries (only the nested-doubles case
    # contains a double-quote run).
    spans = split_block_spans(text)
    assert "".join(spans) == text
    if "“" not in text:
        assert spans == [text]


def test_straight_singles_never_switch_the_splitter():
    # SINGLE_STRAIGHT is warn-only: there is no straight-single split mode.
    text = "'Hello,' she said."
    assert split_block_spans(text, DialogueConvention.DOUBLE) == [text]


def test_thought_candidate_spans_single_curly_marks_quotes():
    text = "‘Leave now,’ said Ann. He never listens, she thought."
    start = text.index("He never")
    end = start + len("He never listens, she thought.")
    spans = thought_candidate_spans(
        "ch001_b0001", text, [(start, end)], DialogueConvention.SINGLE_CURLY
    )
    assert "".join(s.text for s in spans) == text
    head = [(s.text, s.quoted) for s in spans][:2]
    assert head == [("‘Leave now,’", True), (" said Ann. ", False)]
    # The italic run inside the PROSE region still nominates a thought candidate.
    assert any(s.candidate_id for s in spans if not s.quoted)


# ---------------------------------------------------------------------------------------
# Review surfacing: single-curly quoted spans read as quotes, apostrophes never do
# ---------------------------------------------------------------------------------------


def test_is_quoted_span_recognises_single_curly_opener():
    # Every _SINGLE_CURLY_RUN span starts with U+2018, so these must read as quotes or the
    # unattributed-quote review surfaces silently no-op for a whole UK book.
    for text in ("‘Hello,’", "‘Don’t worry, it’s fine,’", "‘’Tis nothing,’"):
        assert is_quoted_span(text), text


@pytest.mark.parametrize("text,expected", SINGLE_CASES)
def test_split_single_curly_quoted_spans_all_read_as_quotes(text, expected):
    # The splitter's quoted spans (the ones opening with U+2018) and is_quoted_span agree.
    for span in split_block_spans(text, DialogueConvention.SINGLE_CURLY):
        assert is_quoted_span(span) == span.startswith("‘"), span


def test_apostrophe_led_prose_is_not_a_quote():
    # U+2019 is the apostrophe glyph — elisions must stay prose, not become dialogue.
    assert not is_quoted_span("’tis the season")
    assert not is_quoted_span("’twas the night before")


def test_is_unattributed_quote_predicate():
    assert is_unattributed_quote(None, "‘Who goes there?’")
    assert is_unattributed_quote(None, '"Who goes there?"')
    assert not is_unattributed_quote("tom", "‘Who goes there?’")  # attributed
    assert not is_unattributed_quote(None, "He waited.")  # prose narration
    assert not is_unattributed_quote(None, "’tis the season")  # apostrophe, not a quote


# ---------------------------------------------------------------------------------------
# Prompt-version suffix normalization (cache key vs versioned behavior)
# ---------------------------------------------------------------------------------------


def test_base_prompt_version_strips_only_the_suffix():
    assert base_prompt_version("v5-sq") == "v5"
    assert base_prompt_version("v5") == "v5"
    assert base_prompt_version("v6-sq") == "v6"


def _fake_client(content: str):
    def create(**kwargs):
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice])

    completions = types.SimpleNamespace(create=create)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def test_suffixed_version_keeps_per_quote_contract():
    # "v5-sq" must load the v5 template AND keep the F1 per-quote path — the suffix can
    # never silently disable versioned behavior.
    from seiyuu.attribute.chunking import chunk_blocks

    labels = {
        "blocks": [
            {
                "block_id": "ch001_b0001",
                "quotes": [{"index": 0, "speaker": "Ann"}, {"index": 1, "speaker": "Bob"}],
            }
        ]
    }
    provider = OllamaProvider(
        model="m",
        prompts_dir=PROMPTS_DIR,
        transport="openai",
        prompt_version="v5-sq",
        client=_fake_client(json.dumps(labels)),
    )
    block = Block(
        id="ch001_b0001", type=BlockType.PARAGRAPH, text='"Hi," said Ann. "Bye," said Bob.'
    )
    chunk = chunk_blocks([block], overlap_blocks=0)[0]
    result = provider.attribute_chunk(chunk, CharacterRegistry())
    speakers = [s.speaker for s in result.segments if s.type is SegmentType.DIALOGUE]
    assert speakers == ["Ann", "Bob"]  # per-quote labels honored, not whole-block collapsed


# ---------------------------------------------------------------------------------------
# Pipeline: detection flows down, cache keys separate, notes surface
# ---------------------------------------------------------------------------------------


class SpanLabelingProvider(AttributionLLM):
    """Fake backend that exercises the REAL template path (span split included): it labels
    every block it sees in the prompt with the same speaker and lets the base class slice
    segments from source spans."""

    provider_id = "fake"
    uses_gpu = False

    def __init__(self, speaker: str = "Jane") -> None:
        super().__init__(model="fake-1.0", prompts_dir=PROMPTS_DIR, prompt_version="v1")
        self._speaker = speaker

    def _complete_json(self, prompt, schema, attempt=0):
        block_ids = re.findall(r"^\[(\w+)\]$", prompt, flags=re.MULTILINE)
        return {"blocks": [{"block_id": b, "speaker": self._speaker} for b in block_ids]}


def _book(paragraphs: list[str]) -> NormalizedBook:
    blocks = [Block(id="ch001_b0001", type=BlockType.HEADING, text="Chapter 1")]
    blocks += [
        Block(id=f"ch001_b{i:04d}", type=BlockType.PARAGRAPH, text=text)
        for i, text in enumerate(paragraphs, start=2)
    ]
    return NormalizedBook(
        book_meta=BookMeta(
            book_id="uk-book-00000000",
            title="UK Book",
            source_path="uk.epub",
            source_sha256="0" * 64,
        ),
        chapters=[Chapter(title="Chapter 1", blocks=blocks)],
    )


def _uk_book() -> NormalizedBook:
    return _book([f"‘Line {i} here, don’t you think,’ said Tom." for i in range(25)])


def _keys_in_cache(tmp_path) -> set[str]:
    import sqlite3

    with sqlite3.connect(tmp_path / "attribution.db") as conn:
        rows = conn.execute("SELECT DISTINCT prompt_version FROM attribution_chunks").fetchall()
    return {r[0] for r in rows}


def test_single_curly_book_yields_dialogue_segments(tmp_path):
    provider = SpanLabelingProvider()
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(_uk_book(), provider, cache=cache)

    assert report.flagged == []
    dialogue = [s for c in report.chapters for s in c.segments if s.type is SegmentType.DIALOGUE]
    assert len(dialogue) == 25  # every UK quote became a dialogue segment (was: zero)
    assert all(s.text.startswith("‘") and s.text.endswith("’") for s in dialogue)
    # Detection is surfaced in the report and the run provenance carries the suffixed key.
    assert any(n.startswith("dialogue convention: single curly") for n in report.registry_notes)
    assert report.prompt_version == "v1" + SINGLE_QUOTE_KEY_SUFFIX


def test_single_curly_cache_rows_are_keyed_apart(tmp_path):
    provider = SpanLabelingProvider()
    with AttributionCache(tmp_path / "attribution.db") as cache:
        attribute_book(_uk_book(), provider, cache=cache)
        assert _keys_in_cache(tmp_path) == {"v1-sq"}
        # A pre-fix all-narration row under the UNsuffixed key must not be replayed; only
        # the suffixed key hits.
        import sqlite3

        with sqlite3.connect(tmp_path / "attribution.db") as conn:
            row = conn.execute(
                "SELECT book_id, chapter_index, chunk_hash, provider_id, model_id "
                "FROM attribution_chunks"
            ).fetchone()
        base = dict(
            book_id=row[0],
            chapter_index=row[1],
            chunk_hash=row[2],
            provider_id=row[3],
            model_id=row[4],
        )
        assert cache.get(ChunkCacheKey(**base, prompt_version="v1")) is None
        assert cache.get(ChunkCacheKey(**base, prompt_version="v1-sq")) is not None


def test_second_single_curly_run_hits_the_suffixed_cache(tmp_path):
    with AttributionCache(tmp_path / "attribution.db") as cache:
        attribute_book(_uk_book(), SpanLabelingProvider(), cache=cache)

        def _never_called(chunk, registry, attempt):
            raise AssertionError("chunk was not served from cache")

        second = FakeProvider(_never_called)
        report = attribute_book(_uk_book(), second, cache=cache)
    assert second.calls == []  # deterministic re-detection -> same suffixed keys -> all cached
    assert any(n.startswith("dialogue convention:") for n in report.registry_notes)


def test_double_book_keys_and_report_are_unchanged(tmp_path):
    # Regression guard: a double-quote book keeps byte-identical cache keys (no suffix),
    # no convention note, and its dialogue still splits.
    book = _book([f'"Line {i}," said Ann.' for i in range(25)])
    provider = SpanLabelingProvider(speaker="Ann")
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(book, provider, cache=cache)
    assert _keys_in_cache(tmp_path) == {"v1"}
    assert report.prompt_version == "v1"
    assert not any(n.startswith("dialogue convention:") for n in report.registry_notes)
    dialogue = [s for c in report.chapters for s in c.segments if s.type is SegmentType.DIALOGUE]
    assert len(dialogue) == 25


def test_straight_single_book_warns_but_does_not_switch(tmp_path):
    book = _book([f"'Line {i} here, don't stop,' said Tom." for i in range(25)])
    provider = SpanLabelingProvider()
    with AttributionCache(tmp_path / "attribution.db") as cache:
        report = attribute_book(book, provider, cache=cache)
    # Warn-only: no dialogue split (too ambiguous), no cache suffix, but a loud note.
    assert _keys_in_cache(tmp_path) == {"v1"}
    assert report.prompt_version == "v1"
    note = next(n for n in report.registry_notes if n.startswith("dialogue convention:"))
    assert "dialogue may be missed" in note
    assert not any(s.type is SegmentType.DIALOGUE for c in report.chapters for s in c.segments)


def test_detection_note_reaches_progress(tmp_path):
    messages: list[str] = []
    with AttributionCache(tmp_path / "attribution.db") as cache:
        attribute_book(_uk_book(), SpanLabelingProvider(), cache=cache, progress=messages.append)
    assert any(m.startswith("dialogue convention: single curly") for m in messages)


# ---------------------------------------------------------------------------------------
# CLI surfacing
# ---------------------------------------------------------------------------------------


def test_cli_attribute_echoes_convention_note(tmp_path, monkeypatch):
    import seiyuu.attribute.providers

    def _narrate(chunk, registry, attempt):
        return ChunkAttribution(
            segments=[
                Segment(block_id=b.id, type=SegmentType.NARRATION, text=b.text)
                for b in chunk.owned_blocks
            ]
        )

    monkeypatch.setattr(
        seiyuu.attribute.providers, "get_provider", lambda *a, **k: FakeProvider(_narrate)
    )
    book = _uk_book()
    book_dir = tmp_path / "books" / book.book_meta.book_id
    book_dir.mkdir(parents=True)
    (book_dir / "normalized.json").write_text(book.model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(
        main, ["attribute", "uk-book", "--books-dir", str(tmp_path / "books")]
    )
    assert result.exit_code == 0, result.output
    assert "dialogue convention: single curly" in result.output
