"""Attribution accuracy harness: pure scoring, gold export (with a Review edit applied), and
an end-to-end `seiyuu eval run` over FakeProvider arms that must never touch the book's own
attribution files. No live LLM: every arm is a scripted FakeProvider."""

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import seiyuu.services.attribution as attribution_svc
from fake_provider import FakeProvider
from seiyuu.attribute.evaluate import (
    UNATTRIBUTED,
    GoldQuote,
    QuoteType,
    align_quote,
    dump_gold,
    export_gold,
    gold_conflicts,
    gold_speakers,
    load_gold,
    match_strength,
    normalize_name,
    score_report,
)
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterMention,
    CharacterRegistry,
    ChunkAttribution,
    Segment,
    SegmentType,
)
from seiyuu.attribute.providers.base import AttributionError, MalformedOutputError
from seiyuu.attribute.providers.local import OllamaModelMissingError, OllamaPreflightError
from seiyuu.attribute.spans import quoted_ordinals, thought_candidate_spans
from seiyuu.cli import main
from seiyuu.cli.evaluation import parse_chapters
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.services import ReassignSegment, record_edit, run_attribution
from seiyuu.services.common import ServiceError
from seiyuu.services.evaluation import (
    ESCALATED_MARKER,
    EvalReport,
    build_stats,
    preflight_models,
    render_markdown,
    run_arm,
    run_eval,
)
from seiyuu.settings import get_settings

BOOK_ID = "eval-book-00000000"

# ------------------------------------------------------------------ scoring helpers


def _char(cid: str, name: str, *aliases: str) -> Character:
    return Character(id=cid, canonical_name=name, aliases=list(aliases))


def _dlg(block_id: str, text: str, speaker: str) -> Segment:
    return Segment(block_id=block_id, type=SegmentType.DIALOGUE, speaker=speaker, text=text)


def _nar(block_id: str, text: str) -> Segment:
    return Segment(block_id=block_id, type=SegmentType.NARRATION, text=text)


def _report(chars: list[Character], segments_by_chapter: dict[int, list[Segment]]):
    return AttributionReport(
        book_id=BOOK_ID,
        provider_id="fake",
        model_id="m",
        prompt_version="v1",
        registry=CharacterRegistry(characters=chars),
        chapters=[
            AttributedChapter(index=i, title=f"Ch {i}", segments=segs)
            for i, segs in sorted(segments_by_chapter.items())
        ],
    )


def _gold(block_id: str, idx: int, text: str, speaker: str, *aliases, ch=1, qtype=None):
    return GoldQuote(
        book_id=BOOK_ID,
        chapter_index=ch,
        block_id=block_id,
        quote_index=idx,
        text=text,
        speaker=speaker,
        aliases=list(aliases),
        quote_type=qtype,
    )


Q1, Q2, Q3 = '"Hello."', '"Goodbye."', '"Again."'

# ---------------------------------------------------------------------- name matching


def test_normalize_name_folds_case_and_punctuation():
    assert normalize_name("Mr. Darcy") == normalize_name("mr darcy") == "mr darcy"
    assert normalize_name("  O’Brien ") == "o'brien"


@pytest.mark.parametrize(
    ("pred", "gold", "strength"),
    [
        ({"mr darcy"}, {"mr darcy"}, 2),
        ({"darcy"}, {"mr darcy"}, 1),  # honorific stripped
        ({"dr watson"}, {"watson"}, 1),  # neutral title
        ({"mrs bennet"}, {"mr bennet"}, 0),  # gendered titles must agree
        ({"miss bennet"}, {"mrs bennet"}, 0),  # generation/marital guard
        ({"lizzy"}, {"elizabeth bennet", "lizzy"}, 2),  # gold alias
        ({"jane"}, {"elizabeth"}, 0),
        ({"mr"}, {"mrs"}, 0),  # a lone title is not stripped to ""
    ],
)
def test_match_strength(pred, gold, strength):
    assert match_strength(pred, gold) == strength


# ---------------------------------------------------------------------------- scoring


def test_exact_alias_and_honorific_matches_are_correct():
    b1, b2, b3 = "ch001_b0001", "ch001_b0002", "ch001_b0003"
    report = _report(
        [_char("lizzy", "Lizzy"), _char("darcy", "Darcy"), _char("jane", "Jane Bennet")],
        {1: [_dlg(b1, Q1, "lizzy"), _dlg(b2, Q2, "darcy"), _dlg(b3, Q3, "jane")]},
    )
    gold = [
        _gold(b1, 0, Q1, "Elizabeth Bennet", "Lizzy"),
        _gold(b2, 0, Q2, "Mr. Darcy"),
        _gold(b3, 0, Q3, "Jane Bennet"),
    ]
    score = score_report(gold, report)
    assert (score.overall.n, score.overall.correct, score.overall.accuracy) == (3, 3, 1.0)
    assert score.confusions == [] and score.alias_drift == 0


def test_predicted_registry_alias_counts():
    """The ARM's registry aliases count too, not just its canonical name."""
    b1 = "ch001_b0001"
    report = _report([_char("x", "The Lady", "Elizabeth")], {1: [_dlg(b1, Q1, "x")]})
    assert score_report([_gold(b1, 0, Q1, "Elizabeth")], report).overall.correct == 1


def test_unattributed_quote_is_wrong_and_counted():
    b1 = "ch001_b0001"
    report = _report([], {1: [_nar(b1, Q1)]})
    score = score_report([_gold(b1, 0, Q1, "Jane")], report)
    assert score.overall.correct == 0 and score.unattributed == 1
    assert score.confusions[0].predicted == UNATTRIBUTED
    assert score.outcomes[0].predicted_character_id is None


def test_ambiguous_surname_scores_wrong():
    """Bare 'Bennet' honorific-matches BOTH Mr. and Mrs. Bennet: ambiguous, never a guess."""
    b1, b2 = "ch001_b0001", "ch001_b0002"
    report = _report(
        [_char("bennet", "Bennet")], {1: [_dlg(b1, Q1, "bennet"), _dlg(b2, Q2, "bennet")]}
    )
    gold = [_gold(b1, 0, Q1, "Mr. Bennet"), _gold(b2, 0, Q2, "Mrs. Bennet")]
    score = score_report(gold, report)
    assert score.overall.correct == 0
    assert {c.predicted for c in score.confusions} == {"Bennet (ambiguous)"}
    mapping = score.mappings[0]
    assert mapping.gold_key is None and mapping.ambiguous_with == ["Mr. Bennet", "Mrs. Bennet"]


def test_exact_beats_honorific_match():
    """A character carrying 'Mr. Bennet' exactly maps there even though its alias 'Bennet'
    would honorific-match Mrs. Bennet too."""
    b1 = "ch001_b0001"
    report = _report([_char("mrb", "Mr. Bennet", "Bennet")], {1: [_dlg(b1, Q1, "mrb")]})
    gold = [_gold(b1, 0, Q1, "Mr. Bennet"), _gold("ch001_b0009", 0, Q2, "Mrs. Bennet")]
    score = score_report(gold, report)
    assert score.outcomes[0].correct
    assert score.unaligned == 1  # the second gold block has no segment


def test_wrong_gendered_title_is_not_a_match():
    b1 = "ch001_b0001"
    report = _report([_char("mrsb", "Mrs. Bennet")], {1: [_dlg(b1, Q1, "mrsb")]})
    score = score_report([_gold(b1, 0, Q1, "Mr. Bennet")], report)
    assert score.overall.correct == 0
    assert score.confusions[0].predicted == "Mrs. Bennet (unmapped)"


def test_alias_drift_counts_distinct_predicted_characters():
    b1, b2, b3 = "ch001_b0001", "ch001_b0002", "ch001_b0003"
    report = _report(
        [
            _char("darcy", "Mr. Darcy"),
            _char("fitz", "Fitzwilliam Darcy"),
            _char("d3", "Darcy"),
        ],
        {1: [_dlg(b1, Q1, "darcy"), _dlg(b2, Q2, "fitz"), _dlg(b3, Q3, "d3")]},
    )
    gold = [
        _gold(b1, 0, Q1, "Mr. Darcy", "Fitzwilliam Darcy"),
        _gold(b2, 0, Q2, "Mr. Darcy"),
        _gold(b3, 0, Q3, "Mr. Darcy"),
    ]
    score = score_report(gold, report)
    assert score.overall.correct == 3  # every name is right ...
    assert score.alias_drift == 2  # ... but one speaker is split across three characters
    assert score.drift[0].gold_speaker == "Mr. Darcy"
    assert len(score.drift[0].predicted_characters) == 3


def test_quote_type_and_chapter_breakdown():
    a, b, c = "ch001_b0001", "ch001_b0002", "ch002_b0001"
    report = _report(
        [_char("jane", "Jane"), _char("liz", "Elizabeth")],
        {1: [_dlg(a, Q1, "jane"), _dlg(b, Q2, "jane")], 2: [_dlg(c, Q3, "liz")]},
    )
    gold = [
        _gold(a, 0, Q1, "Jane", qtype="Explicit"),  # PDNC capitalisation folds
        _gold(b, 0, Q2, "Elizabeth", qtype=QuoteType.IMPLICIT),
        _gold(c, 0, Q3, "Elizabeth", ch=2),
    ]
    score = score_report(gold, report)
    assert score.by_quote_type["explicit"].accuracy == 1.0
    assert score.by_quote_type["implicit"].accuracy == 0.0
    assert score.by_quote_type["unlabeled"].n == 1
    assert score.by_chapter[1].accuracy == 0.5 and score.by_chapter[2].accuracy == 1.0
    assert score.by_speaker["Elizabeth"].n == 2
    assert score.confusions[0].gold == "Elizabeth" and score.confusions[0].predicted == "Jane"


def test_confusions_ranked_by_frequency():
    blocks = [f"ch001_b{i:04d}" for i in range(1, 5)]
    report = _report(
        [_char("j", "Jane"), _char("l", "Lydia")],
        {
            1: [
                _dlg(blocks[0], Q1, "j"),
                _dlg(blocks[1], Q1, "j"),
                _dlg(blocks[2], Q1, "l"),
                _nar(blocks[3], Q1),
            ]
        },
    )
    gold = [_gold(b, 0, Q1, "Kitty") for b in blocks]
    score = score_report(gold, report, top_confusions=2)
    assert [(c.predicted, c.count) for c in score.confusions] == [
        ("Jane (unmapped)", 2),
        (UNATTRIBUTED, 1),
    ]


def test_multi_quote_block_scores_each_quote():
    b = "ch001_b0001"
    segs = [_dlg(b, Q1, "j"), _nar(b, " she said; then "), _dlg(b, Q2, "l")]
    report = _report([_char("j", "Jane"), _char("l", "Lydia")], {1: segs})
    gold = [_gold(b, 0, Q1, "Jane"), _gold(b, 1, Q2, "Jane")]
    score = score_report(gold, report)
    assert [o.correct for o in score.outcomes] == [True, False]


def test_align_quote_falls_back_to_text_when_index_is_stale():
    b = "ch001_b0001"
    segs = [_dlg(b, Q1, "j"), _dlg(b, Q2, "l")]
    assert align_quote(segs, 1, Q2) is segs[1]  # fast path
    assert align_quote(segs, 0, Q2) is segs[1]  # index stale, text realigns
    assert align_quote(segs, 0, '"Never said."') is None
    fallback = [_nar(b, f"{Q1} she said. {Q2}")]  # a flagged block: one narration segment
    assert align_quote(fallback, 1, Q2) is fallback[0]


# ------------------------------------------------------------------------- gold I/O


def test_load_gold_round_trip_and_errors(tmp_path):
    rows = [_gold("ch001_b0001", 0, Q1, "Jane", "Janie", qtype="anaphoric")]
    path = tmp_path / "gold.jsonl"
    path.write_text(dump_gold(rows), encoding="utf-8")
    assert load_gold(path) == rows

    path.write_text(dump_gold(rows) + '{"book_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"gold.jsonl:2"):
        load_gold(path)
    path.write_text(dump_gold(rows * 2), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_gold(path)


def test_parse_chapters():
    assert parse_chapters("1-3,5") == (1, 2, 3, 5)
    assert parse_chapters(None) == ()
    with pytest.raises(Exception, match="bad chapter range"):
        parse_chapters("3-1")


def test_build_stats_aggregates_chunks():
    from seiyuu.services.evaluation import CallRecord

    calls = [
        CallRecord(chapter_index=1, chunk_index=0, role="primary", attempt=0, seconds=2.0,
                   outcome="schema_failure"),
        CallRecord(chapter_index=1, chunk_index=0, role="primary", attempt=1, seconds=1.0,
                   outcome="ok"),
        CallRecord(chapter_index=2, chunk_index=0, role="primary", attempt=0, seconds=1.0,
                   outcome="ok"),
    ]  # fmt: skip
    stats = build_stats(calls, wall_seconds=5, cache_hits=1, fast_path=2, flagged_blocks=0)
    assert stats.llm_chunks == 2 and stats.llm_calls == 3
    assert stats.schema_failures == 1 and stats.local_retries == 1
    assert stats.seconds_per_chunk == 2.0 and stats.max_seconds_per_chunk == 3.0
    assert stats.cached_chunks == 1 and stats.fast_path_chunks == 2


# ------------------------------------------------------- tiny book with real dialogue


def _p(cid: int, bid: int, text: str) -> Block:
    return Block(id=f"ch{cid:03d}_b{bid:04d}", type=BlockType.PARAGRAPH, text=text)


def _book() -> NormalizedBook:
    return NormalizedBook(
        book_meta=BookMeta(
            book_id=BOOK_ID, title="Eval Book", source_path="e.epub", source_sha256="0" * 64
        ),
        chapters=[
            Chapter(
                title="Chapter 1",
                blocks=[
                    Block(id="ch001_b0001", type=BlockType.HEADING, text="Chapter 1"),
                    _p(1, 2, '"I am glad," said Elizabeth. "Very glad."'),
                    _p(1, 3, '"Indeed," replied Mr. Darcy.'),
                    _p(1, 4, "The evening passed quietly."),
                ],
            ),
            Chapter(
                title="Chapter 2",
                blocks=[
                    Block(id="ch002_b0001", type=BlockType.HEADING, text="Chapter 2"),
                    _p(2, 2, '"Who is there?" asked Jane.'),
                    _p(2, 3, '"It is I," said Mr. Bingley. "Charles."'),
                ],
            ),
        ],
    )


TRUTH = {
    ("ch001_b0002", 0): "Elizabeth",
    ("ch001_b0002", 1): "Elizabeth",
    ("ch001_b0003", 0): "Mr. Darcy",
    ("ch002_b0002", 0): "Jane",
    ("ch002_b0003", 0): "Mr. Bingley",
    ("ch002_b0003", 1): "Mr. Bingley",
}


def _scripted(labels: dict[tuple[str, int], str | None], *, malformed_first: bool = False):
    """A FakeProvider script that slices the SOURCE spans like a real provider and labels
    each quote from ``labels`` (None/missing -> unattributed narration)."""

    def script(chunk, registry, attempt) -> ChunkAttribution:
        if malformed_first and attempt == 0:
            raise MalformedOutputError("scripted bad JSON")
        segments: list[Segment] = []
        names: set[str] = set()
        for block in chunk.owned_blocks:
            spans = thought_candidate_spans(block.id, block.text, [])
            ordinal = {s: o for o, s in quoted_ordinals(spans)}
            for span in spans:
                speaker = labels.get((block.id, ordinal[span])) if span.quoted else None
                if speaker:
                    names.add(speaker)
                    segments.append(_dlg(block.id, span.text, speaker))
                else:
                    segments.append(_nar(block.id, span.text))
        return ChunkAttribution(
            segments=segments, characters=[CharacterMention(name=n) for n in sorted(names)]
        )

    return script


BAD = TRUTH | {
    ("ch001_b0003", 0): "Darcy",  # honorific variant: still correct
    ("ch002_b0002", 0): "Elizabeth",  # wrong speaker
    ("ch002_b0003", 1): None,  # unattributed
}
ARMS = {"good:1b": TRUTH, "bad:1b": BAD}


# The "raw" model mislabels Bingley's second quote as Jane (fixed below by a Review edit)
# and leaves Darcy's line unattributed (skipped by the export: no gold speaker to record).
RAW = TRUTH | {("ch002_b0003", 1): "Jane", ("ch001_b0003", 0): None}


@pytest.fixture
def attributed_book(tmp_path) -> tuple[Path, Path]:
    """books_dir with the tiny book attributed (FakeProvider) + one Review edit applied."""
    books = tmp_path / "books"
    book = _book()
    book_dir = books / BOOK_ID
    book_dir.mkdir(parents=True)
    (book_dir / "normalized.json").write_text(book.model_dump_json(), encoding="utf-8")
    report = run_attribution(
        book, book_dir, cfg=get_settings(), provider=FakeProvider(_scripted(RAW), model="raw")
    )
    bingley = next(c.id for c in report.registry.characters if c.canonical_name == "Mr. Bingley")
    # segments of ch002_b0003: '"It is I,"' (0), ' said Mr. Bingley. ' (1), '"Charles."' (2)
    record_edit(book_dir, ReassignSegment(block_id="ch002_b0003", segment_index=2, speaker=bingley))
    return books, book_dir


def _snapshot(directory: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


def _arm_provider(monkeypatch, arms: dict[str, dict]) -> list[str]:
    """Route build_provider to a scripted FakeProvider per model id; returns built models."""
    built: list[str] = []

    def fake_build(cfg, provider_id, model, prompt_version, *, emit_thoughts=False):
        built.append(model)
        return FakeProvider(_scripted(arms[model]), model=model)

    monkeypatch.setattr(attribution_svc, "build_provider", fake_build)
    return built


def test_export_gold_applies_review_edits(attributed_book):
    _, book_dir = attributed_book
    report, warnings = attribution_svc.load_report(book_dir)
    assert warnings == []
    result = export_gold(_book(), report)
    got = {(r.block_id, r.quote_index): r.speaker for r in result.rows}
    assert got == {k: v for k, v in TRUTH.items() if k != ("ch001_b0003", 0)}
    assert result.skipped_unattributed == 1 and result.skipped_unaligned == 0
    row = next(r for r in result.rows if (r.block_id, r.quote_index) == ("ch002_b0003", 1))
    block_text = _book().chapters[1].blocks[2].text
    assert block_text[row.char_start : row.char_end] == row.text == '"Charles."'
    only2 = export_gold(_book(), report, (2,))
    assert {r.chapter_index for r in only2.rows} == {2}
    with pytest.raises(ValueError, match="no such chapter"):
        export_gold(_book(), report, (9,))


def test_cli_end_to_end_never_touches_book_files(attributed_book, tmp_path, monkeypatch):
    books, book_dir = attributed_book
    before = _snapshot(book_dir)
    assert {"attribution.json", "attribution.db", "edits.json"} <= before.keys()
    runner = CliRunner()
    gold_path = tmp_path / "gold.jsonl"
    res = runner.invoke(
        main,
        ["eval", "gold-export", BOOK_ID, "--chapters", "1-2", "-o", str(gold_path),
         "--books-dir", str(books)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "gold: 5 quotes" in res.output and "skipped 1 unattributed" in res.output
    assert len(load_gold(gold_path)) == 5

    built = _arm_provider(monkeypatch, ARMS)
    out = tmp_path / "out"
    args = ["eval", "run", BOOK_ID, "--gold", str(gold_path), "--models", "good:1b,bad:1b",
            "-o", str(out), "--books-dir", str(books)]  # fmt: skip
    res = runner.invoke(main, args)
    assert res.exit_code == 0, res.output
    assert built == ["good:1b", "bad:1b"]

    data = json.loads((out / "eval_report.json").read_text(encoding="utf-8"))
    arms = {a["model_id"]: a for a in data["arms"]}
    assert data["gold_quotes"] == 5 and data["chapters"] == [1, 2]
    assert arms["good:1b"]["score"]["overall"] == {"n": 5, "correct": 5, "accuracy": 1.0}
    bad = arms["bad:1b"]["score"]
    # Jane->Elizabeth wrong, "Charles." unattributed; Darcy isn't in gold (raw left it narration)
    assert bad["overall"]["correct"] == 3 and bad["unattributed"] == 1
    assert {(c["gold"], c["predicted"]) for c in bad["confusions"]} == {
        ("Jane", "Elizabeth"),
        ("Mr. Bingley", UNATTRIBUTED),
    }
    assert "outcomes" not in bad  # the per-quote detail lives in arm-*/score.json
    stats = arms["good:1b"]["stats"]
    assert stats["llm_chunks"] >= 1 and stats["seconds_per_chunk"] is not None
    assert stats["schema_failures"] == 0 and stats["escalations"] == 0

    md = (out / "eval_report.md").read_text(encoding="utf-8")
    assert "| fake/good:1b | 100.0% (5/5)" in md and "Jane → Elizabeth: 1" in md
    assert "good:1b" in res.output and "acc" in res.output
    for slug in ("good_1b", "bad_1b"):
        arm_dir = out / f"arm-{slug}"
        assert {"attribution.json", "attribution.db", "run_stats.json", "score.json"} <= {
            p.name for p in arm_dir.iterdir()
        }
    # THE isolation guarantee: the book's own attribution/edits/cache are byte-identical.
    assert _snapshot(book_dir) == before

    # A re-run is served from each ARM's own cache (no model call, no timing).
    res = runner.invoke(main, args)
    assert res.exit_code == 0, res.output
    data = json.loads((out / "eval_report.json").read_text(encoding="utf-8"))
    assert all(a["stats"]["llm_chunks"] == 0 for a in data["arms"])
    assert all(a["stats"]["cached_chunks"] >= 1 for a in data["arms"])
    # --fresh wipes the arm cache so timings are real again.
    res = runner.invoke(main, [*args, "--fresh"])
    assert res.exit_code == 0, res.output
    data = json.loads((out / "eval_report.json").read_text(encoding="utf-8"))
    assert all(a["stats"]["llm_chunks"] >= 1 for a in data["arms"])
    assert _snapshot(book_dir) == before


def test_run_arm_records_schema_failures_retries_and_escalation(attributed_book, tmp_path):
    _, book_dir = attributed_book
    gold = export_gold(_book(), attribution_svc.load_report(book_dir)[0]).rows
    cfg = get_settings()
    flaky = run_arm(
        _book(), gold, cfg=cfg, model="flaky", out_dir=tmp_path / "o", chapters=(1, 2),
        provider=FakeProvider(_scripted(TRUTH, malformed_first=True), model="flaky"),
    )  # fmt: skip
    assert flaky.stats.schema_failures == flaky.stats.llm_chunks >= 1
    assert flaky.stats.local_retries == flaky.stats.schema_failures
    assert flaky.score.overall.accuracy == 1.0

    def always_bad(chunk, registry, attempt):
        raise MalformedOutputError("never valid")

    escalated = run_arm(
        _book(), gold, cfg=cfg, model="weak", out_dir=tmp_path / "o", chapters=(1, 2),
        provider=FakeProvider(always_bad, model="weak"),
        escalation_provider=FakeProvider(_scripted(TRUTH), model="strong"),
    )  # fmt: skip
    assert escalated.stats.escalations == escalated.stats.llm_chunks >= 1
    assert escalated.stats.flagged_blocks == 0
    assert escalated.score.overall.accuracy == 1.0

    flagged = run_arm(
        _book(), gold, cfg=cfg, model="broken", out_dir=tmp_path / "o", chapters=(1, 2),
        provider=FakeProvider(always_bad, model="broken"),
    )  # fmt: skip
    assert flagged.stats.flagged_blocks >= 1
    assert flagged.score.overall.correct == 0 and flagged.score.unattributed == len(gold)


def test_preflight_names_every_missing_model(monkeypatch):
    def fake_preflight(base_url, models):
        if models[0] != "qwen2.5:7b":
            raise OllamaModelMissingError("missing", base_url=base_url, model=models[0])

    monkeypatch.setattr(attribution_svc, "ollama_preflight", fake_preflight)
    cfg = get_settings()
    with pytest.raises(OllamaPreflightError) as info:
        preflight_models(cfg, "local", ["qwen2.5:7b", "qwen3.5:4b", "gemma3:4b"])
    assert "ollama pull qwen3.5:4b" in str(info.value)
    assert "ollama pull gemma3:4b" in str(info.value)
    preflight_models(cfg, "anthropic", ["anything"])  # no Ollama call for a cloud provider


def test_run_eval_fails_fast_before_any_arm(attributed_book, tmp_path, monkeypatch):
    _, book_dir = attributed_book
    gold = export_gold(_book(), attribution_svc.load_report(book_dir)[0]).rows

    def missing(base_url, models):
        raise OllamaModelMissingError("x", base_url=base_url, model=models[0])

    monkeypatch.setattr(attribution_svc, "ollama_preflight", missing)
    built = _arm_provider(monkeypatch, ARMS)
    out = tmp_path / "out"
    with pytest.raises(OllamaPreflightError, match="ollama pull good:1b"):
        run_eval(_book(), gold, cfg=get_settings(), models=["good:1b"], out_dir=out)
    assert built == [] and not out.exists()


def test_run_eval_rejects_foreign_gold(tmp_path):
    from seiyuu.services import ServiceError

    row = _gold("ch001_b0002", 0, '"I am glad,"', "Elizabeth")
    gold = [row.model_copy(update={"book_id": "other-book"})]
    with pytest.raises(ServiceError, match="other book"):
        run_eval(_book(), gold, cfg=get_settings(), models=["m"], out_dir=tmp_path)


# ------------------------------------------------- review fixes: gold duplicate speakers


def test_gold_conflicts_flags_speakers_that_share_a_name():
    rows = [
        _gold("ch001_b0001", 0, Q1, "Jane", "Miss Bennet"),
        _gold("ch001_b0002", 0, Q2, "Elizabeth", "Miss Bennet"),  # shared alias
        _gold("ch001_b0003", 0, Q3, "Mr. Bennet"),
        _gold("ch001_b0004", 0, Q1, "Mrs. Bennet"),  # title guard: NOT the same person
    ]
    conflicts = gold_conflicts(gold_speakers(rows))
    assert [(c.speakers, c.shared) for c in conflicts] == [(["Elizabeth", "Jane"], ["miss bennet"])]
    bare = gold_conflicts(gold_speakers([rows[2], _gold("ch001_b0005", 0, Q2, "Bennet")]))
    assert [c.shared for c in bare] == [["bennet"]]  # bare surname vs Mr. Bennet


def _empty_stats():
    from seiyuu.services.evaluation import RunStats

    return RunStats(
        wall_seconds=0, llm_chunks=0, cached_chunks=0, fast_path_chunks=0, llm_calls=0,
        schema_failures=0, local_retries=0, escalations=0, flagged_blocks=0,
        seconds_per_chunk=None, p50_seconds_per_chunk=None, max_seconds_per_chunk=None,
        chunks=[],
    )  # fmt: skip


def test_unmerged_gold_duplicate_is_reported_not_silently_scored():
    """Gold kept 'Elizabeth' and 'Lizzy' apart (unmerged in Review); the arm CORRECTLY merged
    them. Scoring stays strict (ambiguous = wrong) but the report says why, loudly."""
    from seiyuu.services.evaluation import ArmResult

    b1, b2 = "ch001_b0001", "ch001_b0002"
    report = _report(
        [_char("liz", "Elizabeth", "Lizzy")], {1: [_dlg(b1, Q1, "liz"), _dlg(b2, Q2, "liz")]}
    )
    gold = [_gold(b1, 0, Q1, "Elizabeth"), _gold(b2, 0, Q2, "Lizzy")]
    score = score_report(gold, report)
    assert score.overall.correct == 0
    assert [(m.name, m.ambiguous_with) for m in score.ambiguous] == [
        ("Elizabeth", ["Elizabeth", "Lizzy"])
    ]
    arm = ArmResult(
        arm="arm-m", provider_id="fake", model_id="m", prompt_version="v1", arm_dir="x",
        stats=_empty_stats(), score=score,
    )  # fmt: skip
    md = render_markdown(EvalReport(book_id=BOOK_ID, chapters=[1], gold_quotes=2, arms=[arm]))
    assert "Ambiguous characters" in md and "Elizabeth [liz] → Elizabeth, Lizzy" in md
    assert "merge them in Review" in md


def test_export_gold_warns_on_speakers_sharing_an_alias():
    report = _report(
        [_char("c1", "Elizabeth", "Miss Bennet"), _char("c2", "Jane", "Miss Bennet")],
        {
            1: [
                _dlg("ch001_b0002", '"I am glad,"', "c1"),
                _nar("ch001_b0002", " said Elizabeth. "),
                _dlg("ch001_b0002", '"Very glad."', "c1"),
                _dlg("ch001_b0003", '"Indeed,"', "c2"),
                _nar("ch001_b0003", " replied Mr. Darcy."),
            ]
        },
    )
    result = export_gold(_book(), report, (1,))
    assert len(result.rows) == 3
    assert [c.speakers for c in result.conflicts] == [["Elizabeth", "Jane"]]


def test_run_eval_warns_about_gold_conflicts(attributed_book, tmp_path, monkeypatch):
    _, book_dir = attributed_book
    gold = [
        r.model_copy(update={"aliases": ["Miss Bennet"]})
        if r.speaker in ("Elizabeth", "Jane")
        else r
        for r in export_gold(_book(), attribution_svc.load_report(book_dir)[0]).rows
    ]
    _arm_provider(monkeypatch, ARMS)
    msgs: list[str] = []
    report = run_eval(
        _book(), gold, cfg=get_settings(), models=["good:1b"], out_dir=tmp_path / "o",
        progress=msgs.append,
    )  # fmt: skip
    assert [c.speakers for c in report.gold_conflicts] == [["Elizabeth", "Jane"]]
    assert any(m.startswith("WARNING: gold speakers 'Elizabeth' and 'Jane'") for m in msgs)
    md = (tmp_path / "o" / "eval_report.md").read_text(encoding="utf-8")
    assert "gold speakers that look like one person" in md


# ------------------------------------------- review fixes: hybrid never taints local arms


def _always_bad(chunk, registry, attempt):
    raise MalformedOutputError("never valid")


def test_hybrid_escalations_never_reach_the_local_only_arm(attributed_book, tmp_path):
    """The pipeline caches an escalated chunk under the PRIMARY key: a hybrid run must not
    leave the escalation provider's answers where a local-only run scores them as its own."""
    _, book_dir = attributed_book
    gold = export_gold(_book(), attribution_svc.load_report(book_dir)[0]).rows
    cfg, out = get_settings(), tmp_path / "o"

    def arm(**kw):
        return run_arm(
            _book(), gold, cfg=cfg, model="weak", out_dir=out, chapters=(1, 2),
            provider=FakeProvider(_always_bad, model="weak"), **kw,
        )  # fmt: skip

    def strong():
        return FakeProvider(_scripted(TRUTH), model="strong")

    hybrid = arm(escalation_provider=strong())
    assert hybrid.arm == "arm-weak-hybrid" and hybrid.hybrid
    assert hybrid.score.overall.accuracy == 1.0 and hybrid.stats.escalations >= 1
    assert (out / "arm-weak-hybrid" / ESCALATED_MARKER).exists()

    local = arm()  # same -o, no escalation, no --fresh
    assert local.arm == "arm-weak" and not local.hybrid
    assert local.stats.cached_chunks == 0 and local.stats.flagged_blocks >= 1
    assert local.score.overall.correct == 0  # a model that never produced valid output

    # A cached hybrid re-run still says its chunks came from the escalation provider.
    again = arm(escalation_provider=strong())
    assert again.stats.llm_chunks == 0 and again.stats.escalations == 0
    assert again.stats.cached_escalations == again.stats.cached_chunks
    assert again.stats.cached_escalations == hybrid.stats.escalations
    md = render_markdown(EvalReport(book_id=BOOK_ID, chapters=[1, 2], gold_quotes=1, arms=[again]))
    assert "(hybrid)" in md and f"(+{again.stats.cached_escalations} cached)" in md

    # --fresh clears the marker along with the cache.
    fresh = arm(escalation_provider=strong(), fresh=True)
    assert fresh.stats.cached_escalations == 0 and fresh.stats.escalations >= 1


def test_local_arm_refuses_a_cache_holding_escalated_chunks(tmp_path):
    arm_dir = tmp_path / "o" / "arm-weak"
    arm_dir.mkdir(parents=True)
    (arm_dir / ESCALATED_MARKER).write_text("[]", encoding="utf-8")
    with pytest.raises(ServiceError, match="escalated"):
        run_arm(
            _book(), [_gold("ch001_b0002", 0, '"I am glad,"', "Elizabeth")], cfg=get_settings(),
            model="weak", out_dir=tmp_path / "o", chapters=(1,),
            provider=FakeProvider(_always_bad, model="weak"),
        )  # fmt: skip


# ----------------------------------------------- review fixes: -o inside the book folder


@pytest.mark.parametrize("sub", ["", "eval-out"])
def test_out_dir_inside_book_folder_is_refused(attributed_book, monkeypatch, sub):
    _, book_dir = attributed_book
    gold = export_gold(_book(), attribution_svc.load_report(book_dir)[0]).rows
    built = _arm_provider(monkeypatch, ARMS)
    before = _snapshot(book_dir)
    out = book_dir / sub if sub else book_dir
    with pytest.raises(ServiceError, match="inside the book folder"):
        run_eval(
            _book(), gold, cfg=get_settings(), models=["good:1b"], out_dir=out,
            book_dir=book_dir,
        )  # fmt: skip
    with pytest.raises(ServiceError, match="inside the book folder"):
        run_arm(
            _book(), gold, cfg=get_settings(), model="good:1b", out_dir=out, chapters=(1,),
            book_dir=book_dir, provider=FakeProvider(_scripted(TRUTH), model="good:1b"),
        )  # fmt: skip
    assert built == [] and _snapshot(book_dir) == before
    assert not (book_dir / "eval-out").exists()
    assert not any(p.name.startswith(("arm-", "eval_report")) for p in book_dir.iterdir())


# ---------------------------------------------- review fixes: one failing arm is reported


def test_failing_arm_is_recorded_and_finished_arms_still_reported(
    attributed_book, tmp_path, monkeypatch
):
    books, book_dir = attributed_book
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        dump_gold(export_gold(_book(), attribution_svc.load_report(book_dir)[0]).rows),
        encoding="utf-8",
    )

    def boom(chunk, registry, attempt):
        raise AttributionError("context window exhausted")

    def fake_build(cfg, provider_id, model, prompt_version, *, emit_thoughts=False):
        return FakeProvider(boom if model == "broken:1b" else _scripted(ARMS[model]), model=model)

    monkeypatch.setattr(attribution_svc, "build_provider", fake_build)
    out = tmp_path / "out"
    res = CliRunner().invoke(
        main,
        ["eval", "run", BOOK_ID, "--gold", str(gold_path), "--models",
         "good:1b,broken:1b,bad:1b", "-o", str(out), "--books-dir", str(books)],
    )  # fmt: skip
    assert res.exit_code != 0 and "1 arm(s) failed: broken:1b" in res.output
    data = json.loads((out / "eval_report.json").read_text(encoding="utf-8"))
    assert [a["model_id"] for a in data["arms"]] == ["good:1b", "bad:1b"]
    assert data["failed"][0]["model"] == "broken:1b"
    assert "context window exhausted" in data["failed"][0]["error"]
    md = (out / "eval_report.md").read_text(encoding="utf-8")
    assert "| broken:1b | FAILED: AttributionError: context window exhausted |" in md
    assert "FAILED" in res.output
