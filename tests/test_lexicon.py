"""F3 — per-book pronunciation lexicon: pure respell semantics, IPA-on-Kokoro-only, JSON
round-trip, deterministic auto-suggest, render/estimate KEY PARITY, and the API CRUD surface.

Normalization stays a PURE function: the lexicon is compiled once and passed IN, feeding BOTH
synthesis and the whisper `expected` reference identically.
"""

import pytest
from fastapi.testclient import TestClient

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.api.main import create_app
from seiyuu.gpu import GpuResourceManager
from seiyuu.ingest import write_normalized
from seiyuu.normalize import normalize_text
from seiyuu.normalize.lexicon import (
    BookLexicon,
    LexiconEntry,
    compile_lexicon,
    load_lexicon,
    save_lexicon,
    suggest_terms,
)
from seiyuu.render import (
    estimate_render_cost,
    estimate_render_cost_single,
    render_book,
    render_book_multivoice,
)
from seiyuu.settings import Settings

# reuse the established multivoice fixtures (narrator=free kokoro, alice=paid cloud)
from test_render_cost_gate import FakeElevenEngine, _assignment, _library, _patch, _report


def _compile(*entries: LexiconEntry) -> "object":
    return compile_lexicon(BookLexicon(book_id="b", entries=list(entries)))


# -- pure respell semantics ---------------------------------------------------------------


def test_respell_word_boundary_and_case_insensitive_default() -> None:
    lex = _compile(LexiconEntry(term="Hermione", respelling="Her My Oh Nee"))
    got = normalize_text("hermione and HERMIONE ran.", lexicon=lex)
    assert got == "Her My Oh Nee and Her My Oh Nee ran."


def test_respell_no_partial_word_rewrite() -> None:
    # \b anchoring: 'cat' must not rewrite the middle of 'category' or 'scatter'
    lex = _compile(LexiconEntry(term="cat", respelling="KAT"))
    assert normalize_text("the cat left the category", lexicon=lex) == "the KAT left the category"
    # 'cats' != 'cat', 'scatter' contains but is not the word
    assert normalize_text("a scatter of cats", lexicon=lex) == "a scatter of cats"


def test_respell_longest_term_first() -> None:
    lex = _compile(
        LexiconEntry(term="Ann", respelling="AHN"),
        LexiconEntry(term="Anne", respelling="ANNIE"),
    )
    # the longer term wins at the same position; neither bleeds into 'Annette'
    assert normalize_text("Anne, Ann, and Annette", lexicon=lex) == "ANNIE, AHN, and Annette"


def test_case_sensitive_override() -> None:
    lex = _compile(LexiconEntry(term="Reed", respelling="REED", case_sensitive=True))
    assert normalize_text("Reed read the reed.", lexicon=lex) == "REED read the reed."


def test_single_pass_no_cascade() -> None:
    # a replacement's own output is never re-scanned by a later term
    lex = _compile(
        LexiconEntry(term="cat", respelling="dog feline"),
        LexiconEntry(term="feline", respelling="XXX"),
    )
    assert normalize_text("the cat", lexicon=lex) == "the dog feline"


def test_ipa_applied_on_kokoro_only() -> None:
    lex = _compile(LexiconEntry(term="Nginx", respelling="Engine X", ipa="ENJINX-IPA"))
    # Kokoro (non-validated) honors the IPA
    assert normalize_text("Nginx here", profile="kokoro", lexicon=lex) == "ENJINX-IPA here"
    # validated engines ignore IPA and speak the respelling (keeps the whisper reference valid)
    assert normalize_text("Nginx here", profile="chatterbox", lexicon=lex) == "Engine X here"
    assert normalize_text("Nginx here", profile="default", lexicon=lex) == "Engine X here"


def test_ipa_absent_falls_back_to_respelling_on_kokoro() -> None:
    lex = _compile(LexiconEntry(term="Nginx", respelling="Engine X"))  # no IPA
    assert normalize_text("Nginx", profile="kokoro", lexicon=lex) == "Engine X"


def test_none_lexicon_is_noop_and_pure() -> None:
    assert normalize_text("Hermione ran.", lexicon=None) == normalize_text("Hermione ran.")
    empty = _compile()
    assert normalize_text("Hermione ran.", lexicon=empty) == "Hermione ran."
    assert not empty  # empty compiles to a falsy no-op matcher


def test_respell_runs_before_number_expansion() -> None:
    # respell happens after unicode-clean but before number expansion, so both apply
    lex = _compile(LexiconEntry(term="Chapter", respelling="CHAPTOR"))
    assert normalize_text("Chapter 3", lexicon=lex) == "CHAPTOR three"


def test_determinism_and_idempotence() -> None:
    lex = _compile(LexiconEntry(term="Chapter", respelling="CHAPTOR"))
    out1 = normalize_text("Chapter 3 begins.", lexicon=lex)
    out2 = normalize_text("Chapter 3 begins.", lexicon=lex)
    assert out1 == out2
    # the respelled output is stable under a second pass (target not re-matched)
    assert normalize_text(out1, lexicon=lex) == out1


# -- persistence: round-trip + atomic write -----------------------------------------------


def test_lexicon_json_round_trip_and_atomic_write(tmp_path) -> None:
    book_dir = tmp_path / "books" / "b1"
    lex = BookLexicon(
        book_id="b1",
        entries=[
            LexiconEntry(
                term="Hermione", respelling="Her My Oh Nee", ipa="H-IPA", note="the witch"
            ),
            LexiconEntry(term="Reed", respelling="REED", case_sensitive=True),
        ],
    )
    path = save_lexicon(book_dir, lex)
    assert path.name == "lexicon.json"
    loaded = load_lexicon(book_dir, book_id="b1")
    assert loaded == lex
    # absent file -> empty lexicon, never an error
    assert load_lexicon(tmp_path / "books" / "nope", book_id="nope").entries == []


def test_lexicon_entry_rejects_blank_term() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        LexiconEntry(term="   ", respelling="x")


# -- deterministic auto-suggest -----------------------------------------------------------


def test_suggest_surfaces_repeated_mid_sentence_proper_nouns() -> None:
    texts = [
        "The wizard Gandalf spoke. Later Gandalf left with Frodo.",
        "A hobbit named Frodo carried it. Gandalf watched Frodo.",
    ]
    out = suggest_terms(texts, min_count=2)
    names = [s.term for s in out]
    # Frodo x3, Gandalf x3 -> sorted by count desc then alpha
    assert names[:2] == ["Frodo", "Gandalf"]
    assert all(s.count >= 2 for s in out)


def test_suggest_ignores_sentence_openers_and_existing_terms() -> None:
    texts = ["Gandalf went. Gandalf returned. The end. The end."]
    # 'The' is a sentence opener (never mid-sentence here) and stoplisted; Gandalf already known
    out = suggest_terms(texts, existing_terms=["gandalf"], min_count=2)
    assert out == []


# -- KEY PARITY: single-voice render vs estimate ------------------------------------------


def test_single_voice_key_parity(tmp_path) -> None:
    """A lexicon term shifts normalized_text_hash IDENTICALLY at render and estimate: render
    with the lexicon, and the estimate WITH it counts every segment cached, WITHOUT it misses
    exactly the affected segments."""
    out = tmp_path / "out"
    lex = _compile(LexiconEntry(term="chapter", respelling="CHAPTOR"))
    render_book(make_book(), FakeEngine(), "test_voice", out, gpu=GpuResourceManager(), lexicon=lex)
    # 5 speakable blocks; 'chapter' (case-insensitive) hits "Chapter 1", "Chapter 2",
    # "Second chapter." -> 3 affected, 2 untouched
    est_with = estimate_render_cost_single(
        make_book(), FakeEngine(), "test_voice", out, lexicon=lex
    )
    assert est_with.cached_segments == 5

    est_without = estimate_render_cost_single(make_book(), FakeEngine(), "test_voice", out)
    assert est_without.cached_segments == 2
    assert est_without.free_segments == 3  # the affected segments miss the lexicon-keyed cache


# -- KEY PARITY: multivoice render vs estimate --------------------------------------------


def test_multivoice_key_parity(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, FakeElevenEngine())
    lib, out = _library(tmp_path), tmp_path / "out"
    lex = _compile(LexiconEntry(term="chapter", respelling="CHAPTOR"))
    render_book_multivoice(
        _report(), make_book(), lib, _assignment(), out,
        gpu=GpuResourceManager(), allow_paid=True, lexicon=lex,
    )  # fmt: skip
    est_with = estimate_render_cost(_report(), make_book(), lib, _assignment(), out, lexicon=lex)
    assert est_with.cached_segments == 5

    est_without = estimate_render_cost(_report(), make_book(), lib, _assignment(), out)
    # "Chapter 1", "Chapter 2", "Second chapter." carry the term -> 3 miss, 2 stay cached
    assert est_without.cached_segments == 2


# -- KEY PARITY: the money.compute_estimate site (API path) --------------------------------


def _settings(tmp_path, **over) -> Settings:
    defaults = dict(
        books_dir=tmp_path / "books",
        output_dir=tmp_path / "output",
        voices_dir=tmp_path / "voices",
        data_dir=tmp_path / "data",
        anthropic_api_key=None,
        elevenlabs_api_key=None,
    )
    defaults.update(over)
    return Settings(_env_file=None, **defaults)


def test_compute_estimate_uses_on_disk_lexicon(tmp_path, monkeypatch) -> None:
    from seiyuu.api.money import compute_estimate, resolve_single
    from seiyuu.api.registry import EngineRegistry

    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda engine_id, **kw: FakeEngine())
    cfg = _settings(tmp_path)
    book = make_book()
    book_id = book.book_meta.book_id
    write_normalized(book, cfg.books_dir)
    save_lexicon(
        cfg.books_dir / book_id,
        BookLexicon(book_id=book_id, entries=[LexiconEntry(term="chapter", respelling="CHAPTOR")]),
    )
    registry = EngineRegistry(cfg)
    monkeypatch.setattr(registry, "get", lambda engine_id: FakeEngine())
    single = resolve_single(cfg, None)  # kokoro default preset, speed 1.0, pinned seed

    # render under the SAME on-disk lexicon compute_estimate will load
    render_book(
        book, FakeEngine(), single.voice_id, cfg.output_dir / book_id,
        settings=single.settings, seed=single.seed, gpu=GpuResourceManager(),
        lexicon=compile_lexicon(load_lexicon(cfg.books_dir / book_id)),
    )  # fmt: skip
    est = compute_estimate(cfg, registry, book, book_id, mode="single", chapters=(), single=single)
    assert est.est.cached_segments == 5  # compute_estimate applied the lexicon -> all hit

    # remove the lexicon: compute_estimate now normalizes without it and the 3 affected miss
    (cfg.books_dir / book_id / "lexicon.json").unlink()
    est2 = compute_estimate(cfg, registry, book, book_id, mode="single", chapters=(), single=single)
    assert est2.est.cached_segments == 2


# -- API CRUD -----------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path):
    cfg = _settings(tmp_path)
    write_normalized(make_book(), cfg.books_dir)
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        c.cfg = cfg
        yield c


BOOK_ID = "test-book-00000000"


def test_api_get_empty_then_put_then_get(api) -> None:
    got = api.get(f"/api/books/{BOOK_ID}/lexicon")
    assert got.status_code == 200
    body = got.json()
    assert body["entries"] == []
    assert isinstance(body["suggestions"], list)

    put = api.put(
        f"/api/books/{BOOK_ID}/lexicon",
        json={"entries": [{"term": "chapter", "respelling": "CHAPTOR"}]},
    )
    assert put.status_code == 200
    saved = put.json()
    assert len(saved["entries"]) == 1
    assert saved["total_speakable_blocks"] == 5
    assert saved["affected_blocks"] == 3  # Chapter 1, Chapter 2, Second chapter.

    again = api.get(f"/api/books/{BOOK_ID}/lexicon").json()
    assert again["entries"][0]["term"] == "chapter"
    assert again["entries"][0]["respelling"] == "CHAPTOR"


def test_api_preview_counts_without_saving(api) -> None:
    resp = api.post(
        f"/api/books/{BOOK_ID}/lexicon/preview",
        json={"entries": [{"term": "chapter", "respelling": "CHAPTOR"}]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"affected_blocks": 3, "total_speakable_blocks": 5}
    # nothing was persisted
    assert api.get(f"/api/books/{BOOK_ID}/lexicon").json()["entries"] == []


def test_api_rejects_duplicate_terms(api) -> None:
    resp = api.put(
        f"/api/books/{BOOK_ID}/lexicon",
        json={
            "entries": [
                {"term": "Chapter", "respelling": "A"},
                {"term": "chapter", "respelling": "B"},  # same term, case-insensitive collision
            ]
        },
    )
    assert resp.status_code == 422


def test_api_unknown_book_404(api) -> None:
    assert api.get("/api/books/nope-00000000/lexicon").status_code == 404
