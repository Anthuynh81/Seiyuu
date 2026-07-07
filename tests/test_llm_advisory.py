"""Theme A — opt-in LLM advisory layers (F3 respell suggester, F4 Layer-2 caster).

Fakes only, NO live LLM. The load-bearing guarantees under test:

- F3 is advisory: a fake provider's respellings are surfaced, hallucinated/blank/duplicate
  entries are dropped, and the deterministic hard-name surfacer keeps working with no LLM.
- F4 can NEVER break cast_book: a DEGENERATE preference (every character wants the same trait)
  still yields DISTINCT, DETERMINISTIC voices; with the layer off the heuristic is byte-identical.
- The paid gate fires ONLY for anthropic and never on an automatic path: anthropic-without-confirm
  is blocked (402); local is free-but-explicit and runs.
- A local provider acquires the GPU through the resource manager; anthropic (network-only) doesn't.
"""

import re

import pytest
from fastapi.testclient import TestClient

from factories import make_book
from seiyuu.api.main import create_app
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    Segment,
)
from seiyuu.attribute.providers.base import AttributionLLM
from seiyuu.gpu import GpuResourceManager
from seiyuu.ingest import write_normalized
from seiyuu.ingest.models import Block, BlockType, BookMeta, Chapter, NormalizedBook
from seiyuu.normalize.respell import RespellSuggestion, suggest_respellings
from seiyuu.services.llm_advisory import (
    ResolvedAdvisory,
    resolve_advisory,
    run_respell_suggestions,
)
from seiyuu.settings import Settings
from seiyuu.voices import VoiceKind, VoiceLibrary
from seiyuu.voices.casting import KNOWN_TRAITS, cast_book
from seiyuu.voices.llm_caster import suggest_trait_hints

PROMPTS_DIR = Settings(_env_file=None).prompts_dir


# -- fakes --------------------------------------------------------------------------------


class _FakeProvider(AttributionLLM):
    """A schema-enforced-call stand-in: returns whatever ``respond(prompt)`` yields, counting
    calls. Subclasses AttributionLLM so it is a valid GpuConsumer (default no-op ``unload``)."""

    def __init__(self, respond, *, provider_id="fake-local", model="fake-1.0", uses_gpu=True):
        self.model_id = model
        self.provider_id = provider_id
        self.uses_gpu = uses_gpu
        self._respond = respond
        self.calls = 0

    def complete_structured(self, prompt, schema, *, tool_name="", tool_description=""):
        self.calls += 1
        return self._respond(prompt)

    def _complete_json(self, prompt, schema, attempt=0):  # pragma: no cover - never called
        raise NotImplementedError


def _echo_respell(prompt: str) -> dict:
    """Respell exactly the requested terms (the section after the '## Terms' header), UPPERCASED —
    plus one out-of-list hallucination the filter must drop."""
    tail = prompt.split("## Terms to respell", 1)[-1]
    terms = re.findall(r"^- (.+)$", tail, re.MULTILINE)
    suggestions = [{"term": t, "respelling": t.upper()} for t in terms]
    suggestions.append({"term": "Hallucinated", "respelling": "NOPE"})  # not requested -> dropped
    return {"suggestions": suggestions}


def _degenerate_caster(traits=("young",)):
    """Every character prefers the SAME trait — the worst case for collision-freeness."""

    def respond(prompt: str) -> dict:
        tail = prompt.split("## Characters to cast", 1)[-1]
        ids = re.findall(r"character_id:\s*([^\s,]+)", tail)
        return {"preferences": [{"character_id": cid, "traits": list(traits)} for cid in ids]}

    return respond


# ==========================================================================================
# F3 — respell suggester (core)
# ==========================================================================================


def test_f3_suggest_respellings_returns_and_filters() -> None:
    provider = _FakeProvider(_echo_respell)
    out = suggest_respellings(
        provider, ["Zorblax", "Qwyx"], prompts_dir=PROMPTS_DIR, prompt_version="v1"
    )
    assert {s.term: s.respelling for s in out} == {"Zorblax": "ZORBLAX", "Qwyx": "QWYX"}
    # the hallucinated (non-requested) term never appears
    assert all(s.term != "Hallucinated" for s in out)


def test_f3_drops_blank_and_duplicate_and_unrequested() -> None:
    reply = {
        "suggestions": [
            {"term": "Aster", "respelling": "AS-ter"},
            {"term": "Aster", "respelling": "second"},  # duplicate -> first wins
            {"term": "Aster", "respelling": "   "},  # blank respelling -> dropped
            {"term": "Ghost", "respelling": "BOO"},  # not requested -> dropped
        ]
    }
    out = suggest_respellings(_FakeProvider(lambda _p: reply), ["Aster"], prompts_dir=PROMPTS_DIR)
    assert out == [RespellSuggestion(term="Aster", respelling="AS-ter")]


def test_f3_empty_terms_never_calls_the_llm() -> None:
    provider = _FakeProvider(lambda _p: pytest.fail("LLM must not be called for no terms"))
    assert suggest_respellings(provider, [], prompts_dir=PROMPTS_DIR) == []
    assert provider.calls == 0


def test_f3_malformed_reply_degrades_to_empty() -> None:
    assert suggest_respellings(_FakeProvider(lambda _p: []), ["X"], prompts_dir=PROMPTS_DIR) == []


# ==========================================================================================
# F4 — Layer-2 caster (core): the LLM can NEVER break cast_book
# ==========================================================================================


def _chars(n: int, gender: str = "female") -> list[Character]:
    g = gender[0]
    return [Character(id=f"{g}{i:03d}", canonical_name=f"N{i}", gender=gender) for i in range(n)]


def _sigs(cast) -> list[tuple]:
    return [tuple(r) for r in cast.values()]


def test_f4_degenerate_hints_still_collision_free_and_deterministic() -> None:
    chars = _chars(10, "female") + _chars(10, "male")
    hints = {c.id: {"young"} for c in chars}  # EVERY character wants the same trait
    a = cast_book(chars, narrator_preset="af_heart", accent="a", trait_hints=hints)
    b = cast_book(list(reversed(chars)), narrator_preset="af_heart", accent="a", trait_hints=hints)
    assert a == b  # deterministic given the same (degenerate) LLM output
    sigs = _sigs(a)
    assert len(sigs) == len(set(sigs)) == len(chars)  # still every character a DISTINCT voice


def test_f4_garbage_traits_are_ignored() -> None:
    chars = _chars(6, "female")
    garbage = {c.id: {"sparkly", "loud", "☠"} for c in chars}  # none in KNOWN_TRAITS
    biased = cast_book(chars, narrator_preset="af_heart", trait_hints=garbage)
    baseline = cast_book(chars, narrator_preset="af_heart")  # keyword bias (chars have none)
    # unknown tags filter to empty -> same effect as no bias; still distinct either way
    assert biased == baseline
    assert len({tuple(r) for r in biased.values()}) == len(chars)


def test_f4_layer_off_is_byte_identical_to_heuristic() -> None:
    chars = _chars(8, "male")
    assert cast_book(chars, narrator_preset="am_adam") == cast_book(
        chars, narrator_preset="am_adam", trait_hints=None
    )
    assert cast_book(chars, narrator_preset="am_adam", trait_hints={}) == cast_book(
        chars, narrator_preset="am_adam"
    )


def test_f4_hint_can_steer_which_distinct_voice() -> None:
    # A single 'deep' hint should pull a character onto a deep-tagged preset when one is free.
    chars = _chars(2, "male")
    hinted = cast_book(chars, narrator_preset="af_heart", trait_hints={chars[0].id: {"deep"}})
    from seiyuu.voices.casting import _DEEP

    primary = max(hinted[chars[0].id], key=lambda pw: pw[1])[0]
    assert primary in _DEEP  # the advisory hint biased the tie-breaker


def test_f4_suggest_trait_hints_filters_unknown_ids_and_traits() -> None:
    chars = _chars(2, "female")
    reply = {
        "preferences": [
            {"character_id": chars[0].id, "traits": ["young", "bogus"]},
            {"character_id": "not-a-real-id", "traits": ["deep"]},  # unknown id -> dropped
        ]
    }
    hints = suggest_trait_hints(_FakeProvider(lambda _p: reply), chars, prompts_dir=PROMPTS_DIR)
    assert hints == {chars[0].id: {"young"}}  # unknown id gone, "bogus" filtered out
    assert all(t in KNOWN_TRAITS for tags in hints.values() for t in tags)


# ==========================================================================================
# service layer: GPU discipline + paid resolution
# ==========================================================================================


def _cfg(tmp_path, **over) -> Settings:
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


def test_local_provider_acquires_and_frees_the_gpu(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    gpu = GpuResourceManager()
    provider = _FakeProvider(_echo_respell, uses_gpu=True)

    seen = {}

    def respond(prompt):
        seen["resident_during_call"] = gpu.resident  # the LLM is GPU-resident while it runs
        return _echo_respell(prompt)

    provider._respond = respond
    resolved = ResolvedAdvisory("local", "m", is_paid=False)
    run_respell_suggestions(cfg, resolved, ["Zorblax"], gpu=gpu, provider=provider)
    assert seen["resident_during_call"] is not None  # acquired for the call (GPU discipline)
    assert gpu.resident is None  # freed in finally afterwards


def test_anthropic_provider_never_touches_the_gpu(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    gpu = GpuResourceManager()
    provider = _FakeProvider(_echo_respell, provider_id="anthropic", uses_gpu=False)

    def respond(prompt):
        assert gpu.resident is None  # network-only provider is never made resident
        return _echo_respell(prompt)

    provider._respond = respond
    run_respell_suggestions(
        cfg,
        ResolvedAdvisory("anthropic", "m", is_paid=True),
        ["Zorblax"],
        gpu=gpu,
        provider=provider,
    )
    assert gpu.resident is None


def test_resolve_advisory_marks_only_anthropic_paid(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    assert resolve_advisory(cfg, "local", None, None).is_paid is False
    assert resolve_advisory(cfg, "local", None, "anthropic").is_paid is True  # per-call override
    # default model resolves per provider
    assert resolve_advisory(cfg, "anthropic", None, None).model == cfg.anthropic_model
    assert resolve_advisory(cfg, "local", None, None).model == cfg.attribution_model


# ==========================================================================================
# F3 — API surface
# ==========================================================================================

BOOK_ID = "test-book-00000000"


@pytest.fixture
def api(tmp_path):
    cfg = _cfg(tmp_path)
    write_normalized(make_book(), cfg.books_dir)
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        c.cfg = cfg
        yield c


def _patch_provider(monkeypatch, provider) -> None:
    monkeypatch.setattr("seiyuu.services.llm_advisory.build_provider", lambda *a, **k: provider)


def test_f3_api_local_runs_and_returns_suggestions(api, monkeypatch) -> None:
    _patch_provider(monkeypatch, _FakeProvider(_echo_respell))
    resp = api.post(
        f"/api/books/{BOOK_ID}/lexicon/suggest-respellings",
        json={"terms": ["Zorblax", "Qwyx"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "local"
    assert {s["term"]: s["respelling"] for s in body["suggestions"]} == {
        "Zorblax": "ZORBLAX",
        "Qwyx": "QWYX",
    }


def test_f3_api_anthropic_without_confirm_is_blocked(api, monkeypatch) -> None:
    # If the gate ever failed, this would build a paid client — so make that fatal.
    _patch_provider(
        monkeypatch, _FakeProvider(lambda _p: pytest.fail("paid path ran without confirm"))
    )
    resp = api.post(
        f"/api/books/{BOOK_ID}/lexicon/suggest-respellings",
        json={"terms": ["Zorblax"], "provider": "anthropic"},
    )
    assert resp.status_code == 402
    assert resp.json()["error"]["code"] == "payment_confirmation_required"


def test_f3_api_anthropic_confirmed_but_no_key_is_503(api, monkeypatch) -> None:
    _patch_provider(monkeypatch, _FakeProvider(_echo_respell))
    resp = api.post(
        f"/api/books/{BOOK_ID}/lexicon/suggest-respellings",
        json={"terms": ["Zorblax"], "provider": "anthropic", "confirm_paid": True},
    )
    assert resp.status_code == 503  # confirmed, but no ANTHROPIC_API_KEY


def test_f3_api_empty_terms_falls_back_to_deterministic_suggestions(tmp_path, monkeypatch) -> None:
    # A book with a recurring mid-sentence proper noun so the deterministic surfacer finds it.
    book = NormalizedBook(
        book_meta=BookMeta(
            book_id="fallback-book-00", title="T", source_path="t.epub", source_sha256="0" * 64
        ),
        chapters=[
            Chapter(
                title="One",
                blocks=[
                    Block(id="ch001_b0001", type=BlockType.HEADING, text="One"),
                    Block(
                        id="ch001_b0002",
                        type=BlockType.PARAGRAPH,
                        text="the Zorblax hummed and the Zorblax glowed",
                    ),
                ],
            )
        ],
    )
    cfg = _cfg(tmp_path)
    write_normalized(book, cfg.books_dir)
    _patch_provider(monkeypatch, _FakeProvider(_echo_respell))
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        resp = c.post("/api/books/fallback-book-00/lexicon/suggest-respellings", json={})
    assert resp.status_code == 200, resp.text
    terms = {s["term"] for s in resp.json()["suggestions"]}
    assert "Zorblax" in terms  # empty request -> deterministic terms -> LLM respelled them


def test_f3_api_uningested_book_409(tmp_path) -> None:
    cfg = _cfg(tmp_path)  # no book written
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        resp = c.post("/api/books/ghost-00/lexicon/suggest-respellings", json={"terms": ["X"]})
    assert resp.status_code in (404, 409)


# ==========================================================================================
# F4 — API surface (draft with the LLM caster)
# ==========================================================================================


def _write_report(cfg: Settings) -> None:
    report = AttributionReport(
        book_id="cast-book",
        provider_id="local",
        model_id="m",
        prompt_version="v5",
        registry=CharacterRegistry(
            characters=[
                Character(id="alice", canonical_name="Alice", gender="female"),
                Character(id="bob", canonical_name="Bob", gender="male"),
                Character(id="cara", canonical_name="Cara", gender="female"),
                Character(id="dan", canonical_name="Dan", gender="male"),
            ]
        ),
        chapters=[
            AttributedChapter(
                index=1,
                title="Chapter 1",
                segments=[
                    Segment(block_id="ch001_b0001", type="narration", speaker=None, text="Hi."),
                ],
            )
        ],
    )
    (cfg.books_dir / "cast-book").mkdir(parents=True, exist_ok=True)
    (cfg.books_dir / "cast-book" / "attribution.json").write_text(
        report.model_dump_json(), encoding="utf-8"
    )


@pytest.fixture
def cast_api(tmp_path):
    cfg = _cfg(tmp_path)
    _write_report(cfg)
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        c.cfg = cfg
        yield c


def _recipe_sig(meta) -> tuple:
    if meta.kind is VoiceKind.PRESET:
        return (meta.preset_id,)
    return tuple((b.preset_id, b.weight) for b in meta.blend)


def test_f4_api_draft_with_llm_yields_distinct_voices(cast_api, monkeypatch) -> None:
    _patch_provider(monkeypatch, _FakeProvider(_degenerate_caster(("young",))))
    resp = cast_api.post(
        "/api/books/cast-book/assignment/draft",
        json={"strategy": "smart", "use_llm": True},
    )
    assert resp.status_code == 201, resp.text
    lib = VoiceLibrary(cast_api.cfg.voices_dir)
    assignment = resp.json()["assignment"]
    sigs = [_recipe_sig(lib.load(vid)) for vid in assignment["assignments"].values()]
    # even with EVERY character asking for the same trait, the cast stays collision-free
    assert len(sigs) == len(set(sigs)) == 4


def test_f4_api_draft_llm_anthropic_without_confirm_is_blocked(cast_api, monkeypatch) -> None:
    _patch_provider(monkeypatch, _FakeProvider(lambda _p: pytest.fail("paid caster ran")))
    resp = cast_api.post(
        "/api/books/cast-book/assignment/draft",
        json={"strategy": "smart", "use_llm": True, "cast_provider": "anthropic"},
    )
    assert resp.status_code == 402
    assert resp.json()["error"]["code"] == "payment_confirmation_required"


def test_f4_api_hash_strategy_never_calls_the_llm(cast_api, monkeypatch) -> None:
    # use_llm on the hash path must be a no-op: building a provider at all would be a bug.
    _patch_provider(monkeypatch, _FakeProvider(lambda _p: pytest.fail("LLM ran on hash path")))
    resp = cast_api.post(
        "/api/books/cast-book/assignment/draft",
        json={"strategy": "hash", "use_llm": True},
    )
    assert resp.status_code == 201, resp.text
