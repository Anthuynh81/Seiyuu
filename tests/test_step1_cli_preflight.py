"""B1: the CLI render path verifies-then-consumes, like the API.

A single-use cost token must survive any pre-flight failure that has nothing to do with
money (missing API key, consent, chapter-subset merge identity, GPU held elsewhere) and be
re-usable once the cause is fixed; a valid run consumes it exactly once, and no paid
synthesis can start before that consumption.
"""

import json
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

import seiyuu.engines
from factories import make_book
from seiyuu.cli import main
from seiyuu.cli.common import _check_cost_gate, _commit_cost_gate
from seiyuu.engines import SynthesisError
from seiyuu.render import CostEstimate, CostQuote, issue_quote
from seiyuu.render.gate import _consume_quote, quote_consumed
from seiyuu.repository.lock import FileLockHandle
from seiyuu.settings import get_settings
from seiyuu.voices import VoiceAssignment, VoiceKind, VoiceLibrary, VoiceMeta
from test_render_cost_gate import FakeElevenEngine, _assignment, _library, _patch, _report


class NoKeyElevenEngine(FakeElevenEngine):
    """Same identity/pricing as FakeElevenEngine (same fingerprint), but cannot
    authenticate — what the real adapter does without ELEVENLABS_API_KEY."""

    @property
    def client(self):
        raise SynthesisError("ELEVENLABS_API_KEY not set; required for the elevenlabs engine")


def _data_dir():
    return get_settings().data_dir


def _sig(token: str) -> str:
    return CostQuote.decode(token).sig


def _token_from(output: str) -> str:
    return next(line for line in output.splitlines() if line.startswith("cq1."))


# --- the gate halves -------------------------------------------------------------------------


def _est(total=1.5):
    return CostEstimate(
        total_usd=total, paid_segments=3, cached_segments=0, free_segments=2, fingerprint="fp"
    )


def test_check_never_consumes_and_commit_consumes_exactly_once(tmp_path):
    cfg = SimpleNamespace(render_max_usd=25.0, data_dir=tmp_path)
    quote = issue_quote(
        _est(), book_id="b", chapters=(1,), assignment_hash="ah",
        max_usd=25.0, ttl_seconds=900, data_dir=tmp_path,
    )  # fmt: skip
    kwargs = dict(book_id="b", chapters=(1,), assignment_hash="ah", cost_token=quote.encode())

    for _ in range(3):  # any number of dry-runs leave the token unspent
        pending = _check_cost_gate(cfg, _est(), **kwargs)
        assert pending is not None and not quote_consumed(tmp_path, quote.sig)

    assert _commit_cost_gate(cfg, pending, confirm_cost=False) == pytest.approx(1.5)
    assert quote_consumed(tmp_path, quote.sig)
    with pytest.raises(click.ClickException, match="already used"):
        _check_cost_gate(cfg, _est(), **kwargs)  # the dry-run now refuses a spent token
    with pytest.raises(click.ClickException, match="already used"):
        _commit_cost_gate(cfg, pending, confirm_cost=False)  # and the commit can't re-spend


def test_check_still_enforces_ceiling_and_free_renders_skip_the_gate(tmp_path):
    cfg = SimpleNamespace(render_max_usd=1.0, data_dir=tmp_path)
    no_token = dict(book_id="b", chapters=(), assignment_hash=None, cost_token=None)
    with pytest.raises(click.ClickException, match="ceiling"):
        _check_cost_gate(cfg, _est(total=5.0), **no_token)
    assert _check_cost_gate(cfg, _est(total=0.0), **no_token) is None
    assert _commit_cost_gate(cfg, None, confirm_cost=True) is None  # nothing paid may run


# --- single-voice CLI ------------------------------------------------------------------------


@pytest.fixture
def single_world(tmp_path):
    from seiyuu.ingest.epub import write_normalized

    book = make_book()
    write_normalized(book, tmp_path / "books")
    return tmp_path / "books", tmp_path / "output", book.book_meta.book_id


def _use_engine(monkeypatch, engine):
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: engine)


def _single(command, world, *extra):
    books, out, bid = world
    return [
        command, bid, "--voice", "voice_x",
        "--books-dir", str(books), "--output-dir", str(out), *extra,
    ]  # fmt: skip


def _single_token(world, monkeypatch) -> str:
    _use_engine(monkeypatch, FakeElevenEngine())
    est = CliRunner().invoke(main, _single("estimate-cost", world, "--token"))
    assert est.exit_code == 0, est.output
    return _token_from(est.output)


def test_single_preflight_failure_leaves_token_reusable(single_world, monkeypatch):
    _books, out, bid = single_world
    token = _single_token(single_world, monkeypatch)
    runner = CliRunner()

    broken = NoKeyElevenEngine()
    _use_engine(monkeypatch, broken)
    refused = runner.invoke(main, _single("render", single_world, "--cost-token", token))
    assert refused.exit_code != 0
    assert "cannot authenticate" in refused.output
    assert broken.calls == []
    assert not quote_consumed(_data_dir(), _sig(token))  # the approval survives
    assert not (out / bid / "manifest.json").exists()

    fixed = FakeElevenEngine()  # key configured: the SAME token now renders
    _use_engine(monkeypatch, fixed)
    ok = runner.invoke(main, _single("render", single_world, "--cost-token", token))
    assert ok.exit_code == 0, ok.output
    assert len(fixed.calls) == 5
    assert quote_consumed(_data_dir(), _sig(token))

    # single-use still holds: a forced re-render re-prices the same paid work, and the
    # spent token is refused before any synthesis
    again = runner.invoke(main, _single("render", single_world, "--force", "--cost-token", token))
    assert again.exit_code != 0 and "already used" in again.output
    assert len(fixed.calls) == 5


def test_single_no_spend_prompt_for_a_render_that_cannot_start(single_world, monkeypatch):
    _use_engine(monkeypatch, NoKeyElevenEngine())
    result = CliRunner().invoke(main, _single("render", single_world), input="y\n")
    assert result.exit_code != 0
    assert "cannot authenticate" in result.output
    assert "Continue?" not in result.output  # never asked to approve a doomed render


def test_single_paid_synthesis_only_after_consumption_and_exactly_once(single_world, monkeypatch):
    token = _single_token(single_world, monkeypatch)
    sig, data_dir = _sig(token), _data_dir()
    consumes = []

    def counting_consume(*args, **kwargs):
        consumes.append(args)
        return _consume_quote(*args, **kwargs)

    monkeypatch.setattr("seiyuu.render.gate._consume_quote", counting_consume)

    class ConsumedFirstEngine(FakeElevenEngine):
        def _synthesize_native(self, text, voice, settings):
            assert quote_consumed(data_dir, sig), "paid synthesis before the token was consumed"
            return super()._synthesize_native(text, voice, settings)

    engine = ConsumedFirstEngine()
    _use_engine(monkeypatch, engine)
    ok = CliRunner().invoke(main, _single("render", single_world, "--cost-token", token))
    assert ok.exit_code == 0, ok.output
    assert len(engine.calls) == 5
    assert len(consumes) == 1  # consumed exactly once for the whole render


def test_single_commit_refusal_blocks_all_synthesis(single_world, monkeypatch):
    """The commit (atomic consume) is the enforcement: if the token is spent between the
    dry-run and the commit (a concurrent render), nothing renders or bills."""
    token = _single_token(single_world, monkeypatch)
    engine = FakeElevenEngine()
    _use_engine(monkeypatch, engine)
    quote = CostQuote.decode(token)

    def racing_preflight(*_args, **_kwargs):
        _consume_quote(_data_dir(), quote.sig, quote.expires_at, quote.issued_at)

    render_calls = []
    monkeypatch.setattr("seiyuu.render.preflight.preflight_single", racing_preflight)
    monkeypatch.setattr("seiyuu.render.render_book", lambda *a, **k: render_calls.append(k))
    result = CliRunner().invoke(main, _single("render", single_world, "--cost-token", token))
    assert result.exit_code != 0 and "already used" in result.output
    assert render_calls == [] and engine.calls == []


def _cloud_library(tmp_path) -> VoiceLibrary:
    """elena_x: CLONED ElevenLabs (from _library); rachel_x: a `voice add-cloud` PRESET."""
    lib = _library(tmp_path)
    lib.save(
        VoiceMeta(voice_id="rachel_x", name="Rachel", kind=VoiceKind.PRESET,
                  engine="elevenlabs", preset_id="EXAVITQu")
    )  # fmt: skip
    return lib


def _single_lib(command, world, voice_id, lib, *extra):
    books, out, bid = world
    return [
        command, bid, "--voice", voice_id, "--voices-dir", str(lib.voices_dir),
        "--books-dir", str(books), "--output-dir", str(out), *extra,
    ]  # fmt: skip


def test_single_elevenlabs_preset_library_voice_still_renders(single_world, monkeypatch, tmp_path):
    """A `voice add-cloud` PRESET resolves to its preset_id (the raw cloud id) inside
    render_book, so the pre-flight must let it through and the token is spent once."""
    lib = _cloud_library(tmp_path)
    engine = FakeElevenEngine()
    _use_engine(monkeypatch, engine)
    runner = CliRunner()
    est = runner.invoke(
        main, _single_lib("estimate-cost", single_world, "rachel_x", lib, "--token")
    )
    assert est.exit_code == 0, est.output
    token = _token_from(est.output)

    ok = runner.invoke(
        main, _single_lib("render", single_world, "rachel_x", lib, "--cost-token", token)
    )
    assert ok.exit_code == 0, ok.output
    assert engine.calls and {voice for _text, voice in engine.calls} == {"EXAVITQu"}
    assert quote_consumed(_data_dir(), _sig(token))


def test_single_elevenlabs_cloned_library_voice_refused_token_unspent(
    single_world, monkeypatch, tmp_path
):
    """A CLONED ElevenLabs voice would reach the cloud as its library voice_id (doomed):
    refused up front, before the commit, so the approval survives."""
    lib = _cloud_library(tmp_path)
    engine = FakeElevenEngine()
    _use_engine(monkeypatch, engine)
    runner = CliRunner()
    est = runner.invoke(main, _single_lib("estimate-cost", single_world, "elena_x", lib, "--token"))
    assert est.exit_code == 0, est.output
    token = _token_from(est.output)

    refused = runner.invoke(
        main, _single_lib("render", single_world, "elena_x", lib, "--cost-token", token)
    )
    assert refused.exit_code != 0
    assert "--multivoice" in refused.output
    assert engine.calls == []
    assert not quote_consumed(_data_dir(), _sig(token))


# --- multivoice CLI --------------------------------------------------------------------------


@pytest.fixture
def mv_world(tmp_path, monkeypatch):
    book = make_book()
    bid = book.book_meta.book_id
    books, out = tmp_path / "books", tmp_path / "output"
    (books / bid).mkdir(parents=True)
    (books / bid / "normalized.json").write_text(book.model_dump_json(), encoding="utf-8")
    (books / bid / "attribution.json").write_text(_report().model_dump_json(), encoding="utf-8")
    (out / bid).mkdir(parents=True)
    _write_assignment(out, bid, _assignment())
    lib = _library(tmp_path)
    eleven = FakeElevenEngine()
    _patch(monkeypatch, eleven)
    return SimpleNamespace(books=books, out=out, voices=lib.voices_dir, bid=bid, eleven=eleven)


def _write_assignment(out, bid, assignment: VoiceAssignment) -> None:
    (out / bid / "assignments.json").write_text(assignment.model_dump_json(), encoding="utf-8")


def _mv(command, w, *extra):
    return [
        command, w.bid, "--books-dir", str(w.books), "--output-dir", str(w.out),
        "--voices-dir", str(w.voices), *extra,
    ]  # fmt: skip


def _mv_token(w, *extra) -> str:
    est = CliRunner().invoke(main, _mv("estimate-cost", w, "--token", *extra))
    assert est.exit_code == 0, est.output
    return _token_from(est.output)


def _mv_render(w, token, *extra):
    return CliRunner().invoke(main, _mv("render", w, "--multivoice", "--cost-token", token, *extra))


def test_multivoice_consent_failure_leaves_token_reusable(mv_world):
    w = mv_world
    token = _mv_token(w)
    meta_path = VoiceLibrary(w.voices).meta_path("elena_x")
    original = meta_path.read_text(encoding="utf-8")
    revoked = json.loads(original) | {"consent_attested": False}
    meta_path.write_text(json.dumps(revoked), encoding="utf-8")

    refused = _mv_render(w, token)
    assert refused.exit_code != 0 and "consent" in refused.output
    assert not quote_consumed(_data_dir(), _sig(token))
    assert w.eleven.calls == []

    meta_path.write_text(original, encoding="utf-8")  # consent restored: same token works
    ok = _mv_render(w, token)
    assert ok.exit_code == 0, ok.output
    assert quote_consumed(_data_dir(), _sig(token))
    assert len(w.eleven.calls) == 2


def test_multivoice_gpu_busy_leaves_token_reusable(mv_world):
    w = mv_world
    token = _mv_token(w)
    other_process = FileLockHandle(_data_dir() / "gpu.lock")  # e.g. the API server
    assert other_process.try_acquire()
    try:
        refused = _mv_render(w, token)
    finally:
        other_process.release()
    assert refused.exit_code != 0 and "holds the GPU" in refused.output
    assert not quote_consumed(_data_dir(), _sig(token))
    assert w.eleven.calls == []

    ok = _mv_render(w, token)  # card free again: the same approval renders
    assert ok.exit_code == 0, ok.output
    assert quote_consumed(_data_dir(), _sig(token))


def test_multivoice_merge_identity_refusal_leaves_token_reusable(mv_world):
    w = mv_world
    # a prior FREE full render under a different cast leaves a multivoice manifest behind
    _write_assignment(w.out, w.bid, _assignment().model_copy(update={"assignments": {}}))
    free = CliRunner().invoke(main, _mv("render", w, "--multivoice"))
    assert free.exit_code == 0, free.output

    _write_assignment(w.out, w.bid, _assignment())  # alice -> paid voice again
    token = _mv_token(w, "--chapter", "1")
    refused = _mv_render(w, token, "--chapter", "1")
    assert refused.exit_code != 0 and "chapter-subset" in refused.output
    assert not quote_consumed(_data_dir(), _sig(token))
    assert w.eleven.calls == []

    for name in ("manifest.json", "manifest.multi.json"):  # follow the refusal's advice
        (w.out / w.bid / name).unlink(missing_ok=True)
    ok = _mv_render(w, token, "--chapter", "1")
    assert ok.exit_code == 0, ok.output
    assert quote_consumed(_data_dir(), _sig(token))
    assert len(w.eleven.calls) == 1  # chapter 1's one alice line


def test_multivoice_paid_synthesis_only_after_single_consumption(mv_world, monkeypatch):
    w = mv_world
    token = _mv_token(w)
    sig, data_dir = _sig(token), _data_dir()
    consumes = []

    def counting_consume(*args, **kwargs):
        consumes.append(args)
        return _consume_quote(*args, **kwargs)

    monkeypatch.setattr("seiyuu.render.gate._consume_quote", counting_consume)
    original = type(w.eleven)._synthesize_native

    def guarded(self, text, voice, settings):
        assert quote_consumed(data_dir, sig), "paid synthesis before the token was consumed"
        return original(self, text, voice, settings)

    monkeypatch.setattr(type(w.eleven), "_synthesize_native", guarded)
    ok = _mv_render(w, token)
    assert ok.exit_code == 0, ok.output
    assert len(w.eleven.calls) == 2 and len(consumes) == 1
