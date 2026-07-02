"""M6a money gate: signed cost quotes, drift refusal, ceiling, single-voice pre-flight."""

from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

import seiyuu.engines
from factories import make_book
from fake_engine import FakeEngine
from seiyuu.cli import _pass_cost_gate, main
from seiyuu.render import (
    CostEstimate,
    CostGateError,
    CostQuote,
    check_ceiling,
    estimate_render_cost,
    estimate_render_cost_single,
    hash_assignment,
    issue_quote,
    render_book,
    render_book_multivoice,
    verify_quote,
)
from seiyuu.settings import get_settings
from test_render_cost_gate import FakeElevenEngine, _assignment, _library, _patch, _report

# --- quote issue/verify ---


def make_est(total=1.5, fingerprint="fp-abc", paid=3):
    return CostEstimate(
        total_usd=total,
        paid_segments=paid,
        cached_segments=0,
        free_segments=2,
        fingerprint=fingerprint,
    )


def issue(tmp_path, **overrides):
    kwargs = dict(
        book_id="book-1", chapters=(1, 2), assignment_hash="ah-1",
        max_usd=25.0, ttl_seconds=900, data_dir=tmp_path, now=1000.0,
    )  # fmt: skip
    est = overrides.pop("est", make_est())
    kwargs.update(overrides)
    return issue_quote(est, **kwargs)


def verify(tmp_path, quote, **overrides):
    kwargs = dict(
        book_id="book-1", chapters=(1, 2), fingerprint="fp-abc", assignment_hash="ah-1",
        recomputed_total_usd=1.5, max_usd=25.0, data_dir=tmp_path, now=1100.0,
    )  # fmt: skip
    kwargs.update(overrides)
    verify_quote(quote, **kwargs)


def test_quote_token_roundtrip_verifies(tmp_path):
    quote = issue(tmp_path)
    decoded = CostQuote.decode(quote.encode())
    assert decoded == quote
    verify(tmp_path, decoded)  # no raise


def test_chapters_bind_order_insensitively(tmp_path):
    quote = issue(tmp_path, chapters=(2, 1))
    verify(tmp_path, quote, chapters=(1, 2))  # no raise


def test_tampered_quote_fails_signature(tmp_path):
    quote = issue(tmp_path)
    forged = quote.model_copy(update={"total_usd": 0.01})
    with pytest.raises(CostGateError, match="signature"):
        verify(tmp_path, forged, recomputed_total_usd=0.01)


def test_foreign_signing_key_fails(tmp_path):
    quote = issue(tmp_path / "server_a")
    with pytest.raises(CostGateError, match="signature"):
        verify(tmp_path / "server_b", quote)


def test_expired_quote_refused(tmp_path):
    quote = issue(tmp_path, now=1000.0, ttl_seconds=900)
    with pytest.raises(CostGateError, match="expired"):
        verify(tmp_path, quote, now=1000.0 + 901)


def test_scope_mismatches_refused(tmp_path):
    quote = issue(tmp_path)
    with pytest.raises(CostGateError, match="issued for book"):
        verify(tmp_path, quote, book_id="other-book")
    with pytest.raises(CostGateError, match="chapter selection"):
        verify(tmp_path, quote, chapters=(1, 2, 3))
    with pytest.raises(CostGateError, match="assignment changed"):
        verify(tmp_path, quote, assignment_hash="ah-2")
    with pytest.raises(CostGateError, match="segments changed"):
        verify(tmp_path, quote, fingerprint="fp-other")


def test_upward_cost_drift_refused_downward_ok(tmp_path):
    quote = issue(tmp_path)  # quoted $1.50
    with pytest.raises(CostGateError, match="drifted upward"):
        verify(tmp_path, quote, recomputed_total_usd=2.0)
    verify(tmp_path, quote, recomputed_total_usd=0.75)  # cache grew: cheaper is fine


def test_ceiling_enforced_at_issue_and_verify(tmp_path):
    with pytest.raises(CostGateError, match="ceiling"):
        issue(tmp_path, est=make_est(total=30.0))
    quote = issue(tmp_path, est=make_est(total=20.0))
    with pytest.raises(CostGateError, match="ceiling"):  # ceiling lowered since issue
        verify(tmp_path, quote, recomputed_total_usd=20.0, max_usd=10.0)
    check_ceiling(25.0, 25.0)  # equality is not over


def test_malformed_tokens_refused():
    for bad in ("garbage", "cq1.!!!not-base64!!!", "xx9." + "AAAA"):
        with pytest.raises(CostGateError, match="malformed"):
            CostQuote.decode(bad)


def test_signing_key_created_once_and_reused(tmp_path):
    issue(tmp_path)
    key_file = tmp_path / "cost_signing.key"
    assert key_file.is_file()
    key = key_file.read_text(encoding="utf-8").strip()
    assert len(key) == 64 and int(key, 16) is not None  # 32 random bytes, hex
    issue(tmp_path)
    assert key_file.read_text(encoding="utf-8").strip() == key


def test_token_is_single_use(tmp_path):
    """One approval, one render: a replay (M6b double-click, cache wipe, second output
    dir) must refuse instead of billing the quoted amount again."""
    quote = issue(tmp_path)
    verify(tmp_path, quote)
    with pytest.raises(CostGateError, match="already used"):
        verify(tmp_path, quote)


def test_verify_consume_false_is_a_dry_run(tmp_path):
    quote = issue(tmp_path)
    verify(tmp_path, quote, consume=False)
    verify(tmp_path, quote, consume=False)  # dry runs don't burn the token
    verify(tmp_path, quote)  # the real verify still works exactly once
    with pytest.raises(CostGateError, match="already used"):
        verify(tmp_path, quote)


def test_refused_verify_does_not_burn_the_token(tmp_path):
    quote = issue(tmp_path)
    with pytest.raises(CostGateError, match="drifted upward"):
        verify(tmp_path, quote, recomputed_total_usd=99.0)
    verify(tmp_path, quote)  # still spendable after a refusal


def test_duplicate_chapters_are_deduped(tmp_path):
    quote = issue(tmp_path, chapters=(2, 2, 1))
    verify(tmp_path, quote, chapters=(1, 2))  # same work, no wrong refusal


def test_hostile_non_hex_sig_is_malformed_not_a_crash(tmp_path):
    """A non-ASCII sig would make hmac.compare_digest raise TypeError — an unhandled 500
    in M6b where the token comes from an untrusted client. Decode must refuse it."""
    quote = issue(tmp_path)
    hostile = quote.model_copy(update={"sig": "ü" * 64})  # model_copy skips validation
    with pytest.raises(CostGateError, match="malformed"):
        CostQuote.decode(hostile.encode())


def test_paid_quote_requires_a_fingerprint(tmp_path):
    with pytest.raises(CostGateError, match="fingerprint"):
        issue(tmp_path, est=make_est(fingerprint=""))


def test_price_floor_rejects_a_gate_disabling_zero():
    from pydantic import ValidationError

    from seiyuu.settings import Settings

    with pytest.raises(ValidationError):
        Settings(elevenlabs_price_per_1k_chars=0)


def test_hash_assignment_binds_voice_changes():
    a = hash_assignment(_assignment())
    changed = _assignment().model_copy(update={"assignments": {"alice": "narrator_v"}})
    assert hash_assignment(_assignment()) == a
    assert hash_assignment(changed) != a


# --- the CLI gate helper ---


def make_cfg(tmp_path, max_usd=25.0):
    return SimpleNamespace(render_max_usd=max_usd, data_dir=tmp_path)


def test_gate_free_render_needs_no_approval(tmp_path):
    cfg = make_cfg(tmp_path)
    est = make_est(total=0.0, paid=0)
    approved = _pass_cost_gate(
        cfg, est, book_id="b", chapters=(), assignment_hash=None,
        confirm_cost=False, cost_token=None,
    )  # fmt: skip
    assert approved is None  # pipeline safety net stays armed


def test_gate_confirm_cost_under_ceiling_allows(tmp_path):
    approved = _pass_cost_gate(
        make_cfg(tmp_path), make_est(), book_id="b", chapters=(), assignment_hash=None,
        confirm_cost=True, cost_token=None,
    )  # fmt: skip
    assert approved == pytest.approx(1.5)  # the confirmed figure becomes the spend cap


def test_gate_ceiling_blocks_even_with_confirm_cost(tmp_path):
    with pytest.raises(click.ClickException, match="ceiling"):
        _pass_cost_gate(
            make_cfg(tmp_path, max_usd=1.0), make_est(total=5.0),
            book_id="b", chapters=(), assignment_hash=None,
            confirm_cost=True, cost_token=None,
        )  # fmt: skip


def test_gate_token_flow(tmp_path):
    cfg = make_cfg(tmp_path)
    est = make_est(fingerprint="fp-1")
    quote = issue_quote(
        est, book_id="b", chapters=(), assignment_hash="ah",
        max_usd=25.0, ttl_seconds=900, data_dir=tmp_path,
    )  # fmt: skip
    approved = _pass_cost_gate(
        cfg, est, book_id="b", chapters=(), assignment_hash="ah",
        confirm_cost=False, cost_token=quote.encode(),
    )  # fmt: skip
    assert approved == pytest.approx(1.5)  # the quoted figure becomes the spend cap

    drifted = make_est(fingerprint="fp-2")  # attribution changed since the token was issued
    with pytest.raises(click.ClickException, match="re-run estimate-cost"):
        _pass_cost_gate(
            cfg, drifted, book_id="b", chapters=(), assignment_hash="ah",
            confirm_cost=False, cost_token=quote.encode(),
        )  # fmt: skip


# --- estimators: fingerprint semantics + the single-voice pre-flight ---


def test_single_voice_free_engine_estimates_zero(tmp_path):
    est = estimate_render_cost_single(make_book(), FakeEngine(), "test_voice", tmp_path / "out")
    assert est.total_usd == 0.0 and est.paid_segments == 0
    assert est.free_segments == 5  # every speakable block, none cached


def test_single_voice_paid_estimate_counts_and_survives_caching(tmp_path):
    engine = FakeElevenEngine()
    out = tmp_path / "out"
    est = estimate_render_cost_single(make_book(), engine, "voice_x", out)
    assert est.paid_segments == 5 and est.total_usd > 0

    render_book(make_book(), engine, "voice_x", out, allow_paid=True)
    est2 = estimate_render_cost_single(make_book(), engine, "voice_x", out)
    assert est2.total_usd == 0.0 and est2.cached_segments == 5
    assert est2.fingerprint == est.fingerprint  # cache growth never drifts identity


def test_multivoice_fingerprint_tracks_work_not_cache(tmp_path, monkeypatch):
    from seiyuu.gpu import GpuResourceManager

    _patch(monkeypatch, FakeElevenEngine())
    lib, out = _library(tmp_path), tmp_path / "out"
    est = estimate_render_cost(_report(), make_book(), lib, _assignment(), out)
    assert est.fingerprint

    render_book_multivoice(
        _report(), make_book(), lib, _assignment(), out,
        gpu=GpuResourceManager(), allow_paid=True,
    )  # fmt: skip
    est_cached = estimate_render_cost(_report(), make_book(), lib, _assignment(), out)
    assert est_cached.total_usd == 0.0
    assert est_cached.fingerprint == est.fingerprint  # same work, now cached

    reassigned = _assignment().model_copy(update={"assignments": {"alice": "narrator_v"}})
    est_moved = estimate_render_cost(_report(), make_book(), lib, reassigned, out)
    assert est_moved.fingerprint != est.fingerprint  # paid work itself changed


def test_render_budget_cap_blocks_mid_run_overspend(tmp_path):
    """allow_paid must not be a blanket: if the cache changes after approval, the render
    refuses when cumulative paid spend would pass the approved figure."""
    from seiyuu.render import RenderError

    engine = FakeElevenEngine()
    with pytest.raises(RenderError, match="approved budget"):
        render_book(
            make_book(), engine, "voice_x", tmp_path / "out",
            allow_paid=True, max_paid_usd=0.001,
        )  # fmt: skip
    assert not (tmp_path / "out" / "manifest.json").exists()

    render_book(  # a sufficient budget renders normally
        make_book(), engine, "voice_x", tmp_path / "out2", allow_paid=True, max_paid_usd=5.0
    )
    assert (tmp_path / "out2" / "manifest.json").is_file()


def test_voice_dir_meta_mismatch_fails_loudly(tmp_path):
    """A hand-renamed voice dir would silently split the estimator's and render's
    SegmentKeys (the gate's core parity) — the library must refuse to load it."""
    import shutil

    from seiyuu.voices.library import VoiceLibraryError

    lib = _library(tmp_path)
    shutil.copytree(lib.dir_for("narrator_v"), lib.dir_for("renamed_v"))
    with pytest.raises(VoiceLibraryError, match="renamed or copied"):
        lib.load("renamed_v")


# --- CLI: the single-voice paid path now has a real gate ---


@pytest.fixture
def ingested_book(tmp_path):
    from seiyuu.ingest.epub import write_normalized

    book = make_book()
    write_normalized(book, tmp_path / "books")
    return tmp_path / "books", tmp_path / "output", book.book_meta.book_id


def _render_args(books_dir, output_dir, book_id, *extra):
    return [
        "render", book_id, "--voice", "voice_x",
        "--books-dir", str(books_dir), "--output-dir", str(output_dir), *extra,
    ]  # fmt: skip


def test_cli_single_voice_paid_refuses_without_confirmation(ingested_book, monkeypatch):
    books_dir, output_dir, book_id = ingested_book
    fake = FakeElevenEngine()
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: fake)

    result = CliRunner().invoke(main, _render_args(books_dir, output_dir, book_id))
    assert result.exit_code != 0
    assert "cost estimate" in result.output  # the M5 gap: there was no estimate here at all
    assert fake.calls == []  # and no paid synthesis happened
    assert not (output_dir / book_id / "manifest.json").exists()


def test_cli_single_voice_paid_interactive_yes_renders(ingested_book, monkeypatch):
    books_dir, output_dir, book_id = ingested_book
    fake = FakeElevenEngine()
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: fake)

    result = CliRunner().invoke(main, _render_args(books_dir, output_dir, book_id), input="y\n")
    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 5
    assert (output_dir / book_id / "manifest.json").is_file()


def test_cli_single_voice_confirm_cost_flag_renders(ingested_book, monkeypatch):
    books_dir, output_dir, book_id = ingested_book
    fake = FakeElevenEngine()
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: fake)

    result = CliRunner().invoke(
        main, _render_args(books_dir, output_dir, book_id, "--confirm-cost")
    )
    assert result.exit_code == 0, result.output
    assert "cost estimate" in result.output


def test_cli_single_voice_token_end_to_end(ingested_book, monkeypatch, tmp_path):
    """estimate-cost --voice X --token must mint a token render --cost-token accepts —
    the flow the multivoice-only issuer used to brick with a misleading refusal."""
    books_dir, output_dir, book_id = ingested_book
    fake = FakeElevenEngine()
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: fake)
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path / "data")
    runner = CliRunner()

    est = runner.invoke(
        main,
        [
            "estimate-cost",
            book_id,
            "--voice",
            "voice_x",
            "--token",
            "--books-dir",
            str(books_dir),
            "--output-dir",
            str(output_dir),
        ],  # fmt: skip
    )
    assert est.exit_code == 0, est.output
    token = next(line for line in est.output.splitlines() if line.startswith("cq1."))

    ok = runner.invoke(main, _render_args(books_dir, output_dir, book_id, "--cost-token", token))
    assert ok.exit_code == 0, ok.output
    assert (output_dir / book_id / "manifest.json").is_file()
    assert len(fake.calls) == 5


def test_cli_ceiling_blocks_even_confirm_cost(ingested_book, monkeypatch):
    books_dir, output_dir, book_id = ingested_book
    fake = FakeElevenEngine()
    monkeypatch.setattr(seiyuu.engines, "get_engine", lambda engine_id, **kw: fake)
    monkeypatch.setattr(get_settings(), "render_max_usd", 0.0001)

    result = CliRunner().invoke(
        main, _render_args(books_dir, output_dir, book_id, "--confirm-cost")
    )
    assert result.exit_code != 0
    assert "ceiling" in result.output
    assert fake.calls == []


# --- CLI: estimate-cost --token -> render --cost-token, end to end ---


@pytest.fixture
def multivoice_world(tmp_path, monkeypatch):
    book = make_book()
    bid = book.book_meta.book_id
    books, out = tmp_path / "books", tmp_path / "output"
    (books / bid).mkdir(parents=True)
    (books / bid / "normalized.json").write_text(book.model_dump_json(), encoding="utf-8")
    (books / bid / "attribution.json").write_text(_report().model_dump_json(), encoding="utf-8")
    (out / bid).mkdir(parents=True)
    (out / bid / "assignments.json").write_text(_assignment().model_dump_json(), encoding="utf-8")
    lib = _library(tmp_path)
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path / "data")
    return books, out, lib.voices_dir, bid


def _mv_args(command, books, out, voices, bid, *extra):
    return [
        command, bid, "--books-dir", str(books), "--output-dir", str(out),
        "--voices-dir", str(voices), *extra,
    ]  # fmt: skip


def test_cli_token_end_to_end_and_drift_refusal(multivoice_world, monkeypatch):
    import numpy as np

    from seiyuu.engines import AudioFile
    from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceMeta

    books, out, voices, bid = multivoice_world
    _patch(monkeypatch, FakeElevenEngine())
    runner = CliRunner()

    est = runner.invoke(main, _mv_args("estimate-cost", books, out, voices, bid, "--token"))
    assert est.exit_code == 0, est.output
    token = next(line for line in est.output.splitlines() if line.startswith("cq1."))

    # the world drifts: alice moves to a DIFFERENT paid voice after the token was issued
    # (still costs money, but it is not the work the user approved)
    lib = VoiceLibrary(voices)
    lib.save(
        VoiceMeta(voice_id="elena2_y", name="Elena Two", kind=VoiceKind.CLONED,
                  engine="elevenlabs", reference_audio="reference.wav", consent_attested=True)
    )  # fmt: skip
    AudioFile(samples=np.zeros(2400, dtype=np.float32)).save(lib.reference_path("elena2_y"))
    reassigned = _assignment().model_copy(update={"assignments": {"alice": "elena2_y"}})
    (out / bid / "assignments.json").write_text(reassigned.model_dump_json(), encoding="utf-8")
    refused = runner.invoke(
        main,
        _mv_args("render", books, out, voices, bid, "--multivoice", "--cost-token", token),
    )
    assert refused.exit_code != 0
    assert "re-run estimate-cost" in refused.output
    assert not (out / bid / "manifest.json").exists()  # nothing rendered, nothing billed

    # restore the world the token was issued against: the render goes through unprompted
    (out / bid / "assignments.json").write_text(_assignment().model_dump_json(), encoding="utf-8")
    ok = runner.invoke(
        main,
        _mv_args("render", books, out, voices, bid, "--multivoice", "--cost-token", token),
    )
    assert ok.exit_code == 0, ok.output
    assert (out / bid / "manifest.json").is_file()
