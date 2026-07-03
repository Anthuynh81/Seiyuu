"""Render cost gate (M6a): signed, short-lived quotes bound to the exact paid work.

The M5 gate was a bool: estimate, ask, pass ``allow_paid=True``. That is sound for one
process at one terminal, but over HTTP it is a TOCTOU — what the user approved and what
render bills can drift (attribution re-run, assignment edited, cache wiped, price
changed). This module makes the approval an artifact:

- ``issue_quote`` signs (book, chapters, paid-segment fingerprint, assignment hash,
  total) with a persistent server-side key and stamps an expiry.
- ``verify_quote`` re-checks everything at render time against a FRESH estimate and
  refuses on any drift upward, scope change, expiry, or tampering — with a message that
  says exactly what changed.
- ``check_ceiling`` bounds any single approval by ``settings.render_max_usd`` on every
  path, token or not: no flag can authorize an unbounded spend.

The fingerprint is a hash over the SegmentKey hashes of every PAID-engine segment in
scope, cached or not — cache growth (cheaper) never drifts it, while any change to
text, voice, settings, or seed does. Quotes are SINGLE-USE: a successful verify consumes
the signature in a small ledger (``data_dir/quotes.db``), so one approval can never bill
twice (double-click, replay after a cache wipe, a second output dir). The render loop
additionally caps its cumulative paid spend at the approved total, so a cache that
changes mid-run fails loudly instead of overspending. The signing key lives at
``data_dir/cost_signing.key`` and is never logged; deleting it invalidates outstanding
quotes, which only means re-running the estimate.
"""

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from seiyuu.render.pipeline import CostEstimate
from seiyuu.repository import atomic_write_text

_KEY_NAME = "cost_signing.key"
_LEDGER_NAME = "quotes.db"
_TOKEN_PREFIX = "cq1."  # cost quote, format v1
_USD_EPSILON = 1e-6

# A full-book render is a long GPU job; above this many speakable segments both the CLI
# (interactive confirm) and the API (409 full_render_confirmation_required) require an
# explicit go-ahead. Free renders aren't money-gated, so this is the only whole-book stop.
FULL_RENDER_CONFIRM_BLOCKS = 300


class CostGateError(Exception):
    """Loud gate refusal: the message states exactly why and what to do about it."""


class CostQuote(BaseModel):
    """A signed approval for one specific render's paid work."""

    book_id: str
    chapters: tuple[int, ...]  # sorted; () = the whole book
    fingerprint: str  # paid-segment identity (see module docstring)
    assignment_hash: str | None  # multivoice binding; None for single-voice
    total_usd: float
    paid_segments: int
    issued_at: float
    expires_at: float
    # hex-only so a hostile token can't smuggle non-ASCII into hmac.compare_digest
    # (which raises TypeError, not a comparison failure, on non-ASCII str input)
    sig: str = Field(default="", pattern=r"^[0-9a-f]{0,64}$")

    def encode(self) -> str:
        """Opaque transportable token (CLI flag / API field)."""
        raw = self.model_dump_json().encode("utf-8")
        return _TOKEN_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")

    @classmethod
    def decode(cls, token: str) -> "CostQuote":
        if not token.startswith(_TOKEN_PREFIX):
            raise CostGateError("malformed cost token (not a cq1 token); re-run estimate-cost")
        try:
            raw = base64.urlsafe_b64decode(token[len(_TOKEN_PREFIX) :].encode("ascii"))
            return cls.model_validate_json(raw)
        except (ValueError, ValidationError) as exc:
            raise CostGateError(f"malformed cost token: {exc}") from exc


def hash_assignment(assignment) -> str:
    """Canonical hash of a VoiceAssignment, binding a quote to who-speaks-with-which-voice."""
    payload = json.dumps(assignment.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_ceiling(total_usd: float, max_usd: float) -> None:
    """Refuse any single paid render above the configured ceiling — on EVERY path."""
    if total_usd > max_usd + _USD_EPSILON:
        raise CostGateError(
            f"estimated ${total_usd:.2f} exceeds the render_max_usd ceiling (${max_usd:.2f}); "
            f"set RENDER_MAX_USD in .env to authorize a render this large"
        )


def _signing_key(data_dir: Path) -> bytes:
    """The persistent HMAC key (created on first use). If two processes race the creation,
    quotes signed with the losing key fail verification — the remedy is a re-estimate."""
    path = Path(data_dir) / _KEY_NAME
    if not path.is_file():
        atomic_write_text(path, secrets.token_hex(32))
    return bytes.fromhex(path.read_text(encoding="utf-8").strip())


def _sign(quote: CostQuote, key: bytes) -> str:
    payload = json.dumps(quote.model_dump(exclude={"sig"}), sort_keys=True, separators=(",", ":"))
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def issue_quote(
    est: CostEstimate,
    *,
    book_id: str,
    chapters: tuple[int, ...],
    assignment_hash: str | None,
    max_usd: float,
    ttl_seconds: int,
    data_dir: Path,
    now: float | None = None,
) -> CostQuote:
    """Sign a quote for the estimate's paid work. Raises above the ceiling."""
    check_ceiling(est.total_usd, max_usd)
    if est.total_usd > 0 and not est.fingerprint:
        raise CostGateError(
            "refusing to sign a paid quote without a segment fingerprint "
            "(the estimate did not identify the paid work)"
        )
    now = time.time() if now is None else now
    quote = CostQuote(
        book_id=book_id,
        chapters=tuple(sorted(set(chapters))),
        fingerprint=est.fingerprint,
        assignment_hash=assignment_hash,
        total_usd=est.total_usd,
        paid_segments=est.paid_segments,
        issued_at=now,
        expires_at=now + ttl_seconds,
    )
    return quote.model_copy(update={"sig": _sign(quote, _signing_key(data_dir))})


def verify_quote(
    quote: CostQuote,
    *,
    book_id: str,
    chapters: tuple[int, ...],
    fingerprint: str,
    assignment_hash: str | None,
    recomputed_total_usd: float,
    max_usd: float,
    data_dir: Path,
    now: float | None = None,
    consume: bool = True,
) -> None:
    """Check a quote against a FRESH estimate of the same render; raise on any problem.

    Order matters: authenticity first (no field is trusted before the signature checks
    out), then expiry, then scope bindings, then the money itself. Consumption comes
    LAST — a refused verify never burns the token. With ``consume=True`` (the default)
    a successful verify records the signature in the single-use ledger, and a second
    verify of the same quote refuses; pass ``consume=False`` only for dry-run checks
    that will not lead to spending.
    """
    if not hmac.compare_digest(_sign(quote, _signing_key(data_dir)), quote.sig):
        raise CostGateError(
            "cost token signature invalid (tampered with, or issued by a different "
            "server/key); re-run estimate-cost"
        )
    now = time.time() if now is None else now
    if now > quote.expires_at:
        raise CostGateError("cost token expired; re-run estimate-cost")
    if quote.book_id != book_id:
        raise CostGateError(f"cost token was issued for book {quote.book_id!r}, not {book_id!r}")
    if quote.chapters != tuple(sorted(set(chapters))):
        raise CostGateError(
            "cost token covers a different chapter selection; re-run estimate-cost "
            "with the chapters you intend to render"
        )
    if quote.assignment_hash != assignment_hash:
        raise CostGateError("voice assignment changed since the estimate; re-run estimate-cost")
    if quote.fingerprint != fingerprint:
        raise CostGateError(
            "the paid segments changed since the estimate (attribution, voices, or "
            "settings differ); re-run estimate-cost"
        )
    if recomputed_total_usd > quote.total_usd + _USD_EPSILON:
        raise CostGateError(
            f"cost drifted upward since the estimate (quoted ${quote.total_usd:.2f}, "
            f"now ${recomputed_total_usd:.2f} — cache or pricing changed); re-run estimate-cost"
        )
    check_ceiling(quote.total_usd, max_usd)
    if consume:
        _consume_quote(data_dir, quote.sig, quote.expires_at, now)


def quote_consumed(data_dir: Path, sig: str) -> bool:
    """Read-only ledger probe for enqueue-time dry-runs (M6b). ``verify_quote(consume=
    False)`` deliberately never touches the ledger, so a dry-run alone would PASS a
    spent token and defer the refusal to job start — this lets the API surface the
    immediate 402 instead. Best-effort by design: the atomic INSERT in
    ``_consume_quote`` remains the single-use enforcement; a token spent between this
    probe and job start still refuses there."""
    path = Path(data_dir) / _LEDGER_NAME
    if not path.is_file():
        return False
    conn = sqlite3.connect(path, timeout=5.0)
    try:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consumed_quotes'"
        ).fetchone()
        if table is None:
            return False
        return (
            conn.execute("SELECT 1 FROM consumed_quotes WHERE sig=?", (sig,)).fetchone() is not None
        )
    finally:
        conn.close()


def _consume_quote(data_dir: Path, sig: str, expires_at: float, now: float) -> None:
    """Single-use enforcement: atomically record the sig; a second verify refuses.

    The PRIMARY KEY insert is the atomicity — two concurrent verifies of the same quote
    can never both pass. Expired rows are purged opportunistically so the ledger stays
    tiny; a purged sig cannot be replayed because its quote is expired anyway."""
    path = Path(data_dir) / _LEDGER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS consumed_quotes "
            "(sig TEXT PRIMARY KEY, expires_at REAL NOT NULL)"
        )
        conn.execute("DELETE FROM consumed_quotes WHERE expires_at < ?", (now,))
        try:
            conn.execute(
                "INSERT INTO consumed_quotes (sig, expires_at) VALUES (?, ?)", (sig, expires_at)
            )
        except sqlite3.IntegrityError:
            raise CostGateError(
                "cost token already used (tokens are single-use); re-run estimate-cost "
                "for a new approval"
            ) from None
        conn.commit()
    finally:
        conn.close()
