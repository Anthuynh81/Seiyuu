"""Heavy-work gate + audition slot (scoping doc section 1, lifespan step 3).

The gate is one process-level lock around GPU-heavy activity: the attribute/render/
warmup handlers (assemble/master are pure ffmpeg and don't take it) and GPU auditions.
Auditions acquire NON-blockingly — busy means an immediate 409, never a stalled request
thread; job handlers acquire blockingly — a job that starts during an in-flight
audition waits at most one warm-engine synthesis (bounded seconds, accepted at
sign-off). Audition-vs-audition exclusion lives in the separate :class:`AuditionSlot`
so a cloud audition (no GPU) is never serialized behind a gate-holding job.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seiyuu.engines import TTSEngine

AUDITION = "audition"
JOB = "job"


class HeavyWorkGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holder = ""

    @contextmanager
    def hold(self, holder: str = JOB) -> Iterator[None]:
        """Blocking acquire — job handlers on the runner thread."""
        with self._lock:
            self._holder = holder
            try:
                yield
            finally:
                self._holder = ""

    @contextmanager
    def try_hold(self, holder: str = AUDITION) -> Iterator[bool]:
        """Non-blocking acquire — sync request paths. Yields False when busy; the
        route replies 409 with the holder's identity."""
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._holder = holder
        try:
            yield acquired
        finally:
            if acquired:
                self._holder = ""
                self._lock.release()

    @property
    def holder(self) -> str:
        """Unsynchronized snapshot for display/refusal messages only."""
        return self._holder

    @property
    def audition_in_flight(self) -> bool:
        return self._holder == AUDITION


class AuditionSlot:
    """One audition at a time (scoping-doc refusal (c)) — deliberately SEPARATE from the
    heavy-work gate: a cloud audition holds no GPU, so it must not be refused (or worse,
    mislabeled ``audition_in_flight``) just because an attribute/render handler holds
    the gate for hours. The tested allowance — ElevenLabs auditions during attribution —
    only works with this split."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def try_hold(self) -> Iterator[bool]:
        acquired = self._lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()

    @property
    def in_flight(self) -> bool:
        return self._lock.locked()


@dataclass
class BorrowTicket:
    """Internal thread-coordination handle for one in-flight borrow (F1). Deliberately NOT
    a pydantic model and never serialized — it carries two :class:`threading.Event`
    handshakes. ``granted`` fires when the render lends its instance (delivered on
    ``engine``); ``done`` fires when the borrower has finished its one synthesis so the
    parked render may resume."""

    engine_id: str
    granted: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    engine: TTSEngine | None = None


class BorrowBroker:
    """Process-local rendezvous that lets a synchronous audition borrow a render's
    already-resident TTS engine BETWEEN render segments (F1), instead of the audition
    409'ing for the render's whole multi-hour duration.

    The render PUBLISHES its own resident instance and, at each cooperative yield point
    (the same ``check_cancel`` points the render already has), calls :meth:`serve`. If an
    audition is waiting on this exact engine, ``serve`` hands over the instance (fires
    ``granted``) and PARKS on ``done`` while the audition synthesizes exactly one segment,
    then resumes — so the render provably cannot run ahead of an admitted borrow. The
    micro-lock guards only microsecond field updates and is NEVER held across a synthesize.
    There is no queue: :class:`AuditionSlot` caps the system to one outstanding audition,
    so a single pending slot suffices.

    :class:`GpuResourceManager`, :class:`HeavyWorkGate`, and :class:`AuditionSlot` are
    untouched: the audition still synthesizes inside ``gpu.acquire`` (an identity no-op on
    the render's resident instance) as the manager-lock backstop.
    """

    def __init__(self, grant_timeout_s: float = 30.0) -> None:
        self._lock = threading.Lock()
        self._grant_timeout = grant_timeout_s
        self._lender_engine_id: str | None = None
        self._lender_engine: TTSEngine | None = None
        self._pending: BorrowTicket | None = None
        self._closed = False

    # -- render side ---------------------------------------------------------------------

    def publish(self, engine_id: str, engine: TTSEngine) -> None:
        """Advertise the engine the render can currently lend (its own resident instance)."""
        with self._lock:
            if self._closed:
                return
            self._lender_engine_id = engine_id
            self._lender_engine = engine

    def close(self) -> None:
        """Stop lending for good (render teardown). A request racing teardown resolves to
        an immediate None (soft retry), never an about-to-be-unloaded engine. Called BEFORE
        ``gpu.free_all()`` in the render's finally."""
        with self._lock:
            self._closed = True
            self._lender_engine_id = None
            self._lender_engine = None
            pending = self._pending
            self._pending = None
            if pending is not None:
                pending.engine = None
                pending.granted.set()  # wake a blocked wait_grant -> None (soft retry)

    def serve(self, engine_id: str, engine: TTSEngine) -> None:
        """Render thread at a yield point (called AFTER ``check_cancel``, and while holding
        NO ``gpu._lock`` — outside its per-segment acquire): if an audition is waiting on
        this engine, lend it and PARK until the audition signals done (bounded by the
        safety timeout), then resume. Otherwise return after one atomic read — the common
        no-audition cost, mirroring ``check_cancel``."""
        with self._lock:
            if self._closed:
                return
            pending = self._pending
            if pending is None or pending.engine_id != engine_id:
                return
            self._pending = None
            pending.engine = engine
            pending.granted.set()
        # micro-lock released BEFORE the wait: the borrower now synthesizes on `engine`.
        # Park until it signals done (or the safety timeout fires, after which the render
        # resumes and gpu.acquire serialization is the backstop).
        pending.done.wait(self._grant_timeout)

    # -- audition side -------------------------------------------------------------------

    def eligible(self, engine_id: str) -> bool:
        """Lock-free admission predicate (read under the enqueue mutex): is a live render
        currently lending this engine? A benign stale True degrades to a grant timeout ->
        soft retry, never a wrong load."""
        return (
            not self._closed
            and self._lender_engine_id == engine_id
            and self._lender_engine is not None
        )

    def request(self, engine_id: str) -> BorrowTicket:
        """Register the single outstanding borrow request (AuditionSlot caps to one)."""
        ticket = BorrowTicket(engine_id=engine_id)
        with self._lock:
            if self._closed:
                ticket.granted.set()  # nothing to lend -> wait_grant returns None at once
                return ticket
            self._pending = ticket
        return ticket

    def wait_grant(self, ticket: BorrowTicket, timeout: float) -> TTSEngine | None:
        """Block on ``granted`` (holding no lock — only AuditionSlot, up the stack) until
        the render lends its instance (returns it) or timeout/close (returns None)."""
        ticket.granted.wait(timeout)
        with self._lock:
            if self._pending is ticket:  # timed out before any serve matched us
                self._pending = None
            return ticket.engine

    def signal_done(self, ticket: BorrowTicket) -> None:
        """Borrower finished its one segment; release the parked render (or a no-op if the
        render already resumed on a safety timeout)."""
        ticket.done.set()
