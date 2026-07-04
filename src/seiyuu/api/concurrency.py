"""Heavy-work gate + audition slot (scoping doc section 1, lifespan step 3).

The gate is one process-level lock around GPU-heavy activity: the attribute/render/
warmup handlers (assemble/master are pure ffmpeg and don't take it) and GPU auditions.
Auditions acquire NON-blockingly — busy means an immediate 409, never a stalled request
thread; job handlers acquire blockingly — a job that starts during an in-flight
audition waits at most one warm-engine synthesis (bounded seconds, accepted at
sign-off). Audition-vs-audition exclusion lives in the separate :class:`AuditionSlot`
so a cloud audition (no GPU) is never serialized behind a gate-holding job.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

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
