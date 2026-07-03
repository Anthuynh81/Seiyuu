"""Heavy-work gate (scoping doc section 1, lifespan step 3).

One process-level lock around (a) job-handler execution on the runner thread and
(b) sync auditions. Auditions acquire NON-blockingly — busy means an immediate 409,
never a stalled request thread; job handlers acquire blockingly — a job that starts
during an in-flight audition waits at most one warm-engine synthesis (bounded seconds,
accepted in the scoping sign-off). This also closes audition-vs-audition: the second
concurrent audition gets an instant 409 ``audition_in_flight``.
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
