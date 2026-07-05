"""F1 BorrowBroker rendezvous — deterministic two-thread unit tests, no real TTS.

The broker never calls engine methods, so a bare sentinel object stands in for the lent
engine. Every wait is bounded by an injected short timeout so a broken handshake fails the
test fast instead of hanging. These exercise the exact grant/park/done sequencing the
render hot path relies on.
"""

import threading
import time

from seiyuu.api.concurrency import BorrowBroker

ENGINE = object()  # sentinel: the broker only ever hands this reference back
OTHER = object()


def _wait(pred, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.001)
    raise AssertionError("condition not met within timeout")


def test_serve_returns_instantly_with_no_pending_request() -> None:
    broker = BorrowBroker(grant_timeout_s=5.0)
    broker.publish("kokoro", ENGINE)
    t0 = time.monotonic()
    broker.serve("kokoro", ENGINE)  # nobody waiting -> one atomic read, no park
    assert time.monotonic() - t0 < 0.5


def test_render_parks_until_signal_done() -> None:
    broker = BorrowBroker(grant_timeout_s=5.0)
    broker.publish("kokoro", ENGINE)
    granted: list[object] = []
    release = threading.Event()  # the test decides when the "synthesis" finishes

    def audition() -> None:
        ticket = broker.request("kokoro")
        granted.append(broker.wait_grant(ticket, 5.0))
        release.wait(5.0)  # hold the borrow so we can observe the render parked
        broker.signal_done(ticket)

    a = threading.Thread(target=audition)
    a.start()
    _wait(lambda: broker._pending is not None)  # request is registered

    served = threading.Event()

    def render() -> None:
        broker.serve("kokoro", ENGINE)  # grants, then PARKS on done
        served.set()

    r = threading.Thread(target=render)
    r.start()

    _wait(lambda: granted)  # the audition received the lent instance...
    assert granted[0] is ENGINE
    time.sleep(0.1)
    assert not served.is_set()  # ...but the render is still parked, not running ahead

    release.set()  # audition signals done -> render resumes
    r.join(2.0)
    a.join(2.0)
    assert served.is_set()
    assert not r.is_alive() and not a.is_alive()


def test_mismatched_engine_id_never_grants() -> None:
    broker = BorrowBroker(grant_timeout_s=5.0)
    broker.publish("chatterbox", ENGINE)
    result: list[object] = []

    def audition() -> None:
        ticket = broker.request("kokoro")
        result.append(broker.wait_grant(ticket, 0.3))

    a = threading.Thread(target=audition)
    a.start()
    _wait(lambda: broker._pending is not None)
    broker.serve("chatterbox", OTHER)  # a different engine must not satisfy a kokoro wait
    a.join(2.0)
    assert result == [None]  # timed out, never granted


def test_close_during_pending_wait_yields_none() -> None:
    broker = BorrowBroker(grant_timeout_s=5.0)
    broker.publish("kokoro", ENGINE)
    result: list[object] = []

    def audition() -> None:
        ticket = broker.request("kokoro")
        result.append(broker.wait_grant(ticket, 5.0))

    a = threading.Thread(target=audition)
    a.start()
    _wait(lambda: broker._pending is not None)
    broker.close()  # render teardown while a borrow is parked
    a.join(2.0)
    assert not a.is_alive()
    assert result == [None]
    assert broker.eligible("kokoro") is False  # nothing lendable after close


def test_grant_timeout_fires_deterministically() -> None:
    broker = BorrowBroker(grant_timeout_s=5.0)
    broker.publish("kokoro", ENGINE)
    ticket = broker.request("kokoro")
    t0 = time.monotonic()
    engine = broker.wait_grant(ticket, 0.2)  # no serve() ever comes
    elapsed = time.monotonic() - t0
    assert engine is None
    assert 0.15 < elapsed < 2.0
    broker.serve("kokoro", ENGINE)  # a late serve finds no pending and returns at once


def test_serve_resumes_on_done_safety_timeout() -> None:
    """A borrower that gets the grant but never signals done (e.g. client disconnect) must
    not park the render forever — serve()'s done-wait is bounded by the safety timeout."""
    broker = BorrowBroker(grant_timeout_s=0.2)
    broker.publish("kokoro", ENGINE)
    got: list[object] = []

    def borrower() -> None:
        ticket = broker.request("kokoro")
        got.append(broker.wait_grant(ticket, 1.0))
        # deliberately never signal_done

    b = threading.Thread(target=borrower)
    b.start()
    _wait(lambda: broker._pending is not None)
    t0 = time.monotonic()
    broker.serve("kokoro", ENGINE)  # grants, parks on done, then times out
    elapsed = time.monotonic() - t0
    b.join(2.0)
    assert got == [ENGINE]
    assert 0.15 < elapsed < 2.0  # resumed via the safety timeout, not hung


def test_repeated_borrows_are_serial_and_clean() -> None:
    """Back-to-back borrows (as AuditionSlot serializes them) each rendezvous cleanly."""
    broker = BorrowBroker(grant_timeout_s=5.0)
    broker.publish("kokoro", ENGINE)
    for _ in range(5):
        results: list[object] = []

        def audition(results=results) -> None:
            ticket = broker.request("kokoro")
            eng = broker.wait_grant(ticket, 2.0)
            results.append(eng)
            broker.signal_done(ticket)

        a = threading.Thread(target=audition)
        a.start()
        _wait(lambda: broker._pending is not None)
        broker.serve("kokoro", ENGINE)
        a.join(2.0)
        assert results == [ENGINE]
        assert broker._pending is None  # slot cleared for the next borrow
