"""GPU resource manager — serializes the single 8GB GPU across heavy models.

A TTS engine OR the local LLM may be GPU-resident, never both (they would OOM the card).
Consumers acquire the GPU through this manager; when a DIFFERENT consumer acquires, the
resident one is `unload()`-ed first to free VRAM. Release is LAZY — a model stays resident
after its work so back-to-back use is cheap, and is freed only when a competitor acquires
(or `free_all()` at teardown).

The manager imports NO engine/LLM SDK: consumers register themselves by passing `self` to
`acquire()` (import direction is consumer -> manager). `acquire()` holds a lock for the whole
`with` body, so two consumers can never use the GPU concurrently. It is NOT reentrant —
acquire() calls must never nest. The render loops acquire per synthesis unit (re-acquiring
the resident consumer is a cheap no-op) because a multi-voice render switches engines
segment to segment; renders free_all() at the end only if they actually acquired.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Protocol, runtime_checkable


@runtime_checkable
class GpuConsumer(Protocol):
    def unload(self) -> None:
        """Free this consumer's GPU memory. Default no-op for non-GPU consumers."""


class GpuResourceManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resident: GpuConsumer | None = None
        self._resident_name = ""

    @contextmanager
    def acquire(self, consumer: GpuConsumer, name: str) -> Iterator[None]:
        """Hold the GPU for `consumer` across the `with` body, unloading any other resident."""
        with self._lock:
            if self._resident is not None and self._resident is not consumer:
                self._free_resident()
            self._resident, self._resident_name = consumer, name
            yield

    def _free_resident(self) -> None:
        if self._resident is None:
            return
        self._resident.unload()  # propagate errors loudly — a silent OOM later is worse
        self._resident, self._resident_name = None, ""

    def free_all(self) -> None:
        """Unload the resident model (teardown / explicit handoff)."""
        with self._lock:
            self._free_resident()

    @property
    def resident(self) -> str | None:
        return self._resident_name or None

    def holds(self, consumer: GpuConsumer) -> bool:
        """Is exactly this consumer the resident one? Deliberately lock-free: acquire()
        holds the lock for its whole with-body (a job's duration), so taking it here
        would block a display read behind the running job. A GIL-atomic reference
        compare is racy only in the benign direction (a snapshot for UI/refusal text)."""
        return self._resident is consumer


@lru_cache
def get_gpu_manager() -> GpuResourceManager:
    return GpuResourceManager()
