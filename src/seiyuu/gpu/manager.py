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

Cross-PROCESS discipline: the threading.Lock covers one process only, but the CLI and the
API server are separate processes sharing one card. The `get_gpu_manager()` singleton
(every real entry point) therefore also claims an OS file lock (data/gpu.lock) while ANY
model is resident — from the idle->resident acquire until free_all() empties the card.
Lazy release means a server can sit on VRAM long after its job finished; a CLI run must be
refused THEN too, so the file lock tracks RESIDENCY, not acquire bodies. Contention raises
GpuBusyError immediately (truthful refusal beats an OOM/CPU-spill); OS locks die with
their process, so a crashed holder self-releases and no stale-lock recovery exists.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from seiyuu.repository.lock import FileLockHandle
from seiyuu.settings import get_settings


@runtime_checkable
class GpuConsumer(Protocol):
    def unload(self) -> None:
        """Free this consumer's GPU memory. Default no-op for non-GPU consumers."""


class GpuBusyError(RuntimeError):
    """Another seiyuu PROCESS holds the GPU (cross-process gpu.lock contention)."""


class GpuResourceManager:
    def __init__(self, lock_path: Path | None = None) -> None:
        """``lock_path`` (data/gpu.lock via the singleton) arms the cross-process half of
        the discipline; None keeps the manager process-local (tests, injected managers)."""
        self._lock = threading.Lock()
        self._resident: GpuConsumer | None = None
        self._resident_name = ""
        self._card_lock = FileLockHandle(lock_path) if lock_path is not None else None

    @contextmanager
    def acquire(self, consumer: GpuConsumer, name: str) -> Iterator[None]:
        """Hold the GPU for `consumer` across the `with` body, unloading any other resident."""
        with self._lock:
            if self._resident is None:
                self._claim_card(name)
            elif self._resident is not consumer:
                # in-process handoff: the card stays claimed while residency transfers
                self._free_resident()
            self._resident, self._resident_name = consumer, name
            yield

    def _claim_card(self, name: str) -> None:
        if self._card_lock is None or self._card_lock.try_acquire():
            return
        raise GpuBusyError(
            f"cannot load {name}: another seiyuu process holds the GPU "
            f"(lock: {self._card_lock.path}). Is the API server running, or another "
            f"seiyuu command? Stop it or wait for its job to finish, then retry."
        )

    def _free_resident(self) -> None:
        if self._resident is None:
            return
        self._resident.unload()  # propagate errors loudly — a silent OOM later is worse
        self._resident, self._resident_name = None, ""

    def free_all(self) -> None:
        """Unload the resident model (teardown / explicit handoff)."""
        with self._lock:
            try:
                self._free_resident()
            finally:
                # released only when truthful: an unload() failure leaves the model on the
                # card (resident stays set), so the cross-process claim must survive it
                if self._resident is None and self._card_lock is not None:
                    self._card_lock.release()

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
    # Every real entry point (CLI commands, the API server) shares this singleton, so
    # arming it with the cross-process lock covers every heavy path at once. Bare
    # GpuResourceManager() (tests, injected fakes) stays process-local.
    return GpuResourceManager(lock_path=get_settings().data_dir / "gpu.lock")
