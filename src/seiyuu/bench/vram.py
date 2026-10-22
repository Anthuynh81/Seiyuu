"""Peak-VRAM probes for the engine benchmark.

Two views, chosen per engine by ``default_probes``:

- ``TorchCudaProbe`` — ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` of THIS
  process. Exact for in-process engines (Kokoro, Chatterbox) but blind to anything else.
- ``SmiSamplerProbe`` — samples ``nvidia-smi --query-gpu=memory.used`` on a background thread and
  reports the peak delta over a baseline taken just before the engine loads. The only way to see
  the IndexTTS-2 subprocess worker; for in-process engines it also captures the CUDA context +
  allocator slack that torch's counters omit. It is whole-card, so other GPU users (a browser,
  Ollama) add noise — keep the card otherwise idle while benchmarking.

Discipline: no probe is built for an engine with ``uses_gpu False``, and ``torch.cuda`` is never
touched unless CUDA is available AND the engine runs in this process (a ``torch.cuda`` call in the
parent of a subprocess engine would create a CUDA context and pollute the very reading we take).
"""

import shutil
import subprocess
import threading
from collections.abc import Callable
from typing import Any, Protocol

from seiyuu.bench.models import VramStats

# Engines whose model lives in a separate worker process (torch counters here can't see it).
SUBPROCESS_ENGINES = frozenset({"indextts2"})

_MB = 1024 * 1024


class VramProbe(Protocol):
    def begin(self) -> None: ...
    def mark_loaded(self) -> None: ...
    def end(self, stats: VramStats) -> None:
        """Stop sampling and write this probe's fields into ``stats``."""


class TorchCudaProbe:
    def __init__(self, cuda: Any | None = None) -> None:
        if cuda is None:
            import torch

            cuda = torch.cuda
        self._cuda = cuda
        self._after_load: float | None = None
        self._load_peak: float | None = None

    def begin(self) -> None:
        self._cuda.reset_peak_memory_stats()

    def mark_loaded(self) -> None:
        # CURRENT allocation = the resident footprint once loaded (what must fit beside anything
        # else); the peak since begin() also counts load-time transients (fp32->fp16 casts,
        # temporary state-dict copies), so it is kept separately.
        self._after_load = self._cuda.memory_allocated() / _MB
        self._load_peak = self._cuda.max_memory_allocated() / _MB

    def end(self, stats: VramStats) -> None:
        stats.methods.append("torch")
        stats.torch_peak_allocated_mb = self._cuda.max_memory_allocated() / _MB
        stats.torch_peak_reserved_mb = self._cuda.max_memory_reserved() / _MB
        stats.torch_after_load_allocated_mb = self._after_load
        stats.torch_load_peak_allocated_mb = self._load_peak


def nvidia_smi_query(gpu_index: int = 0, *, timeout: float = 5.0) -> float | None:
    """memory.used (MiB) of one GPU via nvidia-smi, or None if it can't be read."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu_index)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def nvidia_smi_info(gpu_index: int = 0) -> dict[str, str] | None:
    """name / memory.total / driver of one GPU (for the report header), or None."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    fields = ("name", "memory.total", "driver_version")
    try:
        out = subprocess.run(
            [
                exe,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        ).stdout
        values = [v.strip() for v in out.strip().splitlines()[0].split(",")]
        return dict(zip(fields, values, strict=True))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class SmiSamplerProbe:
    """Background-thread peak sampler. ``query`` is injectable (tests pass a fake)."""

    def __init__(
        self,
        query: Callable[[], float | None] | None = None,
        *,
        interval_s: float = 0.2,
    ) -> None:
        self._query = query or nvidia_smi_query
        self._interval = interval_s
        self._baseline: float | None = None
        self._peak: float | None = None
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _sample(self) -> None:
        value = self._query()
        if value is None:
            return
        with self._lock:
            self._samples += 1
            self._peak = value if self._peak is None else max(self._peak, value)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def begin(self) -> None:
        self._baseline = self._query()
        if self._baseline is None:
            return  # unreadable: end() reports nothing rather than a bogus delta
        self._peak = self._baseline
        self._thread = threading.Thread(target=self._run, name="bench-smi", daemon=True)
        self._thread.start()

    def mark_loaded(self) -> None:
        if self._baseline is not None:
            self._sample()

    def end(self, stats: VramStats) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._sample()  # one last reading at the end of the workload
        if self._baseline is None:
            return
        stats.methods.append("nvidia-smi")
        stats.smi_baseline_mb = self._baseline
        stats.smi_peak_mb = self._peak
        stats.smi_peak_delta_mb = (
            None if self._peak is None else max(0.0, self._peak - self._baseline)
        )
        stats.smi_samples = self._samples


def torch_cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def default_probes(
    engine: Any,
    *,
    gpu_index: int = 0,
    smi_interval_s: float = 0.2,
    cuda_available: Callable[[], bool] = torch_cuda_available,
    has_smi: Callable[[], bool] = smi_available,
) -> list[VramProbe]:
    """The probes that make sense for ``engine`` on this machine (possibly none)."""
    if not getattr(engine, "uses_gpu", False):
        return []
    probes: list[VramProbe] = []
    if engine.engine_id not in SUBPROCESS_ENGINES and cuda_available():
        probes.append(TorchCudaProbe())
    if has_smi():
        probes.append(
            SmiSamplerProbe(lambda: nvidia_smi_query(gpu_index), interval_s=smi_interval_s)
        )
    return probes
