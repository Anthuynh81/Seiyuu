"""IndexTTS-2 adapter (M7): a SECOND local zero-shot cloning engine, driven out-of-process.

IndexTTS-2 hard-pins torch 2.8/cu128 + transformers 4.52, which conflict irreconcilably with
this venv's torch 2.6 + transformers 5 (chatterbox's pins). One interpreter holds one of each,
so IndexTTS-2 cannot live in-process without evicting the working CUDA build (the #1 project
gotcha). It runs instead as a subprocess *worker* (``indextts2_worker.py``) in its OWN cu128 uv
env; this adapter is stdlib-only (it NEVER imports the IndexTTS-2 SDK) and drives the worker over
newline-delimited JSON on stdin/stdout, taking each segment's audio back as a WAV in a scratch
file.

GPU discipline plugs in for free: the GpuResourceManager frees a resident consumer by calling
``unload()``; this adapter's ``unload()`` TERMINATES the worker, so the OS reclaims all of its
VRAM (there is no in-process free — process death IS the free). The worker is long-lived across
a render's segments (its in-memory speaker-cond cache survives), torn down only when a competitor
(attribution, another engine) acquires the GPU.

Output/determinism truths (confirmed against IndexTTS-2 source; see the M7 scope memo):
- 22050 Hz native -> ``to_canonical`` resamples to the canonical 24 kHz.
- autoregressive -> ``requires_validation`` (its output rides the whisper retry loop).
- ``infer()`` has no seed arg and samples, so the worker seeds torch right before each infer.
- emotion is IndexTTS-2's native 8-dim ``emo_vector`` + ``emo_alpha`` (``voices/emotion.py`` maps
  the shared taxonomy onto it); those settings ride the FROZEN SegmentKey's ``settings_hash`` VALUE.
- ``model_version`` is a fingerprint of the on-disk CHECKPOINTS, not the pip version (checkpoints
  are downloaded and decoupled from the package), computed WITHOUT spawning the worker so the
  offline cost-estimate builds the same SegmentKey the render will.

OOM safety: on 8GB the peak sits at the card's edge, so CUDA OOM is a real failure mode. A worker
reply flagged ``oom`` (or a dead/hung/timed-out worker) triggers a synchronous kill + restart +
one retry before a loud SynthesisError. ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is
baked into the worker env to hold sustained throughput at the ceiling.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine

# Native sample rate mirrored from the worker (kept here too so cost/estimate never spawn it).
INDEXTTS2_SAMPLE_RATE = 22050
# Small config/metadata files whose CONTENT is folded into the weights fingerprint (they change
# the output but not the big weight blobs' sizes). Large weight files contribute (name, size) only.
_FINGERPRINT_CONTENT_SUFFIXES = {".yaml", ".yml", ".json", ".txt", ".cfg"}
_FINGERPRINT_CONTENT_MAX_BYTES = 1 << 20  # 1 MiB — only hash content of small files


class WorkerError(SynthesisError):
    """The worker died, hung, timed out, or replied with a failure (incl. OOM)."""

    def __init__(self, message: str, *, oom: bool = False) -> None:
        super().__init__(message)
        self.oom = oom


class WorkerTransport(Protocol):
    """The wire to one worker process. Injected as a FACTORY so the OOM path can rebuild it."""

    def request(self, message: dict[str, Any], *, timeout: float) -> dict[str, Any]: ...
    def close(self) -> None: ...


# --- checkpoints fingerprint (offline; no worker, no SDK) ------------------------------------


def checkpoints_present(checkpoints_dir: Path | str | None) -> bool:
    """Best-effort: does the checkpoints dir exist and hold at least one file? (UI weights probe)"""
    if checkpoints_dir is None:
        return False
    d = Path(checkpoints_dir)
    try:
        return d.is_dir() and any(p.is_file() for p in d.rglob("*"))
    except OSError:
        return False


def weights_fingerprint(checkpoints_dir: Path | str | None) -> str:
    """Stable digest of the on-disk checkpoints -> ``indextts2-<12hex>``.

    Folds every file's (relative path, size), plus the full content of small config files
    (which change output without changing the big weights' sizes). Content of multi-GB weight
    blobs is intentionally NOT hashed (too slow for a per-estimate call) — a real checkpoint
    swap changes a size or a config, so (name, size, config-content) is sufficient and cheap.
    Raises loudly when the dir is absent/empty: without weights the engine cannot render, and a
    SegmentKey must never be built against a phantom model version.
    """
    if checkpoints_dir is None:
        raise SynthesisError(
            "indextts2: no checkpoints dir configured (set indextts2_checkpoints_dir); "
            "cannot compute model_version"
        )
    d = Path(checkpoints_dir)
    if not d.is_dir():
        raise SynthesisError(f"indextts2: checkpoints dir not found at {d}")
    digest = hashlib.sha256()
    files = sorted(
        (p for p in d.rglob("*") if p.is_file()), key=lambda p: p.relative_to(d).as_posix()
    )
    saw_file = False
    for path in files:
        saw_file = True
        rel = path.relative_to(d).as_posix()
        size = path.stat().st_size
        digest.update(rel.encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
        if (
            path.suffix.lower() in _FINGERPRINT_CONTENT_SUFFIXES
            and size <= _FINGERPRINT_CONTENT_MAX_BYTES
        ):
            digest.update(path.read_bytes())
    if not saw_file:
        raise SynthesisError(f"indextts2: checkpoints dir {d} is empty")
    return f"indextts2-{digest.hexdigest()[:12]}"


# --- subprocess transport --------------------------------------------------------------------

_EOF = object()  # sentinel pushed onto the stdout queue when the worker's stdout closes


class SubprocessTransport:
    """Owns one worker subprocess and its stdio. Reads stdout/stderr on daemon threads (the
    cross-platform way to avoid a pipe-buffer deadlock while blocked on a slow infer), and does a
    boot handshake so a worker that fails to even start is caught at once, not on first synth."""

    def __init__(
        self,
        argv: list[str],
        *,
        env: dict[str, str],
        cwd: str | None,
        boot_timeout: float,
    ) -> None:
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            bufsize=1,  # line-buffered
        )
        self._out_q: queue.Queue[Any] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=100)
        self._out_thread = threading.Thread(target=self._pump_stdout, daemon=True)
        self._err_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._out_thread.start()
        self._err_thread.start()
        try:
            self._boot(boot_timeout)
        except BaseException:
            # A boot that raises must not orphan the spawned worker + its reader threads: nothing
            # upstream ever gets a handle to close() a transport whose __init__ threw, so terminate
            # it here (close() drains the pumps to EOF) before re-raising.
            self.close()
            raise

    def _pump_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._out_q.put(line)
        self._out_q.put(_EOF)

    def _pump_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_tail.append(line.rstrip("\n"))

    def _stderr_context(self) -> str:
        tail = list(self._stderr_tail)[-15:]
        return ("\n  worker stderr:\n    " + "\n    ".join(tail)) if tail else ""

    def _boot(self, timeout: float) -> None:
        reply = self._read(timeout)
        if reply.get("event") != "ready":
            raise WorkerError(f"indextts2 worker did not announce readiness: {reply}")

    def _read(self, timeout: float) -> dict[str, Any]:
        try:
            item = self._out_q.get(timeout=timeout)
        except queue.Empty as exc:
            raise WorkerError(
                f"indextts2 worker timed out after {timeout:g}s{self._stderr_context()}"
            ) from exc
        if item is _EOF:
            code = self._proc.poll()
            raise WorkerError(f"indextts2 worker exited (code {code}){self._stderr_context()}")
        try:
            return json.loads(item)
        except ValueError as exc:
            raise WorkerError(f"indextts2 worker sent non-JSON: {item!r}") from exc

    def request(self, message: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        if self._proc.stdin is None or self._proc.poll() is not None:
            raise WorkerError(f"indextts2 worker is not running{self._stderr_context()}")
        try:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError(
                f"indextts2 worker pipe broke: {exc}{self._stderr_context()}"
            ) from exc
        return self._read(timeout)

    def close(self) -> None:
        """Synchronous terminate: close stdin, then terminate/kill and WAIT so the OS has fully
        reclaimed the worker's VRAM before we return (the GPU manager relies on this)."""
        proc = self._proc
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            proc.wait()  # reap
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


# --- engine ----------------------------------------------------------------------------------


class IndexTTS2Engine(TTSEngine):
    engine_id = "indextts2"
    requires_validation = True  # autoregressive (GPT+S2Mel+BigVGAN): output must pass whisper
    clones_from_library = True  # renders from voices/{voice_id}/reference.wav, like chatterbox

    def __init__(
        self,
        *,
        voices_dir: Path | None = None,
        checkpoints_dir: Path | None = None,
        worker_python: Path | None = None,
        use_fp16: bool | None = None,
        load_timeout: float | None = None,
        request_timeout: float | None = None,
        max_restarts: int | None = None,
        transport_factory: Callable[[], WorkerTransport] | None = None,
    ) -> None:
        if (
            voices_dir is None
            or checkpoints_dir is None
            or worker_python is None
            or use_fp16 is None
            or load_timeout is None
            or request_timeout is None
            or max_restarts is None
        ):
            from seiyuu.settings import get_settings

            cfg = get_settings()
            voices_dir = voices_dir if voices_dir is not None else cfg.voices_dir
            checkpoints_dir = (
                checkpoints_dir if checkpoints_dir is not None else cfg.indextts2_checkpoints_dir
            )
            worker_python = (
                worker_python if worker_python is not None else cfg.indextts2_worker_python
            )
            use_fp16 = use_fp16 if use_fp16 is not None else cfg.indextts2_use_fp16
            load_timeout = (
                load_timeout if load_timeout is not None else cfg.indextts2_worker_load_timeout
            )
            request_timeout = (
                request_timeout
                if request_timeout is not None
                else cfg.indextts2_worker_request_timeout
            )
            max_restarts = (
                max_restarts if max_restarts is not None else cfg.indextts2_worker_max_restarts
            )
        self._voices_dir = Path(voices_dir)
        self._checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else None
        self._worker_python = Path(worker_python) if worker_python is not None else None
        self._use_fp16 = bool(use_fp16)
        self._load_timeout = float(load_timeout)
        self._request_timeout = float(request_timeout)
        self._max_restarts = int(max_restarts)
        self._factory = transport_factory  # injected in tests; real one built lazily otherwise
        self._transport: WorkerTransport | None = None
        self._loaded = False  # has the resident transport been sent `load`?
        self._model_version: str | None = None

    # -- catalog facts ------------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        # Keyed to the on-disk checkpoints, NOT the pip version — the same string keys
        # SegmentKey.engine_model_version. Memoized: the fingerprint walks the weights dir.
        # fp16 vs fp32 changes the sampled audio, so it is part of the version (base.py contract:
        # model_version must change when output would) — flipping precision must miss the cache.
        if self._model_version is None:
            fingerprint = weights_fingerprint(self._checkpoints_dir)  # "indextts2-<hash>"
            self._model_version = f"{fingerprint}-{'fp16' if self._use_fp16 else 'fp32'}"
        return self._model_version

    @property
    def native_sample_rate(self) -> int:
        return INDEXTTS2_SAMPLE_RATE  # 22050; to_canonical resamples to 24 kHz

    def list_voices(self) -> list[EngineVoice]:
        return []  # cloned voices live in the voice library, not the engine

    def cost_estimate(self, text: str) -> float:
        return 0.0

    # -- lifecycle ----------------------------------------------------------------------------

    def warm(self) -> None:
        """Boot the worker and load the weights (M6b warmup job); stays lazily resident."""
        self._ensure_loaded()

    def prepare_voice(self, voice_id: str) -> None:  # type: ignore[override]
        """No on-disk speaker cache exists (the worker caches conds in-memory only), so warm-up
        just boots+loads the model; the reference's cond is computed on first synth."""
        self._reference_path(voice_id)  # fail early if the reference is missing
        self._ensure_loaded()

    def unload(self) -> None:
        """Terminate the worker synchronously (GpuConsumer handoff) — process death frees VRAM."""
        self._close_transport()

    def _close_transport(self) -> None:
        transport, self._transport = self._transport, None
        self._loaded = False
        if transport is not None:
            transport.close()

    # -- worker plumbing ----------------------------------------------------------------------

    def _default_factory(self) -> WorkerTransport:
        if self._worker_python is None or not self._worker_python.exists():
            raise SynthesisError(
                "indextts2: worker interpreter not configured or missing "
                f"({self._worker_python}); set indextts2_worker_python to the cu128 env's python"
            )
        if not checkpoints_present(self._checkpoints_dir):
            raise SynthesisError(
                f"indextts2: checkpoints not found at {self._checkpoints_dir}; "
                "download them and set indextts2_checkpoints_dir"
            )
        worker = Path(__file__).with_name("indextts2_worker.py")
        argv = [
            str(self._worker_python),
            str(worker),
            "--checkpoints",
            str(self._checkpoints_dir),
            "--fp16" if self._use_fp16 else "--no-fp16",
        ]
        env = dict(os.environ)
        # Fragmentation fix at the VRAM ceiling (spike: holds sustained ~4x RTF); force
        # unbuffered stdout so replies aren't stuck in a pipe buffer behind the JSON line.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONUNBUFFERED"] = "1"
        return SubprocessTransport(argv, env=env, cwd=None, boot_timeout=self._load_timeout)

    def _ensure_started(self) -> WorkerTransport:
        if self._transport is None:
            self._transport = (self._factory or self._default_factory)()
            self._loaded = False
        return self._transport

    def _ensure_loaded(self) -> WorkerTransport:
        transport = self._ensure_started()
        if not self._loaded:
            reply = transport.request({"cmd": "load"}, timeout=self._load_timeout)
            if not reply.get("ok"):
                raise WorkerError(
                    f"indextts2 worker failed to load model: {reply.get('error')}",
                    oom=bool(reply.get("oom")),
                )
            self._loaded = True
        return transport

    def _reference_path(self, voice_id: str) -> Path:
        reference = self._voices_dir / voice_id / "reference.wav"
        if not reference.is_file():
            raise SynthesisError(
                f"indextts2: no reference.wav for voice {voice_id!r} at {reference}"
            )
        return reference

    # -- synthesis ----------------------------------------------------------------------------

    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[np.ndarray, int]:
        reference = self._reference_path(voice)
        message: dict[str, Any] = {
            "cmd": "synthesize",
            "text": text,
            "reference_wav": str(reference),
            "seed": settings.get("seed"),
            "emo_vector": settings.get("emo_vector"),  # None -> worker's neutral path
            "emo_alpha": settings.get("emo_alpha"),
        }
        # Up to max_restarts extra attempts: a worker OOM/death/timeout kills+restarts the
        # worker (fresh process == clean VRAM) and retries before failing loudly.
        last_error: WorkerError | None = None
        for _attempt in range(self._max_restarts + 1):
            fd, out_path = tempfile.mkstemp(suffix=".indextts2.wav")
            os.close(fd)
            try:
                transport = self._ensure_loaded()
                message["out_path"] = out_path
                reply = transport.request(message, timeout=self._request_timeout)
                if not reply.get("ok"):
                    raise WorkerError(
                        f"indextts2 worker synthesis failed: {reply.get('error')}",
                        oom=bool(reply.get("oom")),
                    )
                samples, sample_rate = sf.read(out_path, dtype="float32", always_2d=False)
                if samples.size == 0:
                    raise SynthesisError(
                        f"indextts2: worker produced empty audio (text: {text[:80]!r})"
                    )
                return samples, int(sample_rate)
            except WorkerError as exc:
                last_error = exc
                self._close_transport()  # kill the (possibly OOM/hung) worker before retrying
            finally:
                Path(out_path).unlink(missing_ok=True)
        assert last_error is not None
        raise SynthesisError(
            f"indextts2: synthesis failed after {self._max_restarts + 1} attempt(s): {last_error}"
        ) from last_error
