"""Breeze TTS 2 adapter: a local cloning engine with voice direction, driven out-of-process.

Breeze TTS 2 (BreezeBlue, 3B; the Qwen3-TTS codec lineage) is a source checkout whose pins (torch
2.9.1/cu128, transformers 4.57.3) fit neither this venv nor the Qwen3-TTS env. So it lives in its
OWN env and runs as a subprocess *worker* (``breeze_worker.py``); this adapter NEVER imports the
Breeze repo or anything from the Breeze env, and reuses IndexTTS-2's newline-JSON stdio transport,
taking each line's audio back as a WAV in a scratch file. The GpuResourceManager frees the card by
calling ``unload()``, which TERMINATES the worker so the OS reclaims all its VRAM (the Breeze
runtime has no unload path of its own; process death IS the free).

Output/determinism truths (checked against the Breeze repo at daa6cb2):
- 24 kHz mono float out of the codec -> written as 16-bit PCM; already canonical, no resample.
- autoregressive -> ``requires_validation`` (it can drop, repeat or add words and sounds).
- every step samples; the worker seeds random/numpy/torch per line and the runtime reseeds torch
  right before sampling, so one seed gives one PERFORMANCE: measured on the 3080 Ti, repeats of
  a line had identical length and correlated 0.996-0.9997 sample for sample. They are not
  bit-identical (bf16 GPU math is not bit-reproducible, so the odd fine-codebook token flips).
  The segment cache keeps whichever take was rendered first.
- Breeze has NO transcript-free clone mode: the prompt is the reference clip plus its exact
  transcript (``reference.txt``). The voice's settings must BIND that transcript with the same
  ``ref_text_digest`` Qwen3-TTS uses (a digest of the clip's bytes + the text), so it rides the
  segment settings_hash; a voice that binds none, or whose reference.txt no longer matches,
  fails loudly rather than rendering from a transcript the cache key never saw.
- voice direction: ``settings["instruction"]`` (a natural-language delivery note) switches the
  line to Breeze's direction template with classifier-free guidance at ``settings["cfg_scale"]``
  (default 4.0). Both ride settings_hash. Without an instruction the line is a plain clone and
  guidance is off. ``map_emotion`` yields nothing for this engine.
- ``model_version`` fingerprints the checkpoint dir (computed WITHOUT spawning the worker, so
  the offline cost estimate builds the same SegmentKey the render will), plus the precision, the
  fast-path config and the worker's generation recipe (cap, guidance default, runtime constants).

OOM/death safety mirrors IndexTTS-2 and Qwen3-TTS: a reply flagged ``oom`` (or a dead/hung/
timed-out worker) triggers a synchronous kill + restart + retry before a loud SynthesisError.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from seiyuu.engines.audio import AudioFile, to_canonical
from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine

# Importing the worker MODULE is safe (its top level is stdlib only; it spawns nothing): the
# generation recipe, fast configs and protocol version then have one source of truth.
from seiyuu.engines.breeze_worker import (
    FAST_CONFIGS,
    GEN_RECIPE,
    PIECE_GAP_SECONDS,
    PROTOCOL_VERSION,
    split_long_text,
)
from seiyuu.engines.indextts2_engine import (
    _FINGERPRINT_CONTENT_MAX_BYTES,
    _FINGERPRINT_CONTENT_SUFFIXES,
    SubprocessTransport,
    WorkerError,
    WorkerTransport,
)

# The transcript binding is the SAME value under any engine's settings (a digest of the clip and
# its text, no engine in it), so a voice keeps one binding whichever engine clones it.
from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest

BREEZE_SAMPLE_RATE = 24000  # mirrored from the worker so cost/estimate never spawn it
BREEZE_FAST_CONFIGS = tuple(sorted(FAST_CONFIGS))  # ("decode", "off")
DEFAULT_FAST = "decode"
# Per-line render settings this engine reads (both ride the frozen settings_hash).
INSTRUCTION_KEY = "instruction"
CFG_SCALE_KEY = "cfg_scale"


def breeze_config(cfg: Any) -> dict[str, Any]:
    """Worker python, repo checkout, checkpoint dir and fast-path config from settings.
    ``getattr`` defaults keep partial configs (tests, older .env files) working."""
    return {
        "worker_python": getattr(cfg, "breeze_worker_python", None),
        "repo_dir": getattr(cfg, "breeze_repo_dir", None),
        "checkpoint_dir": getattr(cfg, "breeze_checkpoint_dir", None),
        "fast": getattr(cfg, "breeze_fast", DEFAULT_FAST),
    }


def breeze_ready(cfg: Any) -> bool:
    """Is Breeze set up on this machine: the worker interpreter, the repo checkout and the
    weights all present? Cheap (a few stat calls), offline, and never raises."""
    config = breeze_config(cfg)
    python = config["worker_python"]
    try:
        has_python = python is not None and Path(python).is_file()
    except OSError:
        has_python = False
    return (
        has_python
        and repo_present(config["repo_dir"])
        and checkpoint_present(config["checkpoint_dir"])
    )


def worker_env(base: dict[str, str], *, windows: bool | None = None) -> dict[str, str]:
    """The worker process environment, from ``base`` (normally ``os.environ``).

    Weights are a local dir: never reach for the hub. Unbuffered so replies aren't stuck in a
    pipe buffer behind the JSON line. On Windows inductor's static CUDA launcher breaks the
    compiled fast path (``OverflowError: Python int too large to convert to C long``), so it is
    switched off here as well as in the worker (torch reads it at import). The Breeze stream
    runtime refuses any ``QWEN_TTS_STREAM_BACKEND`` but the official one, so a stray value from
    the parent shell is dropped."""
    env = dict(base)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("QWEN_TTS_STREAM_BACKEND", None)
    on_windows = os.name == "nt" if windows is None else windows
    if on_windows:
        env["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"
    return env


# --- repo + weights probes (offline; no worker, no SDK) --------------------------------------


def repo_present(repo_dir: Path | str | None) -> bool:
    """Best-effort: does ``repo_dir`` look like the breeze-tts checkout the worker imports?"""
    if repo_dir is None:
        return False
    d = Path(repo_dir)
    try:
        return (d / "breeze_infer" / "templates.py").is_file() and (
            d / "models" / "fast_streaming.py"
        ).is_file()
    except OSError:
        return False


def checkpoint_present(checkpoint_dir: Path | str | None) -> bool:
    """Best-effort: does ``checkpoint_dir`` hold the model weights and the bundled audio
    tokenizer? (UI/bench probe)"""
    if checkpoint_dir is None:
        return False
    d = Path(checkpoint_dir)
    try:
        return (
            (d / "config.json").is_file()
            and any(p.suffix == ".safetensors" for p in d.iterdir())
            and (d / "audio_tokenizer").is_dir()
        )
    except OSError:
        return False


def checkpoint_fingerprint(checkpoint_dir: Path | str | None) -> str:
    """Stable digest of the on-disk checkpoint dir -> ``breeze-<12hex>``.

    Same recipe as IndexTTS-2's and Qwen3-TTS's: every file's (relative path, size) plus the
    content of small config files; multi-GB weight blobs are not read. Hub bookkeeping under
    ``.cache/`` (download metadata with timestamps) is skipped, so re-running a download never
    invalidates the cache. Raises loudly when the dir is absent/empty: a SegmentKey must never be
    built against a phantom model version.
    """
    if checkpoint_dir is None:
        raise SynthesisError(
            "breeze: no checkpoint dir configured (set BREEZE_CHECKPOINT_DIR); "
            "cannot compute model_version"
        )
    d = Path(checkpoint_dir)
    if not d.is_dir():
        raise SynthesisError(f"breeze: checkpoint dir not found at {d}")
    digest = hashlib.sha256()
    files = sorted(
        (p for p in d.rglob("*") if p.is_file() and ".cache" not in p.relative_to(d).parts),
        key=lambda p: p.relative_to(d).as_posix(),
    )
    for path in files:
        rel = path.relative_to(d).as_posix()
        size = path.stat().st_size
        digest.update(rel.encode("utf-8"))
        digest.update(str(size).encode("utf-8"))
        if (
            path.suffix.lower() in _FINGERPRINT_CONTENT_SUFFIXES
            and size <= _FINGERPRINT_CONTENT_MAX_BYTES
        ):
            digest.update(path.read_bytes())
    if not files:
        raise SynthesisError(f"breeze: checkpoint dir {d} is empty")
    return f"breeze-{digest.hexdigest()[:12]}"


# --- engine ----------------------------------------------------------------------------------


class BreezeEngine(TTSEngine):
    engine_id = "breeze"
    requires_validation = True  # autoregressive: output must pass whisper
    clones_from_library = True  # renders from voices/{voice_id}/reference.wav + bound .txt

    def __init__(
        self,
        *,
        voices_dir: Path | None = None,
        checkpoint_dir: Path | None = None,
        repo_dir: Path | None = None,
        worker_python: Path | None = None,
        fast: str | None = None,
        load_timeout: float | None = None,
        request_timeout: float | None = None,
        max_restarts: int | None = None,
        transport_factory: Callable[[], WorkerTransport] | None = None,
    ) -> None:
        if any(
            v is None
            for v in (
                voices_dir,
                checkpoint_dir,
                repo_dir,
                worker_python,
                fast,
                load_timeout,
                request_timeout,
                max_restarts,
            )
        ):
            from seiyuu.settings import get_settings

            cfg = get_settings()
            paths = breeze_config(cfg)
            voices_dir = voices_dir if voices_dir is not None else cfg.voices_dir
            checkpoint_dir = (
                checkpoint_dir if checkpoint_dir is not None else paths["checkpoint_dir"]
            )
            repo_dir = repo_dir if repo_dir is not None else paths["repo_dir"]
            worker_python = worker_python if worker_python is not None else paths["worker_python"]
            fast = fast if fast is not None else paths["fast"]
            load_timeout = (
                load_timeout if load_timeout is not None else cfg.breeze_worker_load_timeout
            )
            request_timeout = (
                request_timeout
                if request_timeout is not None
                else cfg.breeze_worker_request_timeout
            )
            max_restarts = (
                max_restarts if max_restarts is not None else cfg.breeze_worker_max_restarts
            )
        if fast not in FAST_CONFIGS:
            raise SynthesisError(
                f"breeze: unknown fast config {fast!r}; expected {BREEZE_FAST_CONFIGS} "
                "(BREEZE_FAST)"
            )
        self._voices_dir = Path(voices_dir)  # type: ignore[arg-type]
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self._repo_dir = Path(repo_dir) if repo_dir is not None else None
        self._worker_python = Path(worker_python) if worker_python is not None else None
        self._fast = str(fast)
        self._load_timeout = float(load_timeout)  # type: ignore[arg-type]
        self._request_timeout = float(request_timeout)  # type: ignore[arg-type]
        self._max_restarts = int(max_restarts)  # type: ignore[arg-type]
        self._factory = transport_factory  # injected in tests; real one built lazily otherwise
        self._transport: WorkerTransport | None = None
        self._loaded = False  # has the resident worker been told to load the model?
        self._model_version: str | None = None
        # reference.wav (path, mtime, size) -> sha256: the worker keys its encoded-reference cache
        # by content, and hashing a 15 s clip on every line is wasted work.
        self._ref_hashes: dict[tuple[str, int, int], str] = {}
        # Observability, as on the other worker engines. The engine benchmark reads the restart
        # and OOM counts; capped_lines is not reported anywhere yet.
        self.worker_restarts = 0
        self.worker_ooms = 0
        self.capped_lines = 0  # lines that ran into the generation cap (likely runaways)
        self.split_lines = 0  # lines too long for the model's window, spoken in pieces

    # -- catalog facts ------------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        # Keyed to the on-disk checkpoint, NOT the repo or pip versions. bf16 is the only
        # precision this adapter runs. The fast path is not bit-exact with eager execution, so
        # its config is always part of the version; so is the worker's generation recipe (the
        # per-line cap, the default guidance scale and the runtime constants all change output).
        # Memoized: the fingerprint walks the weights dir.
        if self._model_version is None:
            fingerprint = checkpoint_fingerprint(self._checkpoint_dir)
            self._model_version = f"{fingerprint}-bf16-{self._fast}-gen{GEN_RECIPE}"
        return self._model_version

    @property
    def native_sample_rate(self) -> int:
        return BREEZE_SAMPLE_RATE

    def list_voices(self) -> list[EngineVoice]:
        return []  # cloned voices live in the voice library, not the engine

    def cost_estimate(self, text: str) -> float:
        return 0.0

    # -- lifecycle ----------------------------------------------------------------------------

    def warm(self) -> None:
        """Boot the worker, load the checkpoint and warm the fast graphs (M6b warmup job)."""
        self._ensure_loaded()

    def prepare_voice(self, voice_id: str) -> None:  # type: ignore[override]
        """No on-disk speaker cache (the worker caches encoded references in memory), so warm-up
        just boots+loads the model; the reference is encoded on the first line."""
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
                "breeze: worker interpreter not configured or missing "
                f"({self._worker_python}); set BREEZE_WORKER_PYTHON to the Breeze env's python"
            )
        if not repo_present(self._repo_dir):
            raise SynthesisError(
                f"breeze: no breeze-tts checkout at {self._repo_dir}; clone it and set "
                "BREEZE_REPO_DIR"
            )
        if not checkpoint_present(self._checkpoint_dir):
            raise SynthesisError(
                f"breeze: checkpoint not found at {self._checkpoint_dir}; download it and set "
                "BREEZE_CHECKPOINT_DIR"
            )
        worker = Path(__file__).with_name("breeze_worker.py")
        argv = [
            str(self._worker_python),
            str(worker),
            "--repo-dir",
            str(self._repo_dir),
            "--checkpoint-dir",
            str(self._checkpoint_dir),
            "--fast",
            self._fast,
        ]
        return SubprocessTransport(
            argv,
            env=worker_env(dict(os.environ)),
            cwd=None,
            boot_timeout=self._load_timeout,
            name="breeze",
            protocol=PROTOCOL_VERSION,
        )

    def _ensure_started(self) -> WorkerTransport:
        if self._transport is None:
            self._transport = (self._factory or self._default_factory)()
            self._loaded = False
        return self._transport

    def _ensure_loaded(self) -> WorkerTransport:
        """The worker with the model resident. Loading is its own request so the multi-GB load
        and the fast path's compile get the load timeout, not a line's request timeout."""
        transport = self._ensure_started()
        if not self._loaded:
            reply = transport.request({"cmd": "load"}, timeout=self._load_timeout)
            if not reply.get("ok"):
                raise WorkerError(
                    f"breeze worker failed to load: {reply.get('error')}",
                    oom=bool(reply.get("oom")),
                )
            self._loaded = True
        return transport

    def _reference_path(self, voice_id: str) -> Path:
        reference = self._voices_dir / voice_id / "reference.wav"
        if not reference.is_file():
            raise SynthesisError(f"breeze: no reference.wav for voice {voice_id!r} at {reference}")
        return reference

    def _reference_sha256(self, reference: Path) -> str:
        stat = reference.stat()
        key = (str(reference), stat.st_mtime_ns, stat.st_size)
        digest = self._ref_hashes.get(key)
        if digest is None:
            digest = hashlib.sha256(reference.read_bytes()).hexdigest()
            self._ref_hashes[key] = digest
        return digest

    def _reference_text(self, voice: str, reference: Path, settings: dict[str, Any]) -> str:
        """The reference clip's exact transcript (``reference.txt`` beside it).

        Breeze cannot clone without it, and it shapes every line, so it must be in the SegmentKey:
        the voice's settings carry ``ref_text_digest`` (it rides settings_hash) and the file must
        match that digest for THIS clip. A voice that binds no transcript is refused even when a
        reference.txt sits beside the clip (dropped in by hand, it would change the output under
        a key that never saw it); a missing, edited or stale (re-cloned clip) transcript fails
        loudly for the same reason."""
        bound = settings.get(REF_TEXT_DIGEST_KEY)
        if bound is None:
            raise SynthesisError(
                f"breeze: voice {voice!r} binds no reference transcript. Breeze clones from the "
                "clip AND its exact words: use a voice whose meta.json carries ref_text_digest "
                "(a designed voice whose kept take passed the speech check, or a Qwen clone of "
                "a clip of at most 30 s)"
            )
        path = reference.with_name("reference.txt")
        text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        if not text:
            raise SynthesisError(
                f"breeze: voice {voice!r} binds a reference transcript but {path} is missing "
                "or empty; re-clone the voice or design it again"
            )
        if reference_text_digest(self._reference_sha256(reference), text) != float(bound):
            raise SynthesisError(
                f"breeze: voice {voice!r}: {path.name} does not match the transcript its "
                "meta binds (edited by hand, or reference.wav changed); re-clone the voice or "
                "design it again"
            )
        return text

    @staticmethod
    def _direction(voice: str, settings: dict[str, Any]) -> dict[str, Any]:
        """The message fields for voice direction: an instruction and its guidance scale, or
        nothing for a plain clone. A scale without an instruction is not sent (guidance needs
        one); malformed values fail loudly here rather than inside the worker."""
        instruction = settings.get(INSTRUCTION_KEY)
        if instruction is not None and not isinstance(instruction, str):
            raise SynthesisError(
                f"breeze: voice {voice!r}: {INSTRUCTION_KEY} must be text, got "
                f"{type(instruction).__name__}"
            )
        scale = settings.get(CFG_SCALE_KEY)
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, int | float):
                raise SynthesisError(f"breeze: voice {voice!r}: {CFG_SCALE_KEY} must be a number")
            if not math.isfinite(scale) or scale <= 0:
                raise SynthesisError(
                    f"breeze: voice {voice!r}: {CFG_SCALE_KEY} must be above 0, got {scale!r}"
                )
        if not instruction or not instruction.strip():
            return {}
        fields: dict[str, Any] = {"instruction": instruction.strip()}
        if scale is not None:
            fields["cfg_scale"] = float(scale)
        return fields

    def _run(self, message: dict[str, Any], what: str) -> tuple[np.ndarray, int]:
        """Send one generation request; on a worker OOM/death/timeout kill + restart it (fresh
        process == clean VRAM) and retry up to max_restarts times before failing loudly."""
        last_error: WorkerError | None = None
        for _attempt in range(self._max_restarts + 1):
            fd, out_path = tempfile.mkstemp(suffix=".breeze.wav")
            os.close(fd)
            try:
                transport = self._ensure_loaded()
                message["out_path"] = out_path
                reply = transport.request(message, timeout=self._request_timeout)
                if not reply.get("ok") and reply.get("request_error"):
                    # The worker is healthy and refused THIS request (a line too long for the
                    # model's window, say): a restart would only reload 8 GB to hear it again.
                    raise SynthesisError(f"breeze: {reply.get('error')} ({what})")
                if not reply.get("ok"):
                    raise WorkerError(
                        f"breeze worker {message['cmd']} failed: {reply.get('error')}",
                        oom=bool(reply.get("oom")),
                    )
                self.capped_lines += int(bool(reply.get("hit_cap")))
                samples, sample_rate = sf.read(out_path, dtype="float32", always_2d=False)
                if samples.size == 0:
                    raise SynthesisError(f"breeze: worker produced empty audio ({what})")
                return samples, int(sample_rate)
            except WorkerError as exc:
                last_error = exc
                self.worker_restarts += 1
                self.worker_ooms += int(exc.oom)
                self._close_transport()  # kill the (possibly OOM/hung) worker before retrying
            finally:
                Path(out_path).unlink(missing_ok=True)
        assert last_error is not None
        raise SynthesisError(
            f"breeze: {message['cmd']} failed after {self._max_restarts + 1} attempt(s) "
            f"({what}): {last_error}"
        ) from last_error

    # -- voice design -------------------------------------------------------------------------

    def design_voice(
        self,
        text: str,
        description: str,
        *,
        seed: int,
        settings: dict[str, Any] | None = None,
    ) -> AudioFile:
        """Speak ``text`` in a NEW voice described by ``description`` (no reference clip).

        The one-off voice-creation step: the take is saved as a voice's reference clip and
        cloned from then on. It is the same checkpoint that renders, so nothing is swapped or
        reloaded between designing a cast and rendering with it. Callers hold the GPU through
        the GpuResourceManager exactly as for synthesis. ``settings`` may carry ``cfg_scale``."""
        if not text.strip() or not description.strip():
            raise SynthesisError("breeze: voice design needs both text and a description")
        message: dict[str, Any] = {
            "cmd": "design",
            "text": text,
            "instruction": description.strip(),
            "seed": int(seed),
        }
        scale = (settings or {}).get(CFG_SCALE_KEY)
        if scale is not None:
            message["cfg_scale"] = float(scale)
        samples, sample_rate = self._run(message, f"design {description[:60]!r}")
        return to_canonical(samples, sample_rate)

    # -- synthesis ----------------------------------------------------------------------------

    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[np.ndarray, int]:
        reference = self._reference_path(voice)
        request: dict[str, Any] = {
            "cmd": "synthesize",
            "reference_wav": str(reference),
            "reference_sha256": self._reference_sha256(reference),
            "ref_text": self._reference_text(voice, reference, settings),
            "seed": settings.get("seed"),
            **self._direction(voice, settings),
        }
        pieces = split_long_text(text)
        if len(pieces) == 1:
            return self._run({**request, "text": text}, f"voice {voice!r}, text {text[:80]!r}")
        # A line too long for the model's window: one request per piece, the SAME reference,
        # seed and direction for each (one performance; a validation retry re-seeds the whole
        # line), joined with a sentence's pause. A piece the worker refuses fails the line.
        capped_before = self.capped_lines
        parts: list[np.ndarray] = []
        rate = BREEZE_SAMPLE_RATE
        for n, piece in enumerate(pieces, 1):
            what = f"voice {voice!r}, piece {n} of {len(pieces)}, text {piece[:80]!r}"
            samples, rate = self._run({**request, "text": piece}, what)
            if parts:
                parts.append(np.zeros(round(PIECE_GAP_SECONDS * rate), dtype=np.float32))
            parts.append(np.asarray(samples, dtype=np.float32))
        self.capped_lines = capped_before + int(self.capped_lines > capped_before)  # one line
        self.split_lines += 1
        return np.concatenate(parts), rate
