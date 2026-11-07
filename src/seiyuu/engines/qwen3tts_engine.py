"""Qwen3-TTS adapter: a third local zero-shot cloning engine, driven out-of-process like IndexTTS-2.

Qwen3-TTS (Apache-2.0; 12 Hz codec, 1.7B) runs through faster-qwen3-tts on qwen-tts-hf, which pin
torch 2.8/cu128 + transformers 5.15.1 — not this venv's torch 2.6 (chatterbox's pin). So it lives
in its OWN uv env and runs as a subprocess *worker* (``qwen3tts_worker.py``); this adapter is
stdlib-only (it NEVER imports the Qwen SDK) and reuses IndexTTS-2's newline-JSON stdio transport,
taking each line's audio back as a WAV in a scratch file.

Two checkpoints share the worker, one resident at a time: Base (clone from a reference clip —
every render) and VoiceDesign (a voice from a written description — the one-off design step,
``design_voice``). The GpuResourceManager frees the card by calling ``unload()``, which TERMINATES
the worker so the OS reclaims all its VRAM (process death IS the free).

Output/determinism truths (checked against the installed faster-qwen3-tts / qwen-tts-hf source):
- 24 kHz mono float out of the codec -> written as 16-bit PCM; already canonical, no resample.
- autoregressive -> ``requires_validation`` (it can drop, repeat or add words: laughs, fillers).
- no API takes a seed and every step samples, so the worker seeds torch right before each call.
- ICL clone (reference + its transcript ``reference.txt``) when the voice's meta BINDS a
  transcript (``settings["qwen3tts"]["ref_text_digest"]``, a digest of the clip's bytes + the
  text), else x-vector clone (speaker embedding only). The digest rides the segment
  settings_hash, so switching mode or transcript always changes the SegmentKey; a reference.txt
  the meta doesn't vouch for is ignored, and one that no longer matches fails loudly. Either way
  ``non_streaming_mode=True`` (prefill the whole line): the clone default feeds text step by step
  and drifts toward rushing.
- a line wrapped in quote marks is sent without them (upstream issue #80: those lines are the
  ones most often spoken in the wrong voice); the whisper check still compares the full text.
- no emotion knob: ``map_emotion`` yields nothing for this engine (neutral, cache-stable).
- ``model_version`` fingerprints the Base checkpoint dir (computed WITHOUT spawning the worker,
  so the offline cost estimate builds the same SegmentKey the render will), plus the precision,
  the worker's generation-cap recipe, the backend when it is not the default, the language when
  it is not English and the KV-cache size when it is not 2048 (the cap that cuts long lines
  depends on it).

OOM/death safety mirrors IndexTTS-2: a reply flagged ``oom`` (or a dead/hung/timed-out worker)
triggers a synchronous kill + restart + retry before a loud SynthesisError.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from seiyuu.engines.audio import AudioFile, to_canonical
from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine
from seiyuu.engines.indextts2_engine import (
    _FINGERPRINT_CONTENT_MAX_BYTES,
    _FINGERPRINT_CONTENT_SUFFIXES,
    SubprocessTransport,
    WorkerError,
    WorkerTransport,
)

# Importing the worker MODULE is safe (its top level is stdlib only; it spawns nothing): the cap
# recipe and protocol version then have one source of truth.
from seiyuu.engines.qwen3tts_worker import CAP_RECIPE, PROTOCOL_VERSION

QWEN3TTS_SAMPLE_RATE = 24000  # mirrored from the worker so cost/estimate never spawn it
DEFAULT_MAX_SEQ_LEN = 2048
# meta.settings["qwen3tts"] key binding reference.txt to the voice (see reference_text_digest).
REF_TEXT_DIGEST_KEY = "ref_text_digest"
BASE_CHECKPOINT = "Qwen3-TTS-12Hz-1.7B-Base"
DESIGN_CHECKPOINT = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN3TTS_BACKENDS = ("fast", "stock")

# Per-voice tunables (meta.settings["qwen3tts"]) forwarded to the talker's sampler; they ride the
# frozen settings_hash. do_sample is deliberately excluded (it would break seeded sampling) and so
# is max_new_tokens (the worker caps it from the text length to stop runaways).
_GEN_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")


def qwen3tts_config(cfg: Any) -> dict[str, Any]:
    """Worker python, models dir, backend, language and KV-cache size from settings.
    ``getattr`` defaults keep partial configs (tests, older .env files) working."""
    return {
        "worker_python": getattr(cfg, "qwen3tts_worker_python", None),
        "models_dir": getattr(cfg, "qwen3tts_models_dir", None),
        "backend": getattr(cfg, "qwen3tts_backend", "fast"),
        "language": getattr(cfg, "qwen3tts_language", "English"),
        "max_seq_len": getattr(cfg, "qwen3tts_max_seq_len", DEFAULT_MAX_SEQ_LEN),
    }


def reference_text_digest(reference_sha256: str, text: str) -> float:
    """The value a voice's ``meta.settings["qwen3tts"]["ref_text_digest"]`` holds to clone in ICL
    mode: a digest of the reference clip's bytes AND its transcript, so the transcript is part of
    the segment settings_hash and is bound to the exact clip it describes. 48 bits, so it is exact
    as a float (VoiceMeta.settings holds floats)."""
    digest = hashlib.sha256(f"{reference_sha256}\n{text.strip()}".encode()).hexdigest()
    return float(int(digest[:12], 16))


# --- weights fingerprint (offline; no worker, no SDK) ----------------------------------------


def models_present(models_dir: Path | str | None, checkpoint: str = BASE_CHECKPOINT) -> bool:
    """Best-effort: does ``<models_dir>/<checkpoint>`` hold a weights file? (UI/bench probe)"""
    if models_dir is None:
        return False
    d = Path(models_dir) / checkpoint
    try:
        return d.is_dir() and any(p.suffix == ".safetensors" for p in d.rglob("*"))
    except OSError:
        return False


def weights_fingerprint(model_dir: Path | str | None) -> str:
    """Stable digest of one on-disk checkpoint dir -> ``qwen3tts-<12hex>``.

    Same recipe as IndexTTS-2's: every file's (relative path, size) plus the content of small
    config files; multi-GB weight blobs are not read. Hub bookkeeping under ``.cache/`` (download
    metadata with timestamps) is skipped, so re-running a download never invalidates the cache.
    Raises loudly when the dir is absent/empty: a SegmentKey must never be built against a
    phantom model version.
    """
    if model_dir is None:
        raise SynthesisError(
            "qwen3tts: no models dir configured (set QWEN3TTS_MODELS_DIR); "
            "cannot compute model_version"
        )
    d = Path(model_dir)
    if not d.is_dir():
        raise SynthesisError(f"qwen3tts: checkpoint dir not found at {d}")
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
        raise SynthesisError(f"qwen3tts: checkpoint dir {d} is empty")
    return f"qwen3tts-{digest.hexdigest()[:12]}"


# --- engine ----------------------------------------------------------------------------------


class Qwen3TTSEngine(TTSEngine):
    engine_id = "qwen3tts"
    requires_validation = True  # autoregressive talker: output must pass whisper
    clones_from_library = True  # renders from voices/{voice_id}/reference.wav (+ bound .txt)

    def __init__(
        self,
        *,
        voices_dir: Path | None = None,
        models_dir: Path | None = None,
        worker_python: Path | None = None,
        backend: str | None = None,
        language: str | None = None,
        max_seq_len: int | None = None,
        load_timeout: float | None = None,
        request_timeout: float | None = None,
        max_restarts: int | None = None,
        transport_factory: Callable[[], WorkerTransport] | None = None,
    ) -> None:
        if any(
            v is None
            for v in (
                voices_dir,
                models_dir,
                worker_python,
                backend,
                language,
                max_seq_len,
                load_timeout,
                request_timeout,
                max_restarts,
            )
        ):
            from seiyuu.settings import get_settings

            cfg = get_settings()
            paths = qwen3tts_config(cfg)
            voices_dir = voices_dir if voices_dir is not None else cfg.voices_dir
            models_dir = models_dir if models_dir is not None else paths["models_dir"]
            worker_python = worker_python if worker_python is not None else paths["worker_python"]
            backend = backend if backend is not None else paths["backend"]
            language = language if language is not None else paths["language"]
            max_seq_len = max_seq_len if max_seq_len is not None else paths["max_seq_len"]
            load_timeout = (
                load_timeout if load_timeout is not None else cfg.qwen3tts_worker_load_timeout
            )
            request_timeout = (
                request_timeout
                if request_timeout is not None
                else cfg.qwen3tts_worker_request_timeout
            )
            max_restarts = (
                max_restarts if max_restarts is not None else cfg.qwen3tts_worker_max_restarts
            )
        if backend not in QWEN3TTS_BACKENDS:
            raise SynthesisError(
                f"qwen3tts: unknown backend {backend!r}; expected {QWEN3TTS_BACKENDS}"
            )
        self._voices_dir = Path(voices_dir)  # type: ignore[arg-type]
        self._models_dir = Path(models_dir) if models_dir is not None else None
        self._worker_python = Path(worker_python) if worker_python is not None else None
        self._backend = str(backend)
        self._language = str(language)
        self._max_seq_len = int(max_seq_len)  # type: ignore[arg-type]
        self._load_timeout = float(load_timeout)  # type: ignore[arg-type]
        self._request_timeout = float(request_timeout)  # type: ignore[arg-type]
        self._max_restarts = int(max_restarts)  # type: ignore[arg-type]
        self._factory = transport_factory  # injected in tests; real one built lazily otherwise
        self._transport: WorkerTransport | None = None
        self._loaded: str | None = None  # checkpoint the resident worker was told to load
        self._model_version: str | None = None
        # reference.wav (path, mtime, size) -> sha256: the worker keys its clone-prompt cache by
        # content, and hashing a 15 s clip on every line is wasted work.
        self._ref_hashes: dict[tuple[str, int, int], str] = {}
        # Observability, as on IndexTTS-2 (read by the engine benchmark / A/B script).
        self.worker_restarts = 0
        self.worker_ooms = 0
        self.capped_lines = 0  # lines that ran into the generation cap (likely runaways)

    # -- catalog facts ------------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        # Keyed to the on-disk Base checkpoint (what renders), NOT the pip version. bf16 is the
        # only precision this adapter runs. The generation cap decides where a long line is cut,
        # so its recipe is always part of the version. The stock backend is not bit-exact with
        # the CUDA-graph fast path, the language changes the prompt and the KV-cache size bounds
        # the cap, so each marks the version when it differs from the default. Memoized: the
        # fingerprint walks the weights dir.
        if self._model_version is None:
            base_dir = self._models_dir / BASE_CHECKPOINT if self._models_dir else None
            version = f"{weights_fingerprint(base_dir)}-bf16-cap{CAP_RECIPE}"
            if self._backend != "fast":
                version += f"-{self._backend}"
            if self._language.lower() != "english":
                version += f"-{self._language.lower()}"
            if self._max_seq_len != DEFAULT_MAX_SEQ_LEN:
                version += f"-kv{self._max_seq_len}"
            self._model_version = version
        return self._model_version

    @property
    def native_sample_rate(self) -> int:
        return QWEN3TTS_SAMPLE_RATE

    def list_voices(self) -> list[EngineVoice]:
        return []  # cloned voices live in the voice library, not the engine

    def cost_estimate(self, text: str) -> float:
        return 0.0

    # -- lifecycle ----------------------------------------------------------------------------

    def warm(self) -> None:
        """Boot the worker and load the Base checkpoint (M6b warmup job)."""
        self._ensure_loaded("base")

    def prepare_voice(self, voice_id: str) -> None:  # type: ignore[override]
        """No on-disk speaker cache (the worker caches clone prompts in memory), so warm-up just
        boots+loads the model; the reference is encoded on the first line."""
        self._reference_path(voice_id)  # fail early if the reference is missing
        self._ensure_loaded("base")

    def unload(self) -> None:
        """Terminate the worker synchronously (GpuConsumer handoff) — process death frees VRAM."""
        self._close_transport()

    def _close_transport(self) -> None:
        transport, self._transport = self._transport, None
        self._loaded = None
        if transport is not None:
            transport.close()

    # -- worker plumbing ----------------------------------------------------------------------

    def _default_factory(self) -> WorkerTransport:
        if self._worker_python is None or not self._worker_python.exists():
            raise SynthesisError(
                "qwen3tts: worker interpreter not configured or missing "
                f"({self._worker_python}); set QWEN3TTS_WORKER_PYTHON to the Qwen env's python"
            )
        if not models_present(self._models_dir):
            raise SynthesisError(
                f"qwen3tts: {BASE_CHECKPOINT} not found under {self._models_dir}; "
                "download it and set QWEN3TTS_MODELS_DIR"
            )
        worker = Path(__file__).with_name("qwen3tts_worker.py")
        argv = [
            str(self._worker_python),
            str(worker),
            "--models-dir",
            str(self._models_dir),
            "--backend",
            self._backend,
            "--language",
            self._language,
            "--max-seq-len",
            str(self._max_seq_len),
        ]
        env = dict(os.environ)
        # Weights are local dirs: never reach for the hub. Unbuffered so replies aren't stuck in a
        # pipe buffer behind the JSON line.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        return SubprocessTransport(
            argv,
            env=env,
            cwd=None,
            boot_timeout=self._load_timeout,
            name="qwen3tts",
            protocol=PROTOCOL_VERSION,
        )

    def _ensure_started(self) -> WorkerTransport:
        if self._transport is None:
            self._transport = (self._factory or self._default_factory)()
            self._loaded = None
        return self._transport

    def _ensure_loaded(self, checkpoint: str) -> WorkerTransport:
        """The worker with ``checkpoint`` resident. Loading is its own request so the multi-GB
        load gets the load timeout, not a line's request timeout; the worker unloads the other
        checkpoint first (one resident at a time)."""
        transport = self._ensure_started()
        if self._loaded != checkpoint:
            reply = transport.request(
                {"cmd": "load", "model": checkpoint}, timeout=self._load_timeout
            )
            if not reply.get("ok"):
                raise WorkerError(
                    f"qwen3tts worker failed to load {checkpoint}: {reply.get('error')}",
                    oom=bool(reply.get("oom")),
                )
            self._loaded = checkpoint
        return transport

    def _reference_path(self, voice_id: str) -> Path:
        reference = self._voices_dir / voice_id / "reference.wav"
        if not reference.is_file():
            raise SynthesisError(
                f"qwen3tts: no reference.wav for voice {voice_id!r} at {reference}"
            )
        return reference

    def _reference_sha256(self, reference: Path) -> str:
        stat = reference.stat()
        key = (str(reference), stat.st_mtime_ns, stat.st_size)
        digest = self._ref_hashes.get(key)
        if digest is None:
            digest = hashlib.sha256(reference.read_bytes()).hexdigest()
            self._ref_hashes[key] = digest
        return digest

    def _reference_text(self, voice: str, reference: Path, settings: dict[str, Any]) -> str | None:
        """The reference clip's exact transcript (``reference.txt`` beside it) for ICL mode, or
        None for x-vector mode.

        ICL only when the voice's settings carry ``ref_text_digest`` (it rides the SegmentKey's
        settings_hash, so the cache always sees the mode and the transcript). Without one, a
        reference.txt is ignored: dropping one in by hand would otherwise switch the mode under
        an unchanged key, and a book would mix cached x-vector lines with new ICL ones. With
        one, the transcript must match it for THIS clip; a missing, edited or stale (re-cloned
        clip) transcript fails loudly, because ICL with the wrong transcript is worse than none
        and silently dropping to x-vector would again change the output under the same key."""
        bound = settings.get(REF_TEXT_DIGEST_KEY)
        if bound is None:
            return None
        path = reference.with_name("reference.txt")
        text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        if not text:
            raise SynthesisError(
                f"qwen3tts: voice {voice!r} binds a reference transcript but {path} is missing "
                "or empty; restage the voice (scripts/design_voices.py) or re-clone it"
            )
        if reference_text_digest(self._reference_sha256(reference), text) != float(bound):
            raise SynthesisError(
                f"qwen3tts: voice {voice!r}: {path.name} does not match the transcript its "
                "meta binds (edited by hand, or reference.wav changed); restage the voice "
                "(scripts/design_voices.py) or re-clone it"
            )
        return text

    def _run(self, message: dict[str, Any], checkpoint: str, what: str) -> tuple[np.ndarray, int]:
        """Send one generation request; on a worker OOM/death/timeout kill + restart it (fresh
        process == clean VRAM) and retry up to max_restarts times before failing loudly."""
        last_error: WorkerError | None = None
        for _attempt in range(self._max_restarts + 1):
            fd, out_path = tempfile.mkstemp(suffix=".qwen3tts.wav")
            os.close(fd)
            try:
                transport = self._ensure_loaded(checkpoint)
                message["out_path"] = out_path
                reply = transport.request(message, timeout=self._request_timeout)
                if not reply.get("ok"):
                    raise WorkerError(
                        f"qwen3tts worker {message['cmd']} failed: {reply.get('error')}",
                        oom=bool(reply.get("oom")),
                    )
                self.capped_lines += int(bool(reply.get("hit_cap")))
                samples, sample_rate = sf.read(out_path, dtype="float32", always_2d=False)
                if samples.size == 0:
                    raise SynthesisError(f"qwen3tts: worker produced empty audio ({what})")
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
            f"qwen3tts: {message['cmd']} failed after {self._max_restarts + 1} attempt(s) "
            f"({what}): {last_error}"
        ) from last_error

    # -- synthesis ----------------------------------------------------------------------------

    def _synthesize_native(
        self, text: str, voice: str, settings: dict[str, Any]
    ) -> tuple[np.ndarray, int]:
        reference = self._reference_path(voice)
        message: dict[str, Any] = {
            "cmd": "synthesize",
            "text": text,  # the worker strips a wrapping quote pair before the model sees it
            "reference_wav": str(reference),
            "reference_sha256": self._reference_sha256(reference),
            "ref_text": self._reference_text(voice, reference, settings),  # None -> x-vector
            "seed": settings.get("seed"),
        }
        # Whitelisted per-voice tunables; OMITTED when a voice sets none.
        gen = {k: settings[k] for k in _GEN_KEYS if k in settings}
        if gen:
            message["gen"] = gen
        return self._run(message, "base", f"voice {voice!r}, text {text[:80]!r}")

    def design_voice(
        self,
        text: str,
        description: str,
        *,
        seed: int,
        settings: dict[str, Any] | None = None,
    ) -> AudioFile:
        """Speak ``text`` in a NEW voice described by ``description`` (VoiceDesign checkpoint).

        The one-off voice-creation step: the result is saved as a voice's reference clip and
        cloned with the Base checkpoint from then on. Swaps the worker to VoiceDesign (the Base
        checkpoint is unloaded first); the next render line swaps back. Callers hold the GPU
        through the GpuResourceManager exactly as for synthesis."""
        if not text.strip() or not description.strip():
            raise SynthesisError("qwen3tts: voice design needs both text and a description")
        message: dict[str, Any] = {
            "cmd": "design",
            "text": text,
            "instruct": description,
            "seed": int(seed),
        }
        gen = {k: settings[k] for k in _GEN_KEYS if settings and k in settings}
        if gen:
            message["gen"] = gen
        samples, sample_rate = self._run(message, "design", f"design {description[:60]!r}")
        return to_canonical(samples, sample_rate)
