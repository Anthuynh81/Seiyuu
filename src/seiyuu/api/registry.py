"""Process-lifetime engine instances (scoping doc section 1, lifespan step 2).

One instance per engine id, shared by auditions, warmup jobs, and the single-voice
render handler, so the GPU manager's identity comparison makes re-acquire a no-op
instead of a multi-GB reload. The chatterbox engine is constructed with
``voices_dir = settings.voices_dir`` — the SAME root the ``VoiceLibrary`` uses — so
``verify_consent`` and the engine's in-engine consent check see identical
``reference.wav`` bytes; constructing it any other way silently bypasses the clone
consent gate. ``invalidate`` only drops the cached instance (clone endpoint: stale
per-run ``_ref_hashes`` must not survive a re-clone) — it never calls ``unload()``;
VRAM lifecycle belongs to the GPU manager alone (lazy release on competitor acquire).

Residency is never a flag: ``is_resident`` identity-compares the registry's instance
against the GPU manager's resident consumer, so an attribution run (or any competitor
acquire) that evicts a warmed engine flips it back to cold automatically. A stale
"resident" answer would make the M6b-6 cold-engine refusal skip the warmup job and pin
a request thread on a multi-GB load — the exact failure the refusal exists to prevent.
"""

import os
import threading
from pathlib import Path

from seiyuu.engines import TTSEngine, get_engine, list_engine_ids, voices_dir_kwargs
from seiyuu.gpu import GpuResourceManager, get_gpu_manager
from seiyuu.settings import Settings, get_settings

# Static catalog facts with no home on the adapter classes; uses_gpu/requires_validation
# come from the classes themselves (seiyuu.engines.get_engine_class).
ENGINE_FACTS: dict[str, dict[str, bool]] = {
    "kokoro": {"paid": False, "supports_cloning": False},
    "chatterbox": {"paid": False, "supports_cloning": True},
    "indextts2": {"paid": False, "supports_cloning": True},
    "elevenlabs": {"paid": True, "supports_cloning": True},
}

# HF hub cache dir-name needles for the best-effort weights_cached probe.
_HF_NEEDLES = {"kokoro": "kokoro-82m", "chatterbox": "chatterbox"}


class EngineRegistry:
    def __init__(self, settings: Settings, gpu_manager: GpuResourceManager | None = None) -> None:
        self._settings = settings
        self._gpu = gpu_manager or get_gpu_manager()  # injectable for tests; global otherwise
        self._lock = threading.Lock()
        self._engines: dict[str, TTSEngine] = {}

    def get(self, engine_id: str) -> TTSEngine:
        """The shared instance, constructed on first use. Raises ValueError on an
        unknown id (routes map it to 404)."""
        with self._lock:
            engine = self._engines.get(engine_id)
            if engine is None:
                engine = get_engine(engine_id, **self._construct_kwargs(engine_id))
                self._engines[engine_id] = engine
            return engine

    def _construct_kwargs(self, engine_id: str) -> dict:
        # Consent invariant — see module docstring. Cloning engines (chatterbox, indextts2) must
        # see the SAME voices_dir the VoiceLibrary uses; voices_dir_kwargs tolerates an unknown
        # (test-injected) id by returning {}.
        kwargs: dict = dict(voices_dir_kwargs(engine_id, self._settings.voices_dir))
        if engine_id == "indextts2":
            # Explicit kwargs so the injected Settings governs the out-of-process worker config
            # (the adapter's own fallback reads the global get_settings(), which must not engage
            # here). See engines/indextts2_engine.py.
            kwargs.update(
                checkpoints_dir=self._settings.indextts2_checkpoints_dir,
                worker_python=self._settings.indextts2_worker_python,
                use_fp16=self._settings.indextts2_use_fp16,
                load_timeout=self._settings.indextts2_worker_load_timeout,
                request_timeout=self._settings.indextts2_worker_request_timeout,
                max_restarts=self._settings.indextts2_worker_max_restarts,
            )
            return kwargs
        if engine_id == "chatterbox":
            return kwargs
        if engine_id == "elevenlabs":
            # Explicit kwargs so the injected Settings governs; the adapter's own
            # fallback reads the global get_settings(), which must never engage here.
            # api_key falls back on None specifically — "" keeps the key unconfigured
            # (a paid client cannot be constructed) without re-opening that path.
            return {
                "api_key": self._settings.elevenlabs_api_key or "",
                "model_id": self._settings.elevenlabs_model_id,
                "price_per_1k_chars": self._settings.elevenlabs_price_per_1k_chars,
            }
        return {}

    def invalidate(self, engine_id: str) -> None:
        """Drop the cached instance. No unload() here — the GPU manager still tracks
        the old instance and frees it on the next competitor acquire; residency for the
        NEW instance reads False by identity until it is warmed."""
        with self._lock:
            self._engines.pop(engine_id, None)

    def is_resident(self, engine_id: str) -> bool:
        """True iff THIS registry's instance is the GPU manager's resident consumer —
        truth by identity, never a flag that could go stale on eviction."""
        with self._lock:
            engine = self._engines.get(engine_id)
        return engine is not None and self._gpu.holds(engine)


def weights_cached(engine_id: str) -> bool | None:
    """Best-effort HF hub cache probe; None means unknowable (cloud engines, odd cache).
    A False for a downloaded model is cosmetic (the UI shows a download warning), so a
    substring scan of ``models--*`` dir names is deliberately good enough."""
    if engine_id == "indextts2":
        # IndexTTS-2 weights are a downloaded checkpoints dir (not the HF hub cache).
        from seiyuu.engines.indextts2_engine import checkpoints_present

        try:
            return checkpoints_present(get_settings().indextts2_checkpoints_dir)
        except OSError:
            return None
    needle = _HF_NEEDLES.get(engine_id)
    if needle is None:
        return None
    try:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
        return any(needle in d.name.lower() for d in hub.glob("models--*"))
    except OSError:
        return None


def catalog_ids() -> list[str]:
    return list_engine_ids()
