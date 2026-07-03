"""TTS engines: pipeline code gets engines ONLY via get_engine()."""

import importlib

from seiyuu.engines.audio import CANONICAL_SAMPLE_RATE, AudioFile, to_canonical
from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine

# Engine classes are referenced as strings and imported lazily so that
# importing seiyuu.engines never pulls in an engine SDK.
_ENGINES = {
    "kokoro": "seiyuu.engines.kokoro_engine:KokoroEngine",
    "chatterbox": "seiyuu.engines.chatterbox_engine:ChatterboxEngine",
    "elevenlabs": "seiyuu.engines.elevenlabs_engine:ElevenLabsEngine",
}


def get_engine_class(engine_id: str) -> type[TTSEngine]:
    """The adapter class WITHOUT instantiation — catalog facts (uses_gpu,
    requires_validation) for the API. Imports the adapter module, never an SDK
    (those stay deferred inside the adapters)."""
    if engine_id not in _ENGINES:
        raise ValueError(f"unknown TTS engine {engine_id!r}; available: {sorted(_ENGINES)}")
    module_name, class_name = _ENGINES[engine_id].split(":")
    return getattr(importlib.import_module(module_name), class_name)


def get_engine(engine_id: str, **kwargs) -> TTSEngine:
    return get_engine_class(engine_id)(**kwargs)


def list_engine_ids() -> list[str]:
    return sorted(_ENGINES)


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "AudioFile",
    "EngineVoice",
    "SynthesisError",
    "TTSEngine",
    "get_engine",
    "get_engine_class",
    "list_engine_ids",
    "to_canonical",
]
