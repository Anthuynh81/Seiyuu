"""TTS engines: pipeline code gets engines ONLY via get_engine()."""

import importlib

from seiyuu.engines.audio import CANONICAL_SAMPLE_RATE, AudioFile, to_canonical
from seiyuu.engines.base import EngineVoice, SynthesisError, TTSEngine

# Engine classes are referenced as strings and imported lazily so that
# importing seiyuu.engines never pulls in an engine SDK.
_ENGINES = {
    "kokoro": "seiyuu.engines.kokoro_engine:KokoroEngine",
}


def get_engine(engine_id: str, **kwargs) -> TTSEngine:
    if engine_id not in _ENGINES:
        raise ValueError(f"unknown TTS engine {engine_id!r}; available: {sorted(_ENGINES)}")
    module_name, class_name = _ENGINES[engine_id].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**kwargs)


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "AudioFile",
    "EngineVoice",
    "SynthesisError",
    "TTSEngine",
    "get_engine",
    "to_canonical",
]
