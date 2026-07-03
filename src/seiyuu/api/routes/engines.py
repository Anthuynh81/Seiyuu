"""GET /engines and /engines/{id}/voices — static catalog facts + preset listings.

Catalog facts import adapter CLASSES only (SDK imports stay deferred inside the
adapters). The ElevenLabs listing is a live paid-tier SDK call (free itself) cached
in-process for 60s; it needs the key and maps SDK failures to 502 ``upstream``.
POST /engines/{id}/warmup is M6b-2 — it needs the ``warmup`` job kind.
"""

import time

from fastapi import APIRouter, Request, Response

from seiyuu.api.deps import RegistryDep, RunnerDep, SettingsDep, StoreDep
from seiyuu.api.enqueue import enqueue_job
from seiyuu.api.errors import ApiError
from seiyuu.api.registry import ENGINE_FACTS, catalog_ids, weights_cached
from seiyuu.api.schemas import (
    EngineInfo,
    EnginesOut,
    EngineVoiceOut,
    EngineVoicesOut,
    JobOut,
    WarmupParams,
)
from seiyuu.engines import get_engine_class
from seiyuu.repository import JobKind

router = APIRouter(tags=["engines"])

_VOICES_CACHE_TTL_SECONDS = 60.0
# engine_id -> (monotonic timestamp, voices). Process-local like every other cache here.
_voices_cache: dict[str, tuple[float, list[EngineVoiceOut]]] = {}


def _require_known(engine_id: str) -> None:
    if engine_id not in ENGINE_FACTS:
        raise ApiError(
            404, "not_found", f"unknown engine {engine_id!r}; available: {catalog_ids()}"
        )


@router.get("/engines", response_model=EnginesOut)
def list_engines(registry: RegistryDep) -> EnginesOut:
    infos = []
    for engine_id in catalog_ids():
        cls = get_engine_class(engine_id)
        facts = ENGINE_FACTS[engine_id]
        infos.append(
            EngineInfo(
                engine_id=engine_id,
                uses_gpu=cls.uses_gpu,
                requires_validation=cls.requires_validation,
                paid=facts["paid"],
                supports_cloning=facts["supports_cloning"],
                weights_cached=weights_cached(engine_id),
                resident=registry.is_resident(engine_id),
            )
        )
    return EnginesOut(engines=infos)


@router.post("/engines/{engine_id}/warmup", response_model=JobOut, status_code=202)
def warmup_engine(
    engine_id: str,
    request: Request,
    response: Response,
    store: StoreDep,
    runner: RunnerDep,
) -> JobOut:
    """Pre-load a GPU engine's weights as a job (the sanctioned path for first-ever
    engine use — sync auditions refuse cold engines instead of pinning a request thread
    for a multi-GB download with no cancel story). The model stays lazily resident."""
    _require_known(engine_id)
    if not get_engine_class(engine_id).uses_gpu:
        raise ApiError(
            409, "nothing_to_warm", f"{engine_id} is a cloud engine; it holds no local weights"
        )
    job = enqueue_job(
        store=store,
        runner=runner,
        mutex=request.app.state.enqueue_mutex,
        book_id=f"engine:{engine_id}",  # documented subject-id overload of the book column
        kind=JobKind.WARMUP,
        params=WarmupParams(engine_id=engine_id).model_dump(),
    )
    response.headers["Location"] = f"/api/jobs/{job.job_id}"
    return JobOut.from_job(job)


@router.get("/engines/{engine_id}/voices", response_model=EngineVoicesOut)
def engine_voices(engine_id: str, cfg: SettingsDep, registry: RegistryDep) -> EngineVoicesOut:
    _require_known(engine_id)
    if engine_id == "elevenlabs":
        if not cfg.elevenlabs_api_key:
            raise ApiError(
                503,
                "not_ready",
                "ELEVENLABS_API_KEY not set; configure it to list ElevenLabs voices",
            )
        cached = _voices_cache.get(engine_id)
        if cached is not None and time.monotonic() - cached[0] < _VOICES_CACHE_TTL_SECONDS:
            return EngineVoicesOut(engine_id=engine_id, voices=cached[1])
        try:
            voices = registry.get(engine_id).list_voices()
        except Exception as exc:
            raise ApiError(502, "upstream", f"elevenlabs: {exc}") from exc
        out = [EngineVoiceOut(**v.model_dump()) for v in voices]
        _voices_cache[engine_id] = (time.monotonic(), out)
        return EngineVoicesOut(engine_id=engine_id, voices=out)
    # kokoro: 28 hardcoded presets, no weights load; chatterbox: [] (clones live in the
    # voice library, not the engine).
    voices = registry.get(engine_id).list_voices()
    return EngineVoicesOut(
        engine_id=engine_id, voices=[EngineVoiceOut(**v.model_dump()) for v in voices]
    )
