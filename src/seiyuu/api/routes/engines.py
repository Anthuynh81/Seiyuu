"""GET /engines and /engines/{id}/voices — static catalog facts + preset listings.

Catalog facts import adapter CLASSES only (SDK imports stay deferred inside the
adapters). The ElevenLabs listing is a live paid-tier SDK call (free itself) cached
in-process for 60s; it needs the key and maps SDK failures to 502 ``upstream``.
POST /engines/{id}/warmup is M6b-2 — it needs the ``warmup`` job kind.
"""

import hashlib
import json
import os
import time
from contextlib import ExitStack

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse
from pydantic import ValidationError

from seiyuu.api.deps import BrokerDep, GateDep, RegistryDep, RunnerDep, SettingsDep, StoreDep
from seiyuu.api.enqueue import enqueue_job
from seiyuu.api.errors import ApiError
from seiyuu.api.registry import ENGINE_FACTS, catalog_ids, weights_cached

# shared audition-refusal predicate + BORROW branch so voices.py and engines.py can't drift
from seiyuu.api.routes.voices import Verdict, _refuse_conflicts, borrow_and_synthesize
from seiyuu.api.schemas import (
    AUDITION_DEFAULT_TEXT,
    EngineInfo,
    EnginesOut,
    EngineVoiceOut,
    EngineVoicesOut,
    JobOut,
    WarmupParams,
)
from seiyuu.engines import SynthesisError, get_engine_class
from seiyuu.gpu import get_gpu_manager
from seiyuu.normalize import normalize_text, profile_for
from seiyuu.repository import JobKind, JobState
from seiyuu.voices import render_voice_args
from seiyuu.voices.blends import canonical_recipe
from seiyuu.voices.models import BlendComponent, VoiceKind, VoiceMeta

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


# -- preset / mix preview (kokoro only) ----------------------------------------------------


def _parse_preview_recipe(preset: str | None, components: str | None) -> list[tuple[str, float]]:
    """?preset=af_heart XOR ?components=af_heart:70,af_nicole:30 -> canonical recipe."""
    if (preset is None) == (components is None):
        raise ApiError(422, "invalid", "pass exactly one of ?preset= or ?components=")
    if preset is not None:
        return [(preset, 1.0)]
    pairs: list[tuple[str, float]] = []
    for part in components.split(","):  # type: ignore[union-attr]
        pid, sep, weight = part.strip().partition(":")
        if not sep or not pid:
            raise ApiError(422, "invalid", f"bad component {part!r}; expected preset:weight")
        try:
            pairs.append((pid, float(weight)))
        except ValueError as exc:
            raise ApiError(422, "invalid", f"bad weight in {part!r}") from exc
    try:
        return canonical_recipe(pairs)
    except ValueError as exc:
        raise ApiError(422, "invalid", str(exc)) from exc


@router.get("/engines/{engine_id}/preview")
def preview_voice(
    engine_id: str,
    request: Request,
    cfg: SettingsDep,
    registry: RegistryDep,
    store: StoreDep,
    gate: GateDep,
    broker: BrokerDep,
    preset: str | None = None,
    components: str | None = None,
) -> FileResponse:
    """Demo a preset or an ad-hoc blend recipe WITHOUT creating a library voice — the
    mixer's ear. Free and local by construction (kokoro only); results are cached by
    recipe under data/previews so repeat demos never touch the GPU."""
    _require_known(engine_id)
    if engine_id != "kokoro":
        raise ApiError(
            422, "invalid", "preview supports kokoro only; audition library voices instead"
        )
    recipe = _parse_preview_recipe(preset, components)
    known = {v.id for v in registry.get("kokoro").list_voices()}
    for pid, _ in recipe:
        if pid not in known:
            raise ApiError(422, "invalid", f"unknown kokoro preset {pid!r}")

    # the ephemeral meta enforces the same domain rules as voice creation (e.g. no
    # cross-family blends) and feeds render_voice_args exactly like a saved voice
    try:
        if len(recipe) == 1 and recipe[0][1] == 1.0:
            meta = VoiceMeta(
                voice_id="preview", name="Preview", kind=VoiceKind.PRESET,
                engine="kokoro", preset_id=recipe[0][0], source="preview",
            )  # fmt: skip
        else:
            meta = VoiceMeta(
                voice_id="preview", name="Preview", kind=VoiceKind.BLEND, engine="kokoro",
                blend=[BlendComponent(preset_id=p, weight=w) for p, w in recipe],
                source="preview",
            )  # fmt: skip
    except ValidationError as exc:
        first = str(exc.errors()[0]["msg"]).removeprefix("Value error, ")
        raise ApiError(422, "invalid", first) from exc

    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    out_path = cfg.data_dir / "previews" / f"kokoro_{key}.wav"
    if out_path.is_file():
        return FileResponse(out_path, media_type="audio/wav")

    with ExitStack() as stack:
        with request.app.state.enqueue_mutex:
            # same predicate as auditions (gpu_busy incl. queued GPU jobs, engine_cold
            # with the warmup recourse, and the F1 BORROW verdict), under the same mutex
            live = store.list_jobs(states=[JobState.QUEUED, JobState.RUNNING])
            verdict = _refuse_conflicts(meta, get_engine_class("kokoro"), live, registry, broker)
            if not stack.enter_context(request.app.state.audition_slot.try_hold()):
                raise ApiError(
                    409, "audition_in_flight", "another audition is already synthesizing"
                )
            # BORROW rides a running kokoro render's resident engine and SKIPS the gate.
            if verdict is Verdict.EXCLUSIVE and not stack.enter_context(gate.try_hold("audition")):
                running = next((j for j in live if j.state is JobState.RUNNING), None)
                raise ApiError(
                    409,
                    "gpu_busy",
                    "a job currently holds the GPU; wait for it or cancel it",
                    detail=(JobOut.from_job(running).model_dump(mode="json") if running else None),
                )
        engine = registry.get("kokoro")
        engine_voice, settings = render_voice_args(meta)
        normalized = normalize_text(AUDITION_DEFAULT_TEXT, profile=profile_for("kokoro"))
        synth = lambda eng: eng.synthesize(  # noqa: E731
            normalized, engine_voice, {**settings, "seed": meta.seed}
        )
        try:
            if verdict is Verdict.BORROW:
                audio = borrow_and_synthesize(
                    broker, "kokoro", cfg.borrow_grant_timeout_s, live, synth
                )
            else:
                with get_gpu_manager().acquire(engine, "engine:kokoro"):
                    audio = synth(engine)
                # stays lazily resident, same as auditions — the next preview is warm
        except SynthesisError as exc:
            raise ApiError(502, "upstream", str(exc)) from exc
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(f"kokoro_{key}.part.wav")
        audio.save(tmp)
        os.replace(tmp, out_path)
        return FileResponse(out_path, media_type="audio/wav")
