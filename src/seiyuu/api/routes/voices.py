"""Voice Studio: library CRUD, cloning (purge-on-reclone per sign-off Q1), the
synchronous audition with its refusal predicate, and the read-only cloud-slot view.

Auditions are the one synchronous GPU path: they claim the heavy-work gate
NON-blockingly (busy -> instant 409, never a stalled request thread) with the busy
check and the gate claim made atomic against job creation by sharing the enqueue
mutex. Cold GPU engines are refused toward the warmup job instead of pinning a request
thread on a multi-GB download. The model stays lazily resident afterwards — that is
what makes the next audition (or a single-voice render) an identity no-op re-acquire.
"""

import json
import os
import re
import secrets
import shutil
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from seiyuu.api.concurrency import BorrowBroker
from seiyuu.api.deps import BrokerDep, GateDep, RegistryDep, SettingsDep, StoreDep
from seiyuu.api.errors import ApiError
from seiyuu.api.registry import weights_cached
from seiyuu.api.schemas import (
    AuditionOut,
    AuditionRequest,
    BlendVoiceCreate,
    CloudSlotOut,
    CloudSlotsOut,
    JobOut,
    PresetVoiceCreate,
    UnreadableVoice,
    VoiceCreate,
    VoiceDeletedOut,
    VoiceDetailOut,
    VoiceListOut,
    VoiceOut,
    VoiceReferencesOut,
    VoiceUpdate,
)
from seiyuu.engines import SynthesisError, get_engine_class, list_engine_ids
from seiyuu.gpu import GpuBusyError, get_gpu_manager
from seiyuu.normalize import normalize_text, profile_for
from seiyuu.repository import Job, JobKind, JobState
from seiyuu.services import ServiceError, delete_voice, voice_references
from seiyuu.voices import (
    BlendComponent,
    CloudVoiceError,
    CloudVoiceRegistry,
    ConsentAttestation,
    VoiceKind,
    VoiceLibrary,
    VoiceLibraryError,
    VoiceMeta,
    auto_blend_recipe,
    canonical_recipe,
    ensure_cloud_voice,
    render_voice_args,
    sha256_file,
)
from seiyuu.voices.cloud import REGISTRY_NAME

router = APIRouter(tags=["voices"])

_VOICE_ID_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UPLOAD_CHUNK = 1024 * 1024

# The audition refusal predicate is metadata-driven, not a kind-name list (scoping doc):
# which job kinds hold the GPU, and which touch ElevenLabs voice slots.
GPU_JOB_KINDS = frozenset({JobKind.ATTRIBUTE, JobKind.RENDER, JobKind.WARMUP})
CLOUD_SLOT_JOB_KINDS = frozenset({JobKind.RENDER})


def _check_voice_id(voice_id: str) -> None:
    if not _VOICE_ID_OK.match(voice_id) or ".." in voice_id:
        raise ApiError(422, "invalid", f"invalid voice id {voice_id!r}")


def _load_or_404(library: VoiceLibrary, voice_id: str) -> VoiceMeta:
    try:
        return library.load(voice_id)
    except VoiceLibraryError as exc:  # missing, or dir/meta id mismatch
        raise ApiError(404, "not_found", str(exc)) from exc


def _audition_path(library: VoiceLibrary, voice_id: str) -> Path:
    return library.dir_for(voice_id) / "audition.wav"


def _voice_out(library: VoiceLibrary, meta: VoiceMeta) -> VoiceOut:
    return VoiceOut(
        **meta.model_dump(), has_audition=_audition_path(library, meta.voice_id).is_file()
    )


def _guard_live_render(store, action: str) -> None:
    """Replace-clone and voice deletion mutate state a RUNNING render depends on
    (cached segments the manifest will reference; the consent-attested reference.wav a
    single-voice job reads mid-run — which never appears in assignments.json, so the
    referential guard cannot see it). Refuse globally while any render job is live,
    mirroring the render_active guard on edits/assignment writes."""
    live = store.list_jobs(states=[JobState.QUEUED, JobState.RUNNING])
    render = next((j for j in live if j.kind is JobKind.RENDER), None)
    if render is not None:
        raise ApiError(
            409,
            "render_active",
            f"a render job is {render.state.value}; {action} would corrupt its cached "
            "segments or pull its voice out from under it — wait or cancel it first",
            detail=JobOut.from_job(render).model_dump(mode="json"),
        )


def _drop_cloud_handle(cfg, voice_id: str) -> None:
    """Remove the voice's IVC handle from the slot registry (re-clone/delete): the
    cached cloud voice was trained on the OLD reference audio, and ensure_cloud_voice
    would keep returning it — paid synthesis in the previously-attested speaker's
    voice. The remote voice itself is left for slot-pressure eviction to reap."""
    if not (cfg.voices_dir / REGISTRY_NAME).is_file():
        return
    with CloudVoiceRegistry(cfg.voices_dir).locked(timeout=30.0) as registry:
        registry.remove(voice_id)


# -- library CRUD -------------------------------------------------------------------------


@router.get("/voices", response_model=VoiceListOut)
def list_voices(cfg: SettingsDep) -> VoiceListOut:
    library = VoiceLibrary(cfg.voices_dir)
    voices: list[VoiceOut] = []
    unreadable: list[UnreadableVoice] = []
    if cfg.voices_dir.is_dir():
        for entry in sorted(cfg.voices_dir.iterdir()):
            if not (entry / "meta.json").is_file():
                continue
            try:
                voices.append(_voice_out(library, library.load(entry.name)))
            except (VoiceLibraryError, ValidationError, OSError, ValueError) as exc:
                unreadable.append(UnreadableVoice(voice_id=entry.name, error=str(exc)))
    return VoiceListOut(voices=voices, unreadable=unreadable)


@router.post("/voices", response_model=VoiceOut, status_code=201)
def create_voice(
    body: VoiceCreate, request: Request, response: Response, cfg: SettingsDep
) -> VoiceOut:
    if isinstance(body, PresetVoiceCreate) and body.engine not in list_engine_ids():
        # an arbitrary engine string would 201 here and then 500 on every later use
        raise ApiError(
            422, "invalid", f"unknown engine {body.engine!r}; available: {list_engine_ids()}"
        )
    library = VoiceLibrary(cfg.voices_dir)
    voice_id = body.voice_id or library.new_voice_id(body.name)
    _check_voice_id(voice_id)
    with request.app.state.voices_mutex:
        if library.meta_path(voice_id).is_file():
            raise ApiError(409, "voice_exists", f"voice {voice_id!r} already exists")
        # VoiceMeta construction enforces domain rules (e.g. a kokoro blend can't mix
        # language families) — those must surface as 422s, not unhandled 500s.
        try:
            if isinstance(body, PresetVoiceCreate):
                meta = VoiceMeta(
                    voice_id=voice_id,
                    name=body.name,
                    kind=VoiceKind.PRESET,
                    engine=body.engine,
                    preset_id=body.preset_id,
                    seed=body.seed,
                    source="preset",
                )
            else:
                assert isinstance(body, BlendVoiceCreate)
                if body.components is not None:
                    recipe = canonical_recipe([(c.preset_id, c.weight) for c in body.components])
                    source = "manual_blend"
                else:
                    recipe = auto_blend_recipe(body.name, body.gender, accent=body.accent)
                    source = "auto_blend"
                meta = VoiceMeta(
                    voice_id=voice_id,
                    name=body.name,
                    kind=VoiceKind.BLEND,
                    engine="kokoro",
                    blend=[BlendComponent(preset_id=p, weight=w) for p, w in recipe],
                    seed=body.seed,
                    source=source,
                )
            library.save(meta)
        except ValidationError as exc:
            first = str(exc.errors()[0]["msg"]).removeprefix("Value error, ")
            raise ApiError(422, "invalid", first) from exc
        except (VoiceLibraryError, ValueError) as exc:
            raise ApiError(422, "invalid", str(exc)) from exc
    response.headers["Location"] = f"/api/voices/{voice_id}"
    return _voice_out(library, meta)


def _purge_cached_segments(output_dir: Path, voice_id: str) -> int:
    """Sign-off Q1 (purge-on-reclone): delete every cached segment for this voice across
    all books, via the cache's SegmentKey sidecars. Stale audio from the OLD reference
    can then never replay; previously-paid segments re-bill on the next render (acked)."""
    removed = 0
    if not output_dir.is_dir():
        return removed
    for book_dir in output_dir.iterdir():
        cache_dir = book_dir / "cache"
        if not cache_dir.is_dir():
            continue
        for sidecar in cache_dir.glob("*.json"):
            if sidecar.name.endswith(".validation.json"):
                continue
            try:
                key = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # torn sidecar: its wav is unmatchable, leave for manual cleanup
            if key.get("voice_id") != voice_id:
                continue
            stem = sidecar.stem
            (cache_dir / f"{stem}.wav").unlink(missing_ok=True)
            (cache_dir / f"{stem}.validation.json").unlink(missing_ok=True)
            (cache_dir / f"{stem}.words.json").unlink(missing_ok=True)  # F2 alignment sidecar
            sidecar.unlink(missing_ok=True)
            removed += 1
    return removed


@router.post("/voices/clone", response_model=VoiceOut, status_code=201)
def clone_voice(
    request: Request,
    response: Response,
    cfg: SettingsDep,
    registry: RegistryDep,
    store: StoreDep,
    file: Annotated[UploadFile, File()],
    name: Annotated[str, Form(min_length=1)],
    engine: Annotated[str, Form(pattern="^(chatterbox|elevenlabs|indextts2)$")] = "chatterbox",
    consent: Annotated[bool, Form()] = False,
    attested_by: Annotated[str, Form()] = "",
    seed: Annotated[int, Form()] = 41172,
    voice_id: Annotated[str | None, Form()] = None,
    replace: Annotated[bool, Form()] = False,
) -> VoiceOut:
    """Clone from an uploaded reference clip. The load-bearing ordering from the CLI is
    preserved exactly: copy -> hash the COPY -> publish atomically -> purge stale conds
    -> save meta, so the attestation provably describes the bytes the library serves and
    a failed copy can never leave an attested meta over missing audio."""
    if not consent:
        raise ApiError(
            422,
            "invalid",
            "cloning requires consent=true: you must hold the rights/permission "
            "to clone this voice",
        )
    if not attested_by.strip():
        raise ApiError(
            422, "invalid", "attested_by is required: who is making the consent attestation"
        )
    library = VoiceLibrary(cfg.voices_dir)
    vid = voice_id or library.new_voice_id(name)
    _check_voice_id(vid)

    upload_dir = cfg.data_dir / "uploads" / secrets.token_hex(8)
    upload_dir.mkdir(parents=True, exist_ok=True)
    staged = upload_dir / "reference-upload"
    try:
        size = 0
        with staged.open("wb") as out:
            while chunk := file.file.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > cfg.max_upload_bytes:
                    raise ApiError(
                        413,
                        "payload_too_large",
                        f"upload exceeds the {cfg.max_upload_bytes}-byte limit",
                    )
                out.write(chunk)
        if size == 0:
            raise ApiError(422, "invalid", "empty reference upload")

        with request.app.state.voices_mutex:
            exists = library.meta_path(vid).is_file() or library.reference_path(vid).is_file()
            if exists and not replace:
                raise ApiError(
                    409,
                    "reclone_blocked",
                    f"voice {vid!r} already exists; re-clone replaces its consent-attested "
                    "reference audio — re-send with replace=true to confirm (cached "
                    "segments for this voice are purged and paid ones re-bill)",
                )
            if exists:
                # a running render may be reading/writing exactly the cache files the
                # purge deletes (its manifest would reference missing wavs)
                _guard_live_render(store, "re-cloning this voice")
                # Q1 DECIDED (purge-on-reclone): stale audio out BEFORE the new
                # reference is published, so no window serves old-voice segments.
                _purge_cached_segments(cfg.output_dir, vid)
                _audition_path(library, vid).unlink(missing_ok=True)
                # the IVC handle was trained on the OLD reference; keeping it would let
                # paid synthesis keep speaking the previously-attested speaker
                _drop_cloud_handle(cfg, vid)
            ref_path = library.reference_path(vid)
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ref_path.with_name("reference.wav.part")
            shutil.copyfile(staged, tmp)
            attestation = ConsentAttestation(
                attested_by=attested_by.strip(),
                reference_sha256=sha256_file(tmp),  # hash the COPY: consent binds to it
            )
            os.replace(tmp, ref_path)
            # conds derived from the OLD reference must never speak again (their filename
            # embeds the ref hash, so they are unreachable anyway — removing them is hygiene)
            for stale in ref_path.parent.glob("conds_*.pt"):
                stale.unlink(missing_ok=True)
            library.save(
                VoiceMeta(
                    voice_id=vid,
                    name=name,
                    kind=VoiceKind.CLONED,
                    engine=engine,
                    reference_audio="reference.wav",
                    consent_attested=True,
                    consent=attestation,
                    seed=seed,
                    source="user_upload",
                )
            )
            # the shared cloning-engine instances cache per-run reference state and must not let
            # a warm engine keep speaking the old speaker after a re-clone: chatterbox caches
            # per-run reference hashes; indextts2's worker caches the speaker cond in-memory keyed
            # by reference PATH (unchanged on re-clone), so dropping the instance forces a fresh
            # worker with a cond recomputed from the new reference.wav.
            registry.invalidate("chatterbox")
            registry.invalidate("indextts2")
        response.headers["Location"] = f"/api/voices/{vid}"
        return _voice_out(library, library.load(vid))
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


@router.get("/voices/{voice_id}", response_model=VoiceDetailOut)
def voice_detail(voice_id: str, cfg: SettingsDep) -> VoiceDetailOut:
    _check_voice_id(voice_id)
    library = VoiceLibrary(cfg.voices_dir)
    meta = _load_or_404(library, voice_id)
    has_audition = _audition_path(library, voice_id).is_file()
    return VoiceDetailOut(
        **meta.model_dump(),
        has_audition=has_audition,
        audition_url=f"/api/voices/{voice_id}/audition.wav" if has_audition else None,
    )


@router.patch("/voices/{voice_id}", response_model=VoiceOut)
def update_voice(voice_id: str, body: VoiceUpdate, request: Request, cfg: SettingsDep) -> VoiceOut:
    """Rename and/or re-tag a voice — the library's only mutable fields. Both are optional
    and applied independently. Neither name (a pure label; Characters reference voice_id)
    nor tags feed any render cache key, so no cached audio can drift; recipe/seed/consent
    stay immutable by design."""
    _check_voice_id(voice_id)
    library = VoiceLibrary(cfg.voices_dir)
    tags: list[str] | None = None
    if body.tags is not None:
        tags = []
        seen: set[str] = set()
        for raw in body.tags:
            tag = raw.strip()
            if not tag or len(tag) > 40:
                raise ApiError(422, "invalid", f"bad tag {raw!r} (1-40 characters after trim)")
            if tag.lower() in seen:
                continue
            seen.add(tag.lower())
            tags.append(tag)
    name: str | None = None
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise ApiError(422, "invalid", "name must not be blank")
    if name is None and tags is None:
        raise ApiError(422, "invalid", "nothing to update — provide name and/or tags")
    with request.app.state.voices_mutex:
        meta = _load_or_404(library, voice_id)
        if name is not None:
            meta.name = name
        if tags is not None:
            meta.tags = tags
        try:
            library.save(meta)
        except (VoiceLibraryError, ValueError) as exc:
            raise ApiError(422, "invalid", str(exc)) from exc
    return _voice_out(library, meta)


@router.get("/voices/{voice_id}/references", response_model=VoiceReferencesOut)
def references(voice_id: str, cfg: SettingsDep) -> VoiceReferencesOut:
    """The delete-confirmation scan: every assignment role still using this voice."""
    _check_voice_id(voice_id)
    library = VoiceLibrary(cfg.voices_dir)
    _load_or_404(library, voice_id)
    try:
        refs = voice_references(voice_id, cfg.output_dir)
    except ServiceError as exc:  # fail-closed: an unreadable assignments.json
        raise ApiError(500, "corrupt_artifact", str(exc)) from exc
    return VoiceReferencesOut(voice_id=voice_id, references=refs)


@router.delete("/voices/{voice_id}", response_model=VoiceDeletedOut)
def remove_voice(
    voice_id: str, request: Request, cfg: SettingsDep, store: StoreDep
) -> VoiceDeletedOut:
    """Irreversible rmtree including the consent-attested reference.wav — DELETE is the
    explicit confirmation; the UI shows /references first. Serialized against assignment
    writes by the voices mutex so a voice can't vanish under a PUT's validation, and
    refused while a render job is live: a single-voice render's voice never appears in
    assignments.json, so the referential guard alone cannot protect it."""
    _check_voice_id(voice_id)
    library = VoiceLibrary(cfg.voices_dir)
    with request.app.state.voices_mutex:
        _load_or_404(library, voice_id)
        _guard_live_render(store, "deleting this voice")
        try:
            gone = delete_voice(voice_id, library=library, output_dir=cfg.output_dir)
        except ServiceError as exc:
            message = str(exc)
            if "cannot verify voice references" in message:
                raise ApiError(500, "corrupt_artifact", message) from exc
            if "still assigned" in message:
                raise ApiError(409, "voice_referenced", message) from exc
            raise ApiError(422, "invalid", message) from exc
        _drop_cloud_handle(cfg, voice_id)  # free the tier-limited slot registry entry
    return VoiceDeletedOut(deleted=gone.name)


# -- audition -----------------------------------------------------------------------------


class Verdict(Enum):
    """Three-outcome admission decision for a synchronous GPU audition (F1)."""

    EXCLUSIVE = "exclusive"  # no live GPU job — take the heavy-work gate as before
    BORROW = "borrow"  # a live RENDER is lending this exact engine — ride between segments


def _refuse_conflicts(
    meta: VoiceMeta,
    engine_cls,
    live: list[Job],
    registry,
    broker: BorrowBroker | None = None,
) -> Verdict:
    """The scoping-doc refusal predicate, evaluated under the enqueue mutex so the
    check-and-refuse is atomic against job creation.

    Returns a :class:`Verdict`: EXCLUSIVE (proceed via the heavy-work gate) or BORROW (a
    live RENDER is lending this exact engine, so ride its resident instance between
    segments). Any hard conflict still raises: a non-lending GPU job (ATTRIBUTE/WARMUP, or
    a RENDER on a different engine) → ``gpu_busy``; a render holding cloud slots →
    ``cloud_busy``; a never-loaded local engine → ``engine_cold``."""
    if engine_cls.uses_gpu:
        gpu_job = next((j for j in live if j.kind in GPU_JOB_KINDS), None)
        if gpu_job is not None:
            # A live RENDER already has this exact engine resident and can lend it between its
            # segments — borrow instead of refusing for the whole multi-hour job (F1). Require an
            # actual live RENDER *and* broker.eligible: a coincidental queued ATTRIBUTE/WARMUP that
            # sorts ahead of the render in `live` must not defeat a valid borrow, and a non-render
            # GPU job must never authorize one.
            render_live = any(j.kind is JobKind.RENDER for j in live)
            if render_live and broker is not None and broker.eligible(meta.engine):
                return Verdict.BORROW
            resident = get_gpu_manager().resident
            resident_note = f" ({resident} is resident)" if resident else ""
            raise ApiError(
                409,
                "gpu_busy",
                f"a {gpu_job.kind.value} job is {gpu_job.state.value}; the single GPU "
                f"cannot also host an audition{resident_note} — wait or cancel it",
                detail=JobOut.from_job(gpu_job).model_dump(mode="json"),
            )
    if meta.engine == "elevenlabs":
        slot_job = next((j for j in live if j.kind in CLOUD_SLOT_JOB_KINDS), None)
        if slot_job is not None:
            # closes the widened eviction race: an audition's ensure_cloud_voice can
            # never race an in-flight render's slot use (sign-off Q6 narrowing)
            raise ApiError(
                409,
                "cloud_busy",
                f"a {slot_job.kind.value} job is {slot_job.state.value}; ElevenLabs "
                "voice slots are in use — wait or cancel it",
                detail=JobOut.from_job(slot_job).model_dump(mode="json"),
            )
    if (
        engine_cls.uses_gpu
        and not registry.is_resident(meta.engine)
        and weights_cached(meta.engine) is False
    ):
        raise ApiError(
            409,
            "engine_cold",
            f"{meta.engine} has never loaded and its weights are not downloaded; a "
            "synchronous audition would pin this request on a multi-GB download — "
            "run the warmup job first",
            detail={"warmup": f"/api/engines/{meta.engine}/warmup"},
        )
    return Verdict.EXCLUSIVE


def borrow_and_synthesize(
    broker: BorrowBroker,
    engine_id: str,
    timeout: float,
    live: list[Job],
    synth: Callable[[Any], Any],
) -> Any:
    """Shared BORROW branch for auditions and mixer previews (F1). Requests the running
    render's resident engine, waits for a grant between its segments, synthesizes on THAT
    instance inside ``gpu.acquire`` (an identity no-op that is also the manager-lock
    backstop), and always signals done so the parked render resumes. A grant timeout (or a
    render tearing down) surfaces as a soft 409 ``gpu_busy_retry`` — never an evict+reload.

    ``synth`` receives the lent engine and returns its ``AudioFile``."""
    ticket = broker.request(engine_id)
    engine = broker.wait_grant(ticket, timeout)
    if engine is None:
        render = next((j for j in live if j.kind is JobKind.RENDER), None)
        raise ApiError(
            409,
            "gpu_busy_retry",
            "a render is holding the GPU between segments; it will lend it momentarily — "
            "retry shortly (the render keeps running)",
            detail=(JobOut.from_job(render).model_dump(mode="json") if render else None),
        )
    try:
        with get_gpu_manager().acquire(engine, f"engine:{engine_id}"):
            return synth(engine)
    finally:
        broker.signal_done(ticket)  # release the parked render even on synthesis failure


@router.post("/voices/{voice_id}/audition", response_model=AuditionOut)
def audition(
    voice_id: str,
    body: AuditionRequest,
    request: Request,
    cfg: SettingsDep,
    registry: RegistryDep,
    store: StoreDep,
    gate: GateDep,
    broker: BrokerDep,
) -> AuditionOut:
    _check_voice_id(voice_id)
    library = VoiceLibrary(cfg.voices_dir)
    meta = _load_or_404(library, voice_id)
    try:
        library.verify_consent(meta)  # hash-bound for clones; cheap for the rest
    except VoiceLibraryError as exc:
        raise ApiError(409, "consent_invalid", str(exc)) from exc

    try:
        engine_cls = get_engine_class(meta.engine)
    except ValueError as exc:  # a legacy/hand-written meta with an unknown engine id
        raise ApiError(422, "invalid", str(exc)) from exc

    with ExitStack() as stack:
        with request.app.state.enqueue_mutex:
            live = store.list_jobs(states=[JobState.QUEUED, JobState.RUNNING])
            verdict = _refuse_conflicts(meta, engine_cls, live, registry, broker)
            if not stack.enter_context(request.app.state.audition_slot.try_hold()):
                raise ApiError(
                    409, "audition_in_flight", "another audition is already synthesizing"
                )
            # only GPU engines claim the heavy-work gate; BORROW deliberately SKIPS it and
            # rides the render's own resident engine instead (the gate stays with the job).
            # A cloud audition must not be refused (least of all as a phantom audition)
            # because an attribute or ffmpeg-stage handler is running.
            if (
                verdict is Verdict.EXCLUSIVE
                and engine_cls.uses_gpu
                and not stack.enter_context(gate.try_hold("audition"))
            ):
                running = next((j for j in live if j.state is JobState.RUNNING), None)
                raise ApiError(
                    409,
                    "gpu_busy",
                    "a job currently holds the GPU; wait for it or cancel it",
                    detail=(JobOut.from_job(running).model_dump(mode="json") if running else None),
                )
        # mutex released; slot (+gate for EXCLUSIVE GPU engines) stays held for the
        # synthesis. A job enqueued from here on waits at most one warm-engine synthesis.
        engine = registry.get(meta.engine)
        engine_voice, settings = render_voice_args(meta)
        normalized = normalize_text(body.text, profile=profile_for(meta.engine))
        cost = engine.cost_estimate(normalized)
        if cost > 0 and meta.engine == "elevenlabs" and not cfg.elevenlabs_api_key:
            # config fault, not an upstream failure — the 503 every other route gives
            raise ApiError(
                503, "not_ready", "ELEVENLABS_API_KEY not set; required for paid auditions"
            )
        if cost > 0 and not body.confirm_paid:
            raise ApiError(
                402,
                "payment_confirmation_required",
                f"auditioning {voice_id!r} is a paid call (~${cost:.4f}); re-send with "
                "confirm_paid=true",
                detail={"estimated_usd": cost},
            )
        try:
            if cost > 0 and meta.engine == "elevenlabs" and meta.kind is VoiceKind.CLONED:
                engine_voice = ensure_cloud_voice(
                    meta, engine.client, library, max_slots=cfg.elevenlabs_max_voice_slots
                )
            synth = lambda eng: eng.synthesize(  # noqa: E731
                normalized, engine_voice, {**settings, "seed": meta.seed}
            )
            if verdict is Verdict.BORROW:
                # ride the running render's resident engine between its segments (F1)
                audio = borrow_and_synthesize(
                    broker, meta.engine, cfg.borrow_grant_timeout_s, live, synth
                )
            else:
                gpu = get_gpu_manager()
                ctx = (
                    gpu.acquire(engine, f"engine:{meta.engine}")
                    if engine.uses_gpu
                    else nullcontext()
                )
                with ctx:
                    audio = synth(engine)
                # NO free_all(): the model stays lazily resident by design — the next
                # audition (or single-voice render) re-acquires as an identity no-op.
        except GpuBusyError as exc:
            # cross-PROCESS contention (a CLI run holds gpu.lock): the same hard gpu_busy
            # as the in-server refusals — NOT gpu_busy_retry, whose client auto-retry
            # assumes a wait of seconds; another process won't yield that fast
            raise ApiError(409, "gpu_busy", str(exc)) from exc
        except (SynthesisError, CloudVoiceError) as exc:
            raise ApiError(502, "upstream", str(exc)) from exc
        out_path = _audition_path(library, voice_id)
        # .part.wav, not .wav.part: libsndfile picks the container from the LAST suffix
        tmp = out_path.with_name("audition.part.wav")
        audio.save(tmp)
        os.replace(tmp, out_path)
        return AuditionOut(
            voice_id=voice_id,
            duration_seconds=round(audio.duration_seconds, 3),
            cost_usd=cost,
            audition_url=f"/api/voices/{voice_id}/audition.wav",
        )


@router.get("/voices/{voice_id}/audition.wav")
def audition_wav(voice_id: str, cfg: SettingsDep) -> FileResponse:
    _check_voice_id(voice_id)
    library = VoiceLibrary(cfg.voices_dir)
    path = _audition_path(library, voice_id)
    if not path.is_file():
        raise ApiError(404, "not_found", f"voice {voice_id!r} has never been auditioned")
    return FileResponse(path, media_type="audio/wav", filename=f"{voice_id}_audition.wav")


# -- cloud slots (read-only by policy while the eviction race stands, sign-off Q6) ---------


@router.get("/cloud-slots", response_model=CloudSlotsOut)
def cloud_slots(cfg: SettingsDep) -> CloudSlotsOut:
    path = cfg.voices_dir / REGISTRY_NAME
    slots: list[CloudSlotOut] = []
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            slots = [
                CloudSlotOut(voice_id=vid, cloud_id=entry["cloud_id"], seq=entry["seq"])
                for vid, entry in data.get("voices", {}).items()
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ApiError(
                500, "corrupt_artifact", f"corrupt cloud voice registry {path}: {exc}"
            ) from exc
    slots.sort(key=lambda s: s.seq, reverse=True)  # MRU-first
    return CloudSlotsOut(max_slots=cfg.elevenlabs_max_voice_slots, count=len(slots), slots=slots)
