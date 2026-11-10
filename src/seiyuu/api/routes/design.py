"""Voice design: describe a voice, hear the takes Qwen3-TTS VoiceDesign makes, keep one.

POST /designs              start a design (a ``design`` GPU job makes the takes)
GET  /designs/{id}         the takes so far, scored, with the recommended one
GET  /designs/{id}/takes/{k}.wav
POST /designs/{id}/keep    make a take a library voice (CLONED, engine qwen3tts)

The job's subject is ``design:{id}`` in the book column (the warm-up overload), so it never
reads as a book's job. The flow itself lives in ``services/voice_design.py``.
"""

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse

from seiyuu.api.deps import RunnerDep, SettingsDep, StoreDep
from seiyuu.api.enqueue import enqueue_job
from seiyuu.api.errors import ApiError
from seiyuu.api.schemas import (
    DesignCreate,
    DesignKeep,
    DesignOut,
    DesignStartedOut,
    DesignTakeOut,
    JobOut,
    VoiceOut,
)
from seiyuu.normalize import normalize_text
from seiyuu.repository import JobKind
from seiyuu.services import ServiceError
from seiyuu.services.clone_transcript import default_clone_engine
from seiyuu.services.voice_design import (
    DEFAULT_SAMPLE_TEXT,
    DESIGN_ENGINE,
    DesignRecord,
    design_dir,
    keep_take,
    load_record,
    start_design,
    take_path,
)
from seiyuu.voices import VoiceLibrary

router = APIRouter(tags=["design"])


def _dir_or_404(cfg, design_id: str):
    try:
        directory = design_dir(cfg.data_dir, design_id)
    except ServiceError as exc:
        raise ApiError(404, "not_found", str(exc)) from exc
    if not directory.is_dir():
        raise ApiError(404, "not_found", f"no voice design {design_id!r}")
    return directory


def _design_out(record: DesignRecord) -> DesignOut:
    base = f"/api/designs/{record.design_id}/takes"
    return DesignOut(
        design_id=record.design_id,
        name=record.name,
        description=record.description,
        text=record.text,
        takes_requested=record.takes_requested,
        takes=[
            DesignTakeOut(
                take=t.take,
                seconds=t.seconds,
                wpm=t.wpm,
                in_pace=t.in_pace,
                ok=t.ok,
                score=t.score,
                reason=t.reason,
                audio_url=f"{base}/{t.take}.wav",
                kept_voice_id=record.kept.get(t.take),
            )
            for t in record.takes
        ],
        recommended=record.recommended,
    )


@router.post("/designs", response_model=DesignStartedOut, status_code=202)
def create_design(
    body: DesignCreate,
    request: Request,
    response: Response,
    cfg: SettingsDep,
    store: StoreDep,
    runner: RunnerDep,
) -> DesignStartedOut:
    """Record the description and queue the takes. The line is normalized exactly as a render
    would speak it, so a take that passes the speech check said precisely the text it binds."""
    text = normalize_text(body.sample_text or DEFAULT_SAMPLE_TEXT, profile=DESIGN_ENGINE)
    try:
        record = start_design(
            cfg.data_dir,
            name=body.name,
            description=body.description,
            text=text,
            takes=body.takes,
        )
    except ServiceError as exc:
        raise ApiError(422, "invalid", str(exc)) from exc
    job = enqueue_job(
        store=store,
        runner=runner,
        mutex=request.app.state.enqueue_mutex,
        book_id=f"design:{record.design_id}",  # documented subject-id overload
        kind=JobKind.DESIGN,
        params={"design_id": record.design_id},
    )
    response.headers["Location"] = f"/api/designs/{record.design_id}"
    return DesignStartedOut(design=_design_out(record), job=JobOut.from_job(job))


@router.get("/designs/{design_id}", response_model=DesignOut)
def get_design(design_id: str, cfg: SettingsDep) -> DesignOut:
    return _design_out(load_record(_dir_or_404(cfg, design_id)))


@router.get("/designs/{design_id}/takes/{take}.wav")
def design_take_wav(design_id: str, take: int, cfg: SettingsDep) -> FileResponse:
    path = take_path(_dir_or_404(cfg, design_id), take)
    if not path.is_file():
        raise ApiError(404, "not_found", f"design {design_id!r} has no take {take} yet")
    return FileResponse(path, media_type="audio/wav", filename=f"design_{design_id}_take{take}.wav")


@router.post("/designs/{design_id}/keep", response_model=VoiceOut, status_code=201)
def keep_design_take(
    design_id: str, body: DesignKeep, request: Request, response: Response, cfg: SettingsDep
) -> VoiceOut:
    """Make one take a library voice: a consent-recorded CLONED voice cloned from it, on the
    default cloning engine (Breeze where it is set up, else Qwen3-TTS)."""
    directory = _dir_or_404(cfg, design_id)
    library = VoiceLibrary(cfg.voices_dir)
    with request.app.state.voices_mutex:
        try:
            meta = keep_take(
                directory,
                body.take,
                library=library,
                name=body.name,
                engine=default_clone_engine(cfg),
            )
        except ServiceError as exc:
            code = "conflict" if "already exists" in str(exc) else "invalid"
            raise ApiError(409 if code == "conflict" else 422, code, str(exc)) from exc
    response.headers["Location"] = f"/api/voices/{meta.voice_id}"
    return VoiceOut(**meta.model_dump(), has_audition=False)
