"""API DTOs (scoping doc section 4). Existing pipeline pydantic models serialize
verbatim where the doc says so; everything here is a NEW view-model the doc marks NEW.
API keys are never serialized — only ``*_configured`` booleans."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from seiyuu.repository import Job


class HealthOut(BaseModel):
    status: str
    version: str


def _redact_params(params: dict | None) -> dict | None:
    """The stored params may carry a live cost token (the render handler consumes it);
    over HTTP it is always reduced to presence + a signature suffix for support."""
    if params is None or "cost_token" not in params:
        return params
    token = params["cost_token"]
    redacted = dict(params)
    redacted["cost_token"] = (
        {"present": True, "sig_suffix": str(token)[-8:]} if token else {"present": False}
    )
    return redacted


class JobOut(BaseModel):
    """``Job`` plus ``is_terminal`` (a property, so pydantic won't serialize it from the
    model) and ``params`` with any cost token redacted."""

    job_id: str
    book_id: str
    kind: str
    state: str
    progress_text: str
    error: str | None
    cancel_requested: bool
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    is_terminal: bool
    params: dict | None = None

    @classmethod
    def from_job(cls, job: Job) -> "JobOut":
        return cls(
            job_id=job.job_id,
            book_id=job.book_id,
            kind=job.kind.value,
            state=job.state.value,
            progress_text=job.progress_text,
            error=job.error,
            cancel_requested=job.cancel_requested,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            is_terminal=job.is_terminal,
            params=_redact_params(job.params),
        )


class JobsOut(BaseModel):
    jobs: list[JobOut]


# -- job params (scoping doc section 4): the SAME model validates the route request and
# -- re-parses Job.params inside the handler, so the two can never drift.


class WarmupParams(BaseModel):
    engine_id: str


class AttributeParams(BaseModel):
    chapters: list[int] = Field(default_factory=list)  # [] = whole book; subsets merge
    provider: Literal["local", "anthropic"] | None = None  # None -> settings default
    model: str | None = None
    prompt_version: str | None = None
    use_hybrid: bool | None = None  # None -> settings.attribution_hybrid
    confirm_paid: bool = False  # checked at enqueue against the EFFECTIVE paid-ness


class PauseWrite(BaseModel):
    """Explicit-null semantics: None = settings default, 0.0 honored (deliberately
    fixing the CLI's `override or default` falsy-zero bug)."""

    paragraph: float | None = Field(None, ge=0)
    after_heading: float | None = Field(None, ge=0)
    scene_break: float | None = Field(None, ge=0)
    dialogue: float | None = Field(None, ge=0)
    chapter_lead_in: float | None = Field(None, ge=0)
    chapter_lead_out: float | None = Field(None, ge=0)


class LoudnessWrite(BaseModel):
    enabled: bool | None = None  # None -> settings.loudness_enabled
    target_lufs: float | None = None  # None -> settings.loudness_target_lufs; 0.0 honored


class AssembleParams(BaseModel):
    pauses: PauseWrite | None = None
    loudness: LoudnessWrite | None = None


class MasterParams(BaseModel):
    pauses: PauseWrite | None = None
    loudness: LoudnessWrite | None = None
    bitrate: str = "64k"
    target_minutes: float | None = Field(None, gt=0)
    use_cover: bool = True  # use the uploaded cover art if present


class OllamaStatus(BaseModel):
    base_url: str
    reachable: bool | None  # None = not probed (the default poll stays network-free)


class KeyStatus(BaseModel):
    anthropic_configured: bool
    elevenlabs_configured: bool


class ApiLimits(BaseModel):
    render_max_usd: float
    cost_quote_ttl_seconds: int
    elevenlabs_max_voice_slots: int
    attribution_confidence_threshold: float
    full_render_confirm_blocks: int
    max_upload_bytes: int


class SystemStatus(BaseModel):
    gpu_resident: str | None
    active_job: JobOut | None  # durable truth: the store's running row, not the runner snapshot
    queued_jobs: int
    audition_in_flight: bool
    reconciled_at_startup: int
    ffmpeg_available: bool
    ollama: OllamaStatus
    keys: KeyStatus
    limits: ApiLimits
    engines: list[str]
    version: str


class SettingsView(BaseModel):
    """Read-only redacted config. Settings freeze at first ``get_settings()``; live
    writes need restart semantics — out of scope for M6b."""

    books_dir: str
    output_dir: str
    voices_dir: str
    data_dir: str
    tts_engine: str
    kokoro_default_voice: str
    attribution_provider: str
    attribution_model: str
    attribution_prompt_version: str
    attribution_hybrid: bool
    narration_wpm: float
    render_max_usd: float
    cost_quote_ttl_seconds: int
    elevenlabs_model_id: str
    elevenlabs_price_per_1k_chars: float
    elevenlabs_max_voice_slots: int
    anthropic_key_configured: bool
    elevenlabs_key_configured: bool


class EngineInfo(BaseModel):
    engine_id: str
    uses_gpu: bool
    requires_validation: bool
    paid: bool
    supports_cloning: bool
    weights_cached: bool | None  # best-effort HF-cache probe; None for cloud engines
    resident: bool  # identity truth: the registry's instance IS the GPU manager's resident


class EnginesOut(BaseModel):
    engines: list[EngineInfo]


class EngineVoiceOut(BaseModel):
    id: str
    name: str
    language: str | None = None
    gender: str | None = None


class EngineVoicesOut(BaseModel):
    engine_id: str
    voices: list[EngineVoiceOut]
