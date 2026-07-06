"""API DTOs (scoping doc section 4). Existing pipeline pydantic models serialize
verbatim where the doc says so; everything here is a NEW view-model the doc marks NEW.
API keys are never serialized — only ``*_configured`` booleans."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from seiyuu.attribute.models import AttributionReport
from seiyuu.repository import BookStatus, Job
from seiyuu.services.voices import VoiceReference
from seiyuu.voices import VoiceAssignment, VoiceMeta


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
    use_adjudicate: bool | None = None  # None -> settings.attribution_adjudicate
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


# -- books (scoping doc section 4: Books and ingest / Attribution and review) ------------


class ActiveJobSummary(BaseModel):
    """Deliberately NO progress_text: book payloads must be useless as progress polls —
    GET /api/jobs/{job_id} is the one poll target (scoping doc polling discipline)."""

    job_id: str
    kind: str
    state: str  # queued | running


class BookCard(BaseModel):
    """`BookStatus` verbatim + the live-job summary, one card per library row."""

    book_id: str
    title: str | None
    authors: list[str]
    ingested: bool
    attributed: bool
    assigned: bool
    rendered: bool
    assembled: bool
    mastered: bool
    active_job: ActiveJobSummary | None = None


class BooksOut(BaseModel):
    books: list[BookCard]


class IngestResponse(BaseModel):
    book: BookStatus
    chapters: int
    blocks: int
    skipped_items: list[str]
    dropped_sections: list[str]


class ChapterSummary(BaseModel):
    index: int  # 1-based
    title: str
    blocks: int
    speakable_blocks: int


class FileDownload(BaseModel):
    url: str
    bytes: int


class ChapterDownload(FileDownload):
    index: int


class DownloadsOut(BaseModel):
    m4b: FileDownload | None = None
    chapter_mp3s: list[ChapterDownload] = Field(default_factory=list)


class CoverOut(BaseModel):
    content_type: str
    bytes: int


class BookDetail(BaseModel):
    status: BookStatus
    chapters: list[ChapterSummary] | None  # None until ingested
    runtime_estimate_seconds: float | None
    active_job: ActiveJobSummary | None
    recent_jobs: list[JobOut]  # newest-first, <= 10
    downloads: DownloadsOut
    cover: CoverOut | None


class PaidArtifacts(BaseModel):
    """The PAID cloud work (ElevenLabs/Fish) a book deletion would discard. Per sign-off
    D8 the segment count and voice ids are AUTHORITATIVE (read from the cache's SegmentKey
    sidecars); ``estimated_usd`` is best-effort and may be None — no per-segment cost is
    stored, so it is never reconstructed from a fragile text-join."""

    paid_segment_count: int
    engines: list[str]
    paid_voice_ids: list[str]
    estimated_usd: float | None = None


class BookDeletedOut(BaseModel):
    """Result of a successful ``DELETE /books/{id}``: which on-disk roots were purged, how
    many terminal job rows were reaped, and how many paid segments were discarded (0 unless
    the caller confirmed a paid deletion)."""

    book_id: str
    output_removed: bool
    books_removed: bool
    jobs_rows_deleted: int
    paid_segments_discarded: int


class RuntimeEstimateOut(BaseModel):
    seconds: float
    formatted: str
    wpm_used: float
    chapters: list[int]


class AttributionOut(BaseModel):
    """The EFFECTIVE report (manual-edits overlay applied) — raw attribution.json is
    never served. edit_warnings surface overlay ops that no longer applied."""

    report: AttributionReport
    edit_warnings: list[str]


class SegmentRow(BaseModel):
    block_id: str
    segment_index: int  # 0-based within the block — exactly what ReassignSegment expects
    type: str
    speaker: str | None  # character id; None = narration
    speaker_name: str | None
    text: str
    confidence: float
    has_audio: bool  # any rendered wav for this block in the manifest
    # Listen read-along timing (M6c-5, additive): which manifest segment of the block
    # plays this row (the ?segment= index for the audio route) and its duration. A
    # single-voice render has ONE wav per block, shared by every row of that block;
    # None when unrendered or when re-attribution re-split the block since the render.
    audio_segment: int | None = None
    duration_seconds: float | None = None
    # Render provenance (additive): the voice that actually rendered this row's audio,
    # and the wav's SegmentKey hash — a stable identity the UI shows and uses to
    # cache-bust audio URLs exactly when the content changed.
    voice_id: str | None = None
    audio_key: str | None = None


class SegmentBrowserOut(BaseModel):
    chapter_index: int
    title: str
    segments: list[SegmentRow]
    edit_warnings: list[str]


# -- edits (scoping doc section 4: Edits) -------------------------------------------------
# Request DTOs STRUCTURALLY forbid the anchor fields (expected_name, expected_loser_name,
# expected_winner_name, text_anchor): anchoring is server-authoritative — record_edit
# fills anchors from what is live NOW, so a client can never smuggle a stale or forged
# anchor. extra="forbid" turns any attempt into a 422.


class RenameRequest(BaseModel):
    model_config = {"extra": "forbid"}
    op: Literal["rename"]
    character_id: str
    new_name: str = Field(min_length=1)


class MergeRequest(BaseModel):
    model_config = {"extra": "forbid"}
    op: Literal["merge"]
    loser_id: str
    winner_id: str

    @model_validator(mode="after")
    def _distinct(self) -> "MergeRequest":
        if self.loser_id == self.winner_id:
            raise ValueError("merge needs two different characters")
        return self


class ReassignRequest(BaseModel):
    model_config = {"extra": "forbid"}
    op: Literal["reassign"]
    block_id: str
    segment_index: int = Field(ge=0)
    speaker: str | None  # required-nullable: null means narration, omitting it is a 422


EditRequest = Annotated[RenameRequest | MergeRequest | ReassignRequest, Field(discriminator="op")]


# -- assignment (scoping doc section 4: Assignment) ---------------------------------------


class AssignmentDraftRequest(BaseModel):
    narrator_voice_id: str | None = None  # None -> auto preset narrator
    thought_voice_id: str | None = None  # None -> thoughts use the speaker's own voice
    accent: Literal["a", "b"] = "a"
    stage: Literal["draft", "final"] = "draft"
    overrides: dict[str, str] = Field(default_factory=dict)  # character_id -> voice_id
    # "hash" = legacy per-character isolated blend; "smart" = book-level collision-free cast.
    strategy: Literal["hash", "smart"] = "hash"
    # smart only: OVERWRITE existing {char_id}_auto voices to apply the new cast (re-renders
    # those voices' segments). Without it, smart stays skip-if-exists like the legacy draft.
    recast: bool = False


class AssignmentDraftResponse(BaseModel):
    assignment: VoiceAssignment
    created_voice_ids: list[str]
    edit_warnings: list[str]


class SuggestCastResponse(BaseModel):
    """A smart-cast PREVIEW (nothing written). ``would_create`` are new auto voices; the
    ``would_recast`` voices already exist and only an apply with ``recast=true`` overwrites
    them — surfaced so the UI can warn that applying re-renders those voices."""

    assignment: VoiceAssignment
    would_create_voice_ids: list[str]
    would_recast_voice_ids: list[str]
    edit_warnings: list[str]


class AssignmentWrite(BaseModel):
    """Full-replace write: the COMPLETE casting map, so a PUT never silently resets
    unlisted characters. schema_version/book_id/created_at are server-filled."""

    stage: Literal["draft", "final"]
    narrator_voice_id: str
    assignments: dict[str, str]
    thought_voice_id: str | None = None


# -- money + render (scoping doc sections 4-5) --------------------------------------------


class SingleSpec(BaseModel):
    engine: str | None = None  # None -> settings.tts_engine
    voice: str | None = None  # None -> settings.kokoro_default_voice
    speed: float = Field(1.0, gt=0)
    seed: int = 41172


def _require_single_iff(mode: str, single: SingleSpec | None) -> None:
    if mode == "single" and single is None:
        raise ValueError("mode=single requires the `single` spec")
    if mode == "multivoice" and single is not None:
        raise ValueError("the `single` spec only applies to mode=single")


class QuoteRequest(BaseModel):
    mode: Literal["multivoice", "single"] = "multivoice"
    chapters: list[int] = Field(default_factory=list)
    single: SingleSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> "QuoteRequest":
        _require_single_iff(self.mode, self.single)
        return self


class QuoteResponse(BaseModel):
    token: str  # the opaque cq1. transport — the only form the quote crosses HTTP in
    book_id: str
    chapters: list[int]
    total_usd: float
    paid_segments: int
    fingerprint: str
    assignment_hash: str | None
    issued_at: float
    expires_at: float
    ttl_seconds: int
    max_usd_ceiling: float


class CostEstimateOut(BaseModel):
    total_usd: float
    paid_segments: int
    cached_segments: int
    free_segments: int
    fingerprint: str
    assignment_hash: str | None  # multivoice only
    mode: Literal["multivoice", "single"]
    chapters: list[int]
    edit_warnings: list[str]  # shown BEFORE any money approval — load-bearing


class RenderParams(BaseModel):
    """Route body AND handler params (re-parsed from Job.params — one model, no drift).
    ``cost_token`` is stored plaintext in the job row (user-acked at sign-off) and
    always redacted over HTTP by JobOut."""

    mode: Literal["multivoice", "single"] = "multivoice"
    chapters: list[int] = Field(default_factory=list)
    cost_token: str | None = None
    confirm_full: bool = False
    single: SingleSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> "RenderParams":
        _require_single_iff(self.mode, self.single)
        return self


class RenderChapterOut(BaseModel):
    index: int
    title: str
    segments: int
    duration_seconds: float


class VoiceUseOut(BaseModel):
    engine: str
    engine_model_version: str
    kind: str


class RenderSummaryOut(BaseModel):
    book_id: str
    mode: Literal["multivoice", "single"]  # derived: single manifests carry an engine
    engine: str | None
    engine_model_version: str | None
    voice_id: str | None
    seed: int | None
    chapters: list[RenderChapterOut]
    total_seconds: float
    voices_used: dict[str, VoiceUseOut]
    validation_failures: int
    assignment_present: bool


class ValidationRow(BaseModel):
    chapter_index: int
    block_id: str
    # index among the block's rendered segments (multivoice blocks carry several) —
    # feeds the audio route's ?segment= so the UI can play THIS failure, not just the
    # block's first wav
    segment_index: int
    voice_id: str | None
    ok: bool
    score: float
    expected: str
    transcript: str
    synth_attempts: int


class ValidationReportOut(BaseModel):
    validated_segments: int
    validation_failures: int
    results: list[ValidationRow]  # failures only unless ?all=true


# -- voices (scoping doc section 4: Voices) ------------------------------------------------


class VoiceOut(VoiceMeta):
    """``VoiceMeta`` verbatim plus audition presence. Keys are never in VoiceMeta, so
    nothing needs redaction here."""

    has_audition: bool = False


class UnreadableVoice(BaseModel):
    voice_id: str
    error: str


class VoiceListOut(BaseModel):
    """Tolerant scan: one corrupt meta.json degrades into ``unreadable`` instead of
    bricking the whole Voice Studio screen."""

    voices: list[VoiceOut]
    unreadable: list[UnreadableVoice] = Field(default_factory=list)


class VoiceDetailOut(VoiceOut):
    audition_url: str | None = None


class PresetVoiceCreate(BaseModel):
    """Covers CLI `voice add-preset` AND `voice add-cloud` (a cloud stock voice is a
    preset with engine='elevenlabs' and the remote id as preset_id — no network, no slot)."""

    model_config = {"extra": "forbid"}
    kind: Literal["preset"]
    name: str = Field(min_length=1)
    engine: str = "kokoro"
    preset_id: str = Field(min_length=1)
    seed: int = 41172
    voice_id: str | None = None  # None -> slug + random suffix


class BlendComponentIn(BaseModel):
    preset_id: str
    weight: float = Field(gt=0)


class BlendVoiceCreate(BaseModel):
    model_config = {"extra": "forbid"}
    kind: Literal["blend"]
    name: str = Field(min_length=1)
    components: list[BlendComponentIn] | None = Field(None, min_length=2)  # None -> auto recipe
    gender: str | None = None  # auto-recipe hint
    accent: Literal["a", "b"] = "a"
    seed: int = 41172
    voice_id: str | None = None


VoiceCreate = Annotated[PresetVoiceCreate | BlendVoiceCreate, Field(discriminator="kind")]


class VoiceTagsWrite(BaseModel):
    """PATCH /voices/{id}: replace the tag list (the only mutable organization field)."""

    model_config = {"extra": "forbid"}
    tags: list[str] = Field(max_length=16)


class VoiceReferencesOut(BaseModel):
    voice_id: str
    references: list[VoiceReference]  # empty = deletable


class VoiceDeletedOut(BaseModel):
    deleted: str


AUDITION_DEFAULT_TEXT = (
    'The quick brown fox jumps over the lazy dog. "Well," she said, "how about that?"'
)


class AuditionRequest(BaseModel):
    text: str = Field(AUDITION_DEFAULT_TEXT, min_length=1, max_length=500)
    confirm_paid: bool = False  # paid engines (cost_estimate > 0) require literal true


class AuditionOut(BaseModel):
    voice_id: str
    duration_seconds: float
    cost_usd: float  # 0.0 for local engines
    audition_url: str


class CloudSlotOut(BaseModel):
    voice_id: str
    cloud_id: str
    seq: int


class CloudSlotsOut(BaseModel):
    """Display-only, read without the slot lock (eventually consistent). This surface
    never grows mutation verbs while the eviction race stands (sign-off Q6)."""

    max_slots: int
    count: int
    slots: list[CloudSlotOut]  # MRU-first


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


class AttributionDefaults(BaseModel):
    """What an attribute job runs with when the request doesn't override — surfaced so
    the UI can SHOW which LLM will read the book (and offer the picker)."""

    provider: str  # "local" | "anthropic"
    model: str  # the provider-appropriate default model id
    anthropic_model: str  # what switching to the paid provider would use
    prompt_version: str
    hybrid: bool


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
    attribution: AttributionDefaults
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
    description: str | None = None  # editorial character note (what am I blending?)


class EngineVoicesOut(BaseModel):
    engine_id: str
    voices: list[EngineVoiceOut]
