"""GET /health, /system, /settings — liveness, operational truth, redacted config."""

import shutil
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import urlopen

from fastapi import APIRouter

from seiyuu import __version__
from seiyuu.api.deps import GateDep, ReconciledDep, RegistryDep, SettingsDep, StoreDep
from seiyuu.api.registry import catalog_ids
from seiyuu.api.schemas import (
    ApiLimits,
    HealthOut,
    JobOut,
    KeyStatus,
    OllamaStatus,
    SettingsView,
    SystemStatus,
)
from seiyuu.gpu import get_gpu_manager
from seiyuu.render.gate import FULL_RENDER_CONFIRM_BLOCKS
from seiyuu.repository import JobState

router = APIRouter(tags=["system"])


def _probe_ollama(base_url: str) -> bool:
    """GET the server root (not the /v1 path) with a short timeout. Any HTTP response
    means reachable; only a connection-level failure means down."""
    parts = urlsplit(base_url)
    root = f"{parts.scheme}://{parts.netloc}/"
    try:
        with urlopen(root, timeout=1.0):  # noqa: S310 — scheme/host come from settings
            return True
    except HTTPError:
        return True
    except OSError:
        return False


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok", version=__version__)


@router.get("/system", response_model=SystemStatus)
def system_status(
    cfg: SettingsDep,
    store: StoreDep,
    registry: RegistryDep,
    gate: GateDep,
    reconciled: ReconciledDep,
    probe: bool = False,
) -> SystemStatus:
    running = store.list_jobs(states=[JobState.RUNNING], limit=1)
    return SystemStatus(
        gpu_resident=get_gpu_manager().resident,
        active_job=JobOut.from_job(running[0]) if running else None,
        queued_jobs=len(store.list_jobs(states=[JobState.QUEUED])),
        audition_in_flight=gate.audition_in_flight,
        reconciled_at_startup=reconciled,
        ffmpeg_available=shutil.which("ffmpeg") is not None,
        ollama=OllamaStatus(
            base_url=cfg.ollama_base_url,
            reachable=_probe_ollama(cfg.ollama_base_url) if probe else None,
        ),
        keys=KeyStatus(
            anthropic_configured=bool(cfg.anthropic_api_key),
            elevenlabs_configured=bool(cfg.elevenlabs_api_key),
        ),
        limits=ApiLimits(
            render_max_usd=cfg.render_max_usd,
            cost_quote_ttl_seconds=cfg.cost_quote_ttl_seconds,
            elevenlabs_max_voice_slots=cfg.elevenlabs_max_voice_slots,
            attribution_confidence_threshold=cfg.attribution_confidence_threshold,
            full_render_confirm_blocks=FULL_RENDER_CONFIRM_BLOCKS,
            max_upload_bytes=cfg.max_upload_bytes,
        ),
        engines=catalog_ids(),
        version=__version__,
    )


@router.get("/settings", response_model=SettingsView)
def settings_view(cfg: SettingsDep) -> SettingsView:
    return SettingsView(
        books_dir=str(cfg.books_dir),
        output_dir=str(cfg.output_dir),
        voices_dir=str(cfg.voices_dir),
        data_dir=str(cfg.data_dir),
        tts_engine=cfg.tts_engine,
        kokoro_default_voice=cfg.kokoro_default_voice,
        attribution_provider=cfg.attribution_provider,
        attribution_model=cfg.attribution_model,
        attribution_prompt_version=cfg.attribution_prompt_version,
        attribution_hybrid=cfg.attribution_hybrid,
        narration_wpm=cfg.narration_wpm,
        render_max_usd=cfg.render_max_usd,
        cost_quote_ttl_seconds=cfg.cost_quote_ttl_seconds,
        elevenlabs_model_id=cfg.elevenlabs_model_id,
        elevenlabs_price_per_1k_chars=cfg.elevenlabs_price_per_1k_chars,
        elevenlabs_max_voice_slots=cfg.elevenlabs_max_voice_slots,
        anthropic_key_configured=bool(cfg.anthropic_api_key),
        elevenlabs_key_configured=bool(cfg.elevenlabs_api_key),
    )
