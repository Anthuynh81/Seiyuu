"""Single settings module: .env + defaults, paths resolved from the repo root."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# settings.py lives at src/seiyuu/settings.py; the project is always an
# editable install, so two parents up is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Directories — always absolute, never cwd-relative.
    books_dir: Path = REPO_ROOT / "books"
    output_dir: Path = REPO_ROOT / "output"
    voices_dir: Path = REPO_ROOT / "voices"
    prompts_dir: Path = REPO_ROOT / "prompts"

    # Local attribution LLM (used from M2).
    ollama_base_url: str = "http://localhost:11434/v1"
    # Ollama transport: "native" (/api/chat — needed to disable thinking + set num_ctx,
    # required for reasoning models like Qwen3) or "openai" (the /v1 compat shim).
    ollama_transport: str = "native"
    # Context window for the native transport; reasoning-free attribution of a ~3k-token
    # chunk needs room for prompt + JSON output. Bigger = more VRAM/slower.
    ollama_num_ctx: int = 8192
    attribution_provider: str = "local"
    attribution_model: str = "qwen3.5:9b"
    attribution_prompt_version: str = "v1"
    attribution_chunk_tokens: int = 3000
    attribution_chunk_overlap_blocks: int = 2
    attribution_max_local_retries: int = 2
    # Speaker calls below this confidence are surfaced for review in the characters report.
    attribution_confidence_threshold: float = 0.7
    # Hybrid escalation: when on, chunks that fail local retries re-run through anthropic.
    attribution_hybrid: bool = False

    # TTS defaults (M1).
    tts_engine: str = "kokoro"
    kokoro_default_voice: str = "af_heart"

    # Cloud keys: optional until their providers are explicitly enabled.
    anthropic_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    # Premium attribution model (anthropic provider / hybrid escalation).
    anthropic_model: str = "claude-opus-4-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
