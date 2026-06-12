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
    attribution_provider: str = "local"
    attribution_model: str = "qwen3.5:9b"

    # TTS defaults (M1).
    tts_engine: str = "kokoro"
    kokoro_default_voice: str = "af_heart"

    # Cloud keys: optional until their providers are explicitly enabled.
    anthropic_api_key: str | None = None
    elevenlabs_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
