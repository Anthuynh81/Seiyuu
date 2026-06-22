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
    # Context window for the native transport; the chunk's prompt + JSON output must fit.
    # qwen2.5:7b (~4.7GB) fits fully on an 8GB GPU at 8192; a 9B model would spill to CPU
    # (~10x slower) — drop num_ctx or use a smaller model if you swap to one.
    ollama_num_ctx: int = 8192
    # Keep the model resident between chunks (a book is many calls seconds apart). The
    # explicit free-before-render unload is the M3 GPU resource manager's job; 0 here would
    # reload the model every chunk.
    ollama_keep_alive: str = "5m"
    attribution_provider: str = "local"
    # qwen2.5:7b: non-thinking, fits an 8GB GPU fully, reliable at the per-block speaker
    # task. qwen3.5:9b is higher quality but too large to stay on-GPU here (slow).
    attribution_model: str = "qwen2.5:7b"
    attribution_prompt_version: str = "v3"
    # Smaller chunks keep a local model's JSON output well within num_ctx and make it far
    # more likely to honor the schema; overlap_blocks still gives cross-block context.
    attribution_chunk_tokens: int = 800
    attribution_chunk_overlap_blocks: int = 2
    attribution_max_local_retries: int = 2
    # Speaker calls below this confidence are surfaced for review in the characters report.
    attribution_confidence_threshold: float = 0.7
    # Hybrid escalation: when on, chunks that fail local retries re-run through anthropic.
    attribution_hybrid: bool = False

    # TTS defaults (M1).
    tts_engine: str = "kokoro"
    kokoro_default_voice: str = "af_heart"

    # Text normalization (M3). Output changes auto-invalidate the segment cache via
    # normalized_text_hash; this string is for debuggability only, NOT part of the key.
    normalization_version: str = "1"

    # Cloud keys: optional until their providers are explicitly enabled.
    anthropic_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    # Premium attribution model (anthropic provider / hybrid escalation).
    anthropic_model: str = "claude-opus-4-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
