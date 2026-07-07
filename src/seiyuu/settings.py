"""Single settings module: .env + defaults, paths resolved from the repo root."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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
    # Server-side operational state (M6: jobs.db) — global, not per-book artifacts.
    data_dir: Path = REPO_ROOT / "data"

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
    # v5 (F1+F2): per-quote (hybrid) speaker attribution + per-segment emotion, the new base.
    # v6 = v5 + the thought-candidates section (selected when emit_thoughts is on). prompt_version
    # is part of the ChunkCacheKey, so bumping v3->v5 re-attributes every book onto v5/v6 while
    # old v3/v4 cached rows stay valid and never collide.
    attribution_prompt_version: str = "v5"
    # Emit Segment(type=thought) for interior monologue (its own voice at render). Opt-in,
    # default OFF: when off, attribution uses v5 (per-quote + emotion). When on, the run uses
    # the thought-aware v6 prompt (v5 + candidates) + candidate generation; because
    # prompt_version becomes "v6" the cache key differs from v5, so thought-on and thought-off
    # runs coexist without clobbering each other.
    emit_thoughts: bool = False
    # Smaller chunks keep a local model's JSON output well within num_ctx and make it far
    # more likely to honor the schema; overlap_blocks still gives cross-block context.
    attribution_chunk_tokens: int = 800
    attribution_chunk_overlap_blocks: int = 2
    attribution_max_local_retries: int = 2
    # Speaker calls below this confidence are surfaced for review in the characters report.
    attribution_confidence_threshold: float = 0.7
    # Hybrid escalation: when on, chunks that fail local retries re-run through anthropic.
    attribution_hybrid: bool = False

    # Opt-in LLM alias adjudication (fills the AliasResolver seam). Default OFF: the alias
    # post-pass stays deterministic-only and byte-identical, with no LLM call and no cost.
    # Runs only on a full-book attribute (a --chapter subset carries a partial registry) or
    # via the standalone `seiyuu adjudicate` command.
    attribution_adjudicate: bool = False
    # Adjudication provider: "local" (Ollama, free, reuses the warm GPU) or "anthropic"
    # (PAID; gated by the same missing-key ctor check as hybrid attribution).
    adjudication_provider: str = "local"
    # Adjudication model; defaults per-provider (attribution_model for local, anthropic_model
    # for anthropic) when left unset.
    adjudication_model: str | None = None
    adjudication_prompt_version: str = "v1"
    # Merge only when the adjudicator says same_person AND confidence >= this threshold;
    # otherwise the pair stays flagged for review rather than merged.
    adjudication_confidence_threshold: float = 0.85
    # Cap on candidate pairs sent to the LLM per run (bounds cost + prompt size); overflow is
    # flagged, not paid for. G1 (first-name) candidates are kept first.
    adjudication_candidate_cap: int = 40
    # Curated nickname/diminutive candidates (generator G3). Fuzzy/edit-distance matching is
    # intentionally not implemented (highest over-merge risk); this toggles only the table.
    adjudication_use_nicknames: bool = True

    # F3 opt-in LLM respell suggester (advisory enrichment over the deterministic hard-name
    # surfacer). Runs only on an explicit user action. "local" (Ollama, free) reuses the GPU
    # through the resource manager; "anthropic" is PAID and gated by confirm_paid + the key.
    # respell_model defaults per-provider (attribution_model for local, anthropic_model for
    # anthropic) when left unset.
    respell_provider: str = "local"
    respell_model: str | None = None
    respell_prompt_version: str = "v1"

    # F4 opt-in Layer-2 LLM caster (advisory voice-trait preference over the Phase-0 keyword
    # bias). Same explicit-action + paid-gate discipline as the respell suggester; cast_book
    # still enforces distinctness/determinism regardless of the LLM output.
    cast_provider: str = "local"
    cast_model: str | None = None
    cast_prompt_version: str = "v1"

    # TTS defaults (M1).
    tts_engine: str = "kokoro"
    kokoro_default_voice: str = "af_heart"

    # Per-segment emotion (F2). Opt-in, default OFF: when off, render/estimate IGNORE the
    # attribution's segment_emotions and settings stay byte-identical to a no-emotion render
    # (cache stable). When on, the quantized emotion folds into each dialogue segment's
    # settings_hash for engines that support it (Chatterbox/ElevenLabs; Kokoro degrades to
    # neutral). Attribution ALWAYS captures emotion under v5/v6 regardless of this flag.
    apply_emotion: bool = False

    # Text normalization (M3). Output changes auto-invalidate the segment cache via
    # normalized_text_hash; this string is for debuggability only, NOT part of the key.
    # v2 (F3): the per-book pronunciation lexicon respell pass joined apply_profile.
    normalization_version: str = "2"

    # GPU resource management (M3). One heavy model resident at a time on a single GPU.
    gpu_device: str = "cuda"
    whisper_device: str = "cpu"  # faster-whisper stays on CPU (M4) so it never contends
    gpu_unload_poll_timeout: float = 30.0  # wait for Ollama to free VRAM before loading TTS

    # Whisper validation (M4). LLM-style TTS (Chatterbox/Fish) must pass before assembly;
    # deterministic engines (Kokoro) skip it. CPU small/int8 so it never contends with TTS.
    validation_model_size: str = "small"
    validation_compute_type: str = "int8"
    validation_min_ratio: float = 0.85  # folded fuzzy similarity below which a segment fails
    validation_max_retries: int = 2  # re-synth attempts (new seed) before flagging for review

    # Assembly loudness normalization (M4). EBU R128 loudnorm; -18 LUFS suits audiobooks.
    loudness_enabled: bool = True
    loudness_target_lufs: float = -18.0
    loudness_true_peak: float = -1.5
    loudness_range: float = 11.0

    # Duration (M4): narration pace for runtime estimates; atempo clamp for target-duration.
    narration_wpm: float = 150.0
    tempo_min: float = 0.85
    tempo_max: float = 1.3

    # Cloud keys: optional until their providers are explicitly enabled.
    anthropic_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    # Premium attribution model (anthropic provider / hybrid escalation).
    anthropic_model: str = "claude-opus-4-8"

    # Cloud TTS (M5, ElevenLabs). Paid; renders go through the explicit cost gate. The key's
    # absence must not raise until the provider is actually used.
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    # Conservative USD/1k chars for the cost gate. MUST be > 0: paid-ness everywhere derives
    # from cost_estimate(text) > 0, so a zero price would silently disable the whole gate.
    elevenlabs_price_per_1k_chars: float = Field(0.30, gt=0)
    elevenlabs_max_voice_slots: int = 10  # tier-limited; evict LRU seiyuu voices past this

    # Cost gate (M6a). Hard ceiling on ONE render's paid total — no flag or token can
    # authorize past it; raise it here deliberately for a big cloud render. Quotes
    # (signed cost tokens) expire after the TTL; expiry just means re-estimating.
    render_max_usd: float = 25.0
    cost_quote_ttl_seconds: int = 900

    # HTTP API (M6b). Upload cap for EPUBs / reference audio / cover art; exposed to the
    # UI via /api/system limits so the client can refuse early instead of eating a 413.
    max_upload_bytes: int = Field(100 * 1024 * 1024, gt=0)

    # Engine borrowing (F1). Max seconds an audition waits for a running render to lend its
    # resident engine between segments before falling back to a soft gpu_busy_retry; also
    # the safety timeout the parked render waits for the borrower's done signal. ~30s
    # absorbs a Chatterbox segment + its CPU whisper validation without spurious refusals.
    borrow_grant_timeout_s: float = Field(30.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
