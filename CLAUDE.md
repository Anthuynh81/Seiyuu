# Seiyuu — Multi-Voice Audiobook Creator

EPUB → LLM speaker attribution (local Ollama by default, Claude API as premium
finisher) → text normalization → multi-engine TTS (local cloning via Chatterbox,
presets/blends via Kokoro, cloud via ElevenLabs) → chaptered .m4b audiobook.
Full architecture, data models, and milestones live in SPEC.md — read it before
starting any new milestone. Build CLI-first; the frontend is milestone M6.

## Environment
- Windows 11, native (not WSL). Prefer cross-platform Python (pathlib, shutil) over
  shell-isms. Never use bare relative paths — resolve from repo root or
  `Path(__file__)`.
- Python 3.11 in `.venv` (uv-managed). Activate: `.venv\Scripts\activate`.
- Package name is `seiyuu` (`src/seiyuu/`). NEVER name a package or module `abc`,
  `json`, `email`, or any other stdlib name.
- Ollama serves the local attribution LLM at `http://localhost:11434/v1`
  (OpenAI-compatible). Default model: `qwen3.5:9b` (Q4_K_M, ~6.6GB) per settings;
  Gemma 4 8B is the approved fallback. Ollama being
  down is a clear, actionable error, not a crash.
- ffmpeg is on PATH and is the only sanctioned way to touch audio containers/m4b.

## Critical: GPU discipline (single consumer GPU, assume 8GB)
- At most ONE heavy model resident at a time: a TTS model OR the local LLM, never
  both. The GPU resource manager owns load/unload; stages acquire through it.
- Ollama keeps models resident after requests (default ~5 min). Attribution
  requests set `keep_alive: 0` (or explicitly unload) so VRAM is free before the
  render stage loads a TTS engine.
- faster-whisper validation runs on CPU (small/int8) by default; GPU only via
  explicit config.
- torch MUST be the CUDA build (`torch.__version__` ends in `+cuXXX`). The default
  PyPI wheel on Windows is CPU-only and dependency changes can silently downgrade.
  After ANY dependency change, run:
  `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  and confirm `True` before running TTS code. If torch got replaced: force-reinstall
  the SAME pinned version from `https://download.pytorch.org/whl/cuXXX`. Never bump
  the torch version to fix this — chatterbox-tts pins it.

## Architecture Rules
- Pipeline stages are independent modules under `src/seiyuu/`:
  `ingest/`, `attribute/`, `normalize/`, `voices/`, `engines/`, `render/`,
  `assemble/`, `api/`. Each runnable standalone via CLI, tested against fixtures,
  communicating only through documented file/DB formats.
- TTS engine SDKs live ONLY in `src/seiyuu/engines/` behind the `TTSEngine`
  interface. LLM SDKs (openai-pointed-at-Ollama, anthropic) live ONLY in
  `src/seiyuu/attribute/providers/` behind the `AttributionLLM` interface.
  Pipeline code never imports either SDK directly.
- Attribution providers must use schema-enforced JSON output where the backend
  supports it (Ollama structured outputs, Anthropic tool schema). Don't parse
  free-text JSON and retry — make malformed output impossible.
- Audio policy: every adapter outputs canonical mono 24kHz 16-bit WAV; mixed sample
  rates must never reach assembly.
- Voice model truths:
  - `voices/{voice_id}/reference.wav` is the source of truth for cloned voices;
    embeddings, conds (.pt), and cloud voice IDs are disposable caches keyed by
    engine + model version, regenerable from reference.wav.
  - ElevenLabs voice slots are tier-limited: handle voice-not-found by recreating,
    never by erroring out.
  - Characters reference voice records by voice_id, never engine voice IDs.
  - Every voice has a pinned seed in meta.json; renders must use it.
- Attribution invariants (hard validation, provider-independent, never skip):
  - Concatenated segment texts per block must reproduce the source block exactly
    (whitespace-normalized). The LLM may never drop, reorder, or paraphrase text.
    Violations reject the chunk; retry locally up to the configured limit, then
    escalate (hybrid mode) or flag for review. This guard matters MORE for local
    models.
  - Overlapping chunks: each block owned by exactly one chunk; duplicates discarded.
- Prompts are versioned files in `prompts/`; prompt_version is part of the
  attribution cache key. Tune prompts against the local model first.
- Every expensive call (attribution, TTS) goes through the cache layer. Attribution
  cache key: (book, chapter, chunk_hash, provider_id, model_id, prompt_version).
  TTS segment key: (engine, engine_model_version, voice_id, settings_hash, seed,
  normalized_text_hash).
- LLM-style TTS output (Chatterbox, Fish) must pass whisper validation before
  assembly; never silently ship a hallucinated segment.
- PAID API calls (Anthropic, ElevenLabs, Fish) never run in an automatic code path.
  Cloud TTS renders require an explicit cost-confirmation step; Anthropic
  attribution runs only when provider/hybrid config explicitly enables it.
- SQLite in WAL mode; all access through the repository layer. Metadata and paths
  in the DB, never blobs.

## Code Standards
- Type hints everywhere; pydantic models for all cross-stage payloads.
- Text normalization is a pure function: deterministic, no I/O, fixture-tested;
  normalized text feeds both synthesis and validation comparison.
- Config via .env + a single settings module. ANTHROPIC_API_KEY and
  ELEVENLABS_API_KEY are optional until their providers are enabled; their absence
  with provider=local must not raise. Never log or commit keys.
- Errors fail loudly with which book/chapter/block/segment failed.

## Testing
- `uv run pytest` — default suite, CPU-only, fast; pyproject.toml sets
  `addopts = "-m 'not gpu'"`. Must pass before any task is done.
- `uv run pytest -m gpu` — TTS smoke tests; only when engine code changed, and ask
  before multi-minute GPU jobs.
- No live LLM or TTS calls in the default suite — recorded fixtures only (record
  from the local model; it's free to regenerate).
- The reconstruction invariant has its own adversarial fixture suite (paraphrase,
  dropped sentence, reordered dialogue).
- `uv run ruff check` and `uv run ruff format --check` must pass before done.

## Commands
- `uv run python -m seiyuu.cli convert <file.epub>` — full pipeline
- `uv run python -m seiyuu.cli ingest|attribute|render|assemble ...` — single stages
- `ollama serve` / `ollama pull <model>` — local LLM management (setup docs)
- `uv run uvicorn seiyuu.api.main:app --reload` — backend (M6+)
- `npm run dev` (in frontend/) — UI dev server (M6+)

## Workflow Preferences
- Work milestone by milestone per SPEC.md; don't start later milestones early.
- Propose a short plan before multi-file changes; wait for approval on anything
  touching the voice model, normalized JSON schema, attribution segment schema, or
  cache key formats (migrations are painful).
- Small, frequent commits, one logical change each.
- When adding a dependency: install it, then immediately re-run the torch CUDA
  check.
- If a long render/test or model pull is needed, say so and ask first.

## Gotchas Already Hit
- chatterbox-tts pinning torch reinstalled the CPU wheel and broke CUDA
  ("Torch not compiled with CUDA enabled"). Fix: force-reinstall the same torch
  version from the CUDA index. See GPU discipline section.
- "[Errno 2] No such file" from relative paths: scripts ran from a different cwd
  than the file's folder. Resolve paths from `Path(__file__)` or repo root. Watch
  for Explorer hiding extensions (Test.wav.wav).
- Windows long paths: `git config core.longpaths true` is set; keep generated paths
  short anyway.
- First Chatterbox/Kokoro run downloads multi-GB weights to the HF cache; first
  `ollama pull` is also multi-GB. Slow first runs are downloads, not hangs.