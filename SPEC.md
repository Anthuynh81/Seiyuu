# Seiyuu — Multi-Voice Audiobook Creator — Project Spec (v0.5)

## Overview
A tool that converts EPUB (and later PDF) books into multi-voice audiobooks. An LLM
attributes dialogue to characters, each character is assigned a voice (preset, blended,
or cloned from a user-uploaded sample), and the final output is a chaptered .m4b.
Web UI for managing books, voices, and renders.

Runs locally as a localhost web app (FastAPI + React), architected so it could be
hosted later. Windows-native dev environment, single NVIDIA consumer GPU.

Python package name: `seiyuu` (NOT `abc` — that shadows the Python stdlib module).

## Core Principles
- **Pipeline stages are independent modules** with file/DB-based inputs and outputs,
  each runnable and testable from the CLI.
- **CLI-first.** Full pipeline works via CLI before any frontend exists.
- **Local-first, cloud-as-finisher — for BOTH audio and intelligence.** Local TTS for
  draft renders and a local LLM for draft/dev attribution; cloud TTS (ElevenLabs) and
  cloud LLM (Claude API) are opt-in quality finishers behind explicit gates.
- **The reference clip is the voice.** For cloned voices, the curated reference WAV is
  the engine-agnostic source of truth; embeddings and cloud voice IDs are derived,
  disposable, engine+version-keyed caches.
- **Cache everything expensive** (attribution, TTS segments, voice conditionals),
  keyed by content hashes + provider/model/prompt versions.
- **Never lose the author's words.** Attribution output must reconstruct the source
  text exactly; generated audio must pass transcription validation.
- **One GPU, one resident model.** TTS engines and the local LLM share the same GPU;
  a resource manager ensures only one heavy model is loaded at a time.

## Provider Lineup

### TTS engines (`TTSEngine` interface)
| Engine | Type | Role | Cloning | License |
|---|---|---|---|---|
| Kokoro-82M | Local, preset | Fast/free drafts, auto voice blends | No | Apache-2.0 |
| Chatterbox / Turbo | Local, zero-shot | Primary local cloning engine | Yes (~7–20s ref) | MIT |
| IndexTTS-2 | Local, zero-shot | Accuracy-critical narration (M7) | Yes | Apache-2.0 |
| ElevenLabs (IVC/PVC) | Cloud | Premium final renders | Yes (IVC API) | Commercial API |
| Fish Audio S2 (hosted) | Cloud | Cheap final renders | Yes | Commercial API |

Interface: `list_voices()`, `synthesize(text, voice, settings) -> AudioFile`,
`cost_estimate(text)`, `model_version`, `native_sample_rate`, optional
`prepare_voice(reference_wav)`. Pipeline code never imports an engine SDK directly.

### Attribution LLM providers (`AttributionLLM` interface)
| Provider | Type | Role | Cost |
|---|---|---|---|
| Local (Ollama, OpenAI-compatible endpoint) | Local, default for dev/drafts | All prompt iteration + draft attribution. Default model: Qwen3.5 9B (`qwen3.5:9b`, Q4_K_M, ~6.6GB, full GPU offload with 32K context on 8GB); Gemma 4 8B as fallback | Free |
| Anthropic (Claude API) | Cloud | Final-quality attribution; escalation target for low-confidence chunks | ~$1–5/novel depending on model |

Interface: `attribute_chunk(chunk_text, registry, schema) -> segments`,
`provider_id`, `model_id`. Requirements:
- **Schema-enforced JSON output** where the backend supports it (Ollama structured
  outputs, Anthropic tool-use schema) so malformed JSON is impossible, not retried.
- The local provider is the OpenAI SDK pointed at `http://localhost:11434/v1`;
  the Anthropic provider uses the Anthropic SDK. Both live ONLY in
  `src/seiyuu/attribute/providers/`.
- Provider + model are config (`attribution.provider`, `attribution.model`);
  ANTHROPIC_API_KEY is optional unless the anthropic provider or hybrid mode is on.
- **Hybrid escalation mode (optional):** run everything on local; chunks whose
  confidence stays below threshold after local retries are re-run through the
  premium provider. Best cost/quality tradeoff once both adapters exist.

## Audio Format Policy
- Canonical intermediate: mono, 24,000 Hz, 16-bit PCM WAV. Every engine adapter
  resamples to canonical on output.
- Loudness-normalize segments to a target LUFS at assembly.
- Final export: AAC .m4b at 44.1kHz (single upsample at the end) + per-chapter MP3s.

## GPU Resource Management
- Single consumer GPU, never assume more than 8GB VRAM.
- A resource manager owns the GPU: at most one heavy model resident at a time —
  a TTS model OR the local attribution LLM, never both.
- Ollama keeps models resident after requests (default ~5 min). The attribution
  stage must request `keep_alive: 0` (or explicitly unload) so the model frees VRAM
  before the render stage loads a TTS engine.
- faster-whisper validation runs on CPU (small/int8) by default so it never
  contends with the resident model; GPU optional via config.
- Pipeline stage ordering exploits this naturally: attribute (LLM resident) →
  render (TTS resident) are sequential stages, not concurrent.

## Voice Model

### Voice library on disk
```
voices/{voice_id}/
  reference.wav                  # curated source of truth (cloned voices only)
  meta.json                      # name, kind, settings, seed, provenance, consent
  conds_chatterbox_{ver}.pt      # derived cache: precomputed conditionals
  elevenlabs_voice_id.txt        # derived cache: cloud voice handle
```

### meta.json (illustrative)
```json
{
  "voice_id": "elena_a1b2",
  "name": "Elena",
  "kind": "cloned",
  "reference_audio": "reference.wav",
  "settings": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
  "seed": 41172,
  "consent_attested": true,
  "source": "user_upload",
  "created_at": "2026-06-09"
}
```

### Voice kinds
- **preset**: `{engine, preset_id}` (e.g., Kokoro `af_bella`).
- **blend**: Kokoro style-vector interpolation `{blend: [(preset, weight), ...]}`;
  auto-assign draft voices by hashing character name → blend weights.
- **cloned**: reference.wav + per-engine settings. ElevenLabs voices created via IVC,
  voice_id cached; slots are tier-limited, so handle voice-not-found by recreating
  from reference.wav.

### Voice creation flow
upload → curation (trim, loudness-normalize, resample, noise warning) →
**consent gate** (rights attestation stored in meta.json) → audition →
save. Derived caches built lazily, keyed by engine + model version, always
regenerable from reference.wav.

### Determinism
- Pinned seed per voice; renders must use it.
- Derived embeddings/conds are cache, never truth.
- SQLite stores metadata and paths only, never blobs.

## Pipeline Stages

### 1. Ingestion
- EPUB v1 (ebooklib + BeautifulSoup); front/back matter stripped heuristically with
  user override. PDF (pymupdf) deferred to M8.
- Scene-break detection (`***`, horizontal rules, blank-line gaps) → explicit
  `scene_break` blocks.
- Output: normalized JSON
  `{book_meta, chapters: [{title, blocks: [{type: paragraph|scene_break|heading,
  id, text}]}]}`. Block ids stable and ordered.

### 2. Speaker Attribution (LLM, provider-agnostic)
- Chunk chapters (~2–4k tokens, overlapping), send through the configured
  `AttributionLLM` provider with the running character registry.
- Output: ordered segments `{type: narration|dialogue|thought, speaker, text,
  confidence, block_id}`.
- **Reconstruction invariant (hard validation, provider-independent):**
  concatenated segment texts per block must reproduce the source block exactly
  (whitespace-normalized). Violations reject the chunk and retry. This check is
  MORE important for local models, which paraphrase more readily.
- **Retry/escalation policy:** retry locally up to N times with adjusted prompting;
  then, if hybrid mode is on, escalate the chunk to the premium provider; else flag
  for manual review.
- **Overlap merge policy:** each block owned by exactly one chunk; duplicate
  segments from overlap regions discarded.
- Character registry: `{id, canonical_name, aliases[], gender, age_hint,
  description, first_appearance}`; aliases merge across chapters. Confidence
  threshold configurable. Registry quality degrades gracefully: low-confidence
  alias merges flagged rather than auto-applied (small models are weaker at
  long-range alias resolution).
- Cached in SQLite by (book, chapter, chunk_hash, provider_id, model_id,
  prompt_version). Switching providers/models naturally invalidates the cache
  without clobbering other providers' results.
- **Prompt development workflow:** tune prompts against the local model first; a
  prompt that works on an 8B model transfers up to Claude trivially, never the
  reverse. Keep prompts versioned in-repo (`prompts/` with prompt_version strings).

### 3. Text Normalization (pre-TTS)
- Numbers, currency, ordinals, abbreviations, roman numerals, em dashes/ellipses,
  unicode cleanup → speakable text. Engine-aware profiles.
- Pure function, deterministic, fixture-tested. Normalized text feeds both
  synthesis and whisper validation comparison.

### 4. Voice Assignment
- Characters + narrator map to voices (preset/blend/cloned); auto-assign from
  registry metadata; user overrides in UI with audio preview. Draft and final
  assignments can differ but hang off the same character record.

### 5. TTS Rendering
- Segment-level rendering grouped by voice (cached Chatterbox conds make voice
  switches cheap). GPU resource manager controls model load/unload.
- Segment cache keyed by (engine, engine_model_version, voice_id, settings_hash,
  seed, normalized_text_hash).
- Validation loop (mandatory for Chatterbox/Fish): synthesize → CPU faster-whisper
  transcribe → fuzzy-match vs normalized text → retry with new seed/settings up to
  N times → surface persistent failures. IndexTTS-2 may skip (config flag).
- Cloud renders show cost estimate and require explicit confirmation.

### 6. Assembly
- Pause logic: scene_break → long, paragraph → medium, dialogue exchange → short.
- Loudness normalization pass; per-chapter MP3s + .m4b (AAC) with chapter markers
  and cover art via ffmpeg metadata file.
- Duration estimates from word count + engine pace; optional target-duration mode
  via ffmpeg atempo, bounded 0.85–1.3x.

## Architecture
- **Backend:** Python 3.11, FastAPI; BackgroundTasks initially, queue-swappable.
- **Local LLM serving:** Ollama (OpenAI-compatible at localhost:11434). Model pulls
  are a documented setup step, not runtime magic.
- **Storage:** SQLite in WAL mode behind a repository layer (Postgres-swappable);
  files on disk behind a storage abstraction (S3-swappable).
- **Frontend:** React + Vite + Tailwind, REST + polling.
- **Auth:** none in v1; routes structured for later middleware auth.
- **Config:** .env + single settings module. ANTHROPIC_API_KEY and
  ELEVENLABS_API_KEY both optional until their providers are enabled.

### Key UI Screens (v1)
1. **Library** — books, status, runtime + cost estimates.
2. **Voice Studio** — voice library; upload→curate→consent→audition; Kokoro blend
   builder; ElevenLabs slot management.
3. **Character Review** — detected characters, sample lines, low-confidence fixes
   (badge which provider attributed each flagged segment), voice assignment with
   preview. Gate before any render.
4. **Render** — engine choice (draft vs final), validation report, cost estimate,
   progress, downloads.

## Milestones
1. **M1 — Plumbing:** ✅ **done.** EPUB → normalized JSON (with scene breaks) →
   single-voice Kokoro render through TTSEngine + canonical audio → chapter MP3s, via CLI.
2. **M2 — Attribution (local-first):** ✅ **done.** `AttributionLLM` interface + Ollama
   provider with schema-enforced JSON + reconstruction invariant + overlap merge +
   registry + caching. Prompt tuning happened here against the local model. CLI report of
   characters and sample lines (`seiyuu attribute` / `seiyuu characters`). Anthropic
   adapter + optional hybrid escalation added at the end.
   - **Key design (deviation worth noting):** *reconstruction-by-construction*. Small local
     models can't reproduce prose verbatim, so the pipeline splits each block into spans
     deterministically (double-quote boundaries, `attribute/spans.py`) and the LLM only
     names ONE speaker per block; segment text is sliced from the SOURCE (quoted span =
     dialogue by that speaker, prose = narration). The hard reconstruction guard remains as
     a safety net but can no longer be violated by a well-behaved model.
   - **Local transport:** Ollama native `/api/chat` (the OpenAI `/v1` shim can't disable a
     reasoning model's thinking or set `num_ctx`); default model `qwen2.5:7b` (fits 8GB).
   - **Known gaps (carry forward):** (a) long-range alias resolution is weak on a 7B
     (un-merged `Darcy`/`Mr. Darcy`, occasional invented name) — flagged-not-merged per
     spec; a dedicated resolution pass or the cloud finisher would help. (b) `thought`
     segments are not emitted by the local span path (type is derived from quotes:
     quoted→dialogue, prose→narration); the `Segment` schema still supports `thought` for a
     future markup-aware or cloud pass.
3. **M3 — Voices:** voice library + Chatterbox cloning (upload→curate→audition CLI),
   conds caching, seeds, Kokoro blends, text normalization stage. First multi-voice
   cloned render. GPU resource manager (LLM↔TTS handoff) lands here.
4. **M4 — Validation + Assembly:** whisper validation loop (CPU), pause logic,
   loudness normalization, .m4b with chapters, duration control.
5. **M5 — Cloud TTS:** ElevenLabs IVC adapter (slot-aware), cost gate,
   draft-vs-final workflow. Fish Audio adapter optional.
6. **M6 — Frontend:** FastAPI API + React UI.
7. **M7 — IndexTTS-2:** second local cloning engine; emotion refs.
8. **M8 — PDF ingestion.**

## Testing
- Project Gutenberg fixture + tiny synthetic EPUB.
- Per-stage snapshot tests. Attribution tests use recorded provider responses (from
  the local model — free to regenerate); no live LLM or TTS calls in the default
  suite. Normalization: pure-function fixture tests.
- GPU/TTS smoke tests behind `@pytest.mark.gpu`; pyproject.toml sets
  `addopts = "-m 'not gpu'"`.
- The reconstruction invariant gets its own test suite with adversarial fixtures
  (paraphrase, dropped sentence, reordered dialogue) since it's the main guard
  against local-model failure modes.

## Open Questions
- Thoughts/internal monologue: character voice, narrator, or softened variant?
  *M2 update:* the `Segment` schema keeps a `thought` type, but the local span pipeline
  derives type from quotes only (quoted→dialogue, prose→narration), so it does not emit
  `thought` today. Revisit with italics/markup-aware splitting or a cloud pass; the
  voice-rendering choice is still deferred to M3 voice assignment.
- Hybrid escalation defaults: confidence threshold, max local retries before
  escalating.
- Which local model wins on attribution quality — *M2 finding:* qwen3.5:9b is a reasoning
  model whose weights exceed usable 8GB VRAM (CPU spill, slow); qwen2.5:7b (non-thinking)
  fits fully and is the working default. Re-run the bake-off (incl. Gemma) if more VRAM is
  available; cache is keyed by model_id so results don't clobber.
- Alias-resolution quality on small models: add a dedicated registry-resolution pass
  (or escalate flagged merges to the cloud) — 7B leaves obvious aliases un-merged.
- Emotion handling: IndexTTS-2 emotion refs (M7), per-segment Chatterbox
  exaggeration, or skip for v1?
- Cloud finisher default: ElevenLabs quality vs Fish Audio price; re-verify pricing
  at M5.
- Single-voice "no attribution" fast path for nonfiction — worth a flag?