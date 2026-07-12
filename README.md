# Seiyuu — Multi-Voice Audiobook Creator

Turn an EPUB into a multi-voice `.m4b` audiobook: a local LLM attributes each line of
dialogue to a character, every character is cast with a voice (a preset, a blend, or one
cloned from a short sample), and the book is rendered, validated, and assembled into a
chaptered audiobook. Runs locally as a `localhost` web app, with a full CLI underneath.

> **Local-first, cloud-as-finisher.** Draft everything for free on a local LLM and local
> TTS; opt in to cloud intelligence (Claude) and cloud voices (ElevenLabs) as quality
> finishers, behind explicit cost gates. Designed for a single consumer NVIDIA GPU
> (assume 8 GB).

---

## The pipeline

```
EPUB ─▶ ingest ─▶ attribute ─▶ normalize ─▶ cast ─▶ render ─▶ validate ─▶ assemble ─▶ .m4b
        (parse)   (who speaks)  (speakable)  (voices) (TTS)    (whisper)   (+ chapters)
```

Every stage is an independent module under `src/seiyuu/`, runnable and testable on its own
and communicating only through documented file/DB formats. The whole thing works from the
CLI before any UI is involved.

1. **Ingest** — EPUB → normalized JSON (`chapters` → `blocks`), front/back matter stripped,
   scene breaks detected. Block ids are stable and ordered.
2. **Attribute** — chunk each chapter and run it through the configured LLM to label
   segments as `narration` / `dialogue` / `thought` with a speaker and confidence. A hard
   **reconstruction invariant** guarantees the concatenated segments reproduce the source
   text exactly — the model can never drop, reorder, or paraphrase the author's words.
3. **Normalize** — a pure, deterministic function turns numbers, currency, ordinals,
   abbreviations, and unicode into speakable text (engine-aware profiles).
4. **Cast** — map the narrator and each character to a voice; auto-assign from registry
   metadata, override in the UI with an audio preview.
5. **Render** — synthesize per segment through the `TTSEngine` interface, grouped by voice.
   Every segment goes through a content-addressed cache.
6. **Validate** — LLM-style engines (Chatterbox/Fish) are transcribed with faster-whisper
   and fuzzy-matched against the normalized text; a hallucinated or drifted segment is
   retried, never silently shipped.
7. **Assemble / master** — loudness-normalized per-chapter MP3s and a chaptered `.m4b`
   (AAC) with chapter markers and cover art, via ffmpeg.

## Highlights

- **Reconstruction-by-construction attribution** — the pipeline splits each block into
  spans deterministically (on quote boundaries) and the LLM only *names one speaker per
  block*; segment text is sliced from the source. That's what makes small local models
  usable, and the reconstruction guard can't be violated by a well-behaved splitter.
- **One GPU, one resident model** — a resource manager guarantees only a single heavy
  model (a TTS engine *or* the local LLM) is loaded at a time, so the 8 GB budget is never
  blown. Whisper validation runs on CPU so it never contends.
- **The reference clip is the voice** — for cloned voices, `voices/{id}/reference.wav` is
  the engine-agnostic source of truth; embeddings, conditionals, and cloud voice ids are
  disposable caches, always regenerable. Cloned voices require a stored consent attestation.
- **Nothing expensive runs twice** — attribution and TTS segments are cached by content
  hashes plus provider/model/prompt versions, so switching a model never clobbers another's
  results.
- **No surprise bills** — paid TTS and paid attribution never run on an automatic path.
  Cloud renders require a signed, single-use cost quote confirmed against a fresh estimate,
  bounded by a configurable ceiling.

## Provider lineup

**TTS engines** (behind the `TTSEngine` interface — pipeline code never imports an SDK):

| Engine | Where | Role | Cloning |
|---|---|---|---|
| **Kokoro-82M** | local | fast/free drafts, preset & blended voices | — |
| **Chatterbox** | local | primary zero-shot cloning (~7–20 s reference) | ✅ |
| **IndexTTS-2** | local | slow high-quality finisher; native emotion control | ✅ |
| **ElevenLabs** | cloud | premium final renders | ✅ (IVC) |
| Fish Audio | cloud | cheaper cloud renders (planned) | ✅ |

IndexTTS-2 runs **out-of-process** in its own CUDA env (it pins a torch/transformers combo that
conflicts with Chatterbox's), driven by a stdlib adapter over JSON stdio — see
[IndexTTS-2 setup](#indextts-2-optional-second-local-engine) below. It is opt-in and off the
default path.

**Attribution LLMs** (behind the `AttributionLLM` interface):

| Provider | Where | Role |
|---|---|---|
| **Ollama** (local) | local | default for all drafts & prompt iteration — `qwen2.5:7b` fits 8 GB |
| **Anthropic** (Claude) | cloud | final-quality attribution / low-confidence escalation (opt-in) |

Both attribution backends use **schema-enforced JSON output** (Ollama structured outputs /
Anthropic tool schema), so malformed output is impossible rather than parsed-and-retried.

## Requirements

- **Windows 11**, native (the project is developed Windows-first; paths are cross-platform).
- **Python 3.11**, managed with [`uv`](https://docs.astral.sh/uv/).
- **A CUDA build of PyTorch** — Chatterbox pins torch, and the default PyPI wheel on
  Windows is CPU-only. After any dependency change, confirm CUDA is live (see below).
- **[Ollama](https://ollama.com/)** for local attribution (serves at `http://localhost:11434`).
- **ffmpeg** on `PATH` — the only sanctioned way to touch audio containers / `.m4b`.
- **Node.js** for the web UI (Vite + React).

## Setup

```bash
# 1. Python environment (uv creates .venv and installs from uv.lock)
uv sync
.venv\Scripts\activate

# 2. Confirm PyTorch sees CUDA (must print a version ending in +cuXXX and True)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3. Local attribution model
ollama serve
ollama pull qwen2.5:7b

# 4. (optional) cloud finishers — only needed when you enable them
#    put ANTHROPIC_API_KEY / ELEVENLABS_API_KEY in .env

# 5. Web UI deps
cd frontend && npm install
```

> First runs are slow because they are **downloads**, not hangs: Chatterbox/Kokoro pull
> multi-GB weights into the Hugging Face cache, and `ollama pull` is multi-GB too.

If a dependency change silently reinstalls the CPU torch wheel (the classic "Torch not
compiled with CUDA enabled" failure), force-reinstall the **same** pinned version from the
CUDA index — never bump the version, Chatterbox pins it.

### IndexTTS-2 (optional second local engine)

IndexTTS-2 is a slow, high-quality local cloning engine with **native emotion control** — a
finisher for accuracy-critical narration while Kokoro/Chatterbox stay the fast draft path. It
hard-pins **torch 2.8 / cu128 + transformers 4.52**, which conflict irreconcilably with this
project's torch 2.6 + transformers 5, so it **cannot** be installed into `.venv`. Instead it runs
as a subprocess **worker in its own CUDA environment**; this project only ships a stdlib adapter
that drives it over JSON stdio. Setup is opt-in:

```bash
# 1. Create a SEPARATE env for IndexTTS-2 (its own torch 2.8/cu128 + transformers 4.52).
#    Follow the upstream install (git clone + `uv sync`); on Windows omit --all-extras
#    (DeepSpeed fails to build) and leave the BigVGAN CUDA kernel off.

# 2. Download the IndexTTS-2 checkpoints (multi-GB) into a directory.

# 3. Point this project at that env + weights, in .env:
#    INDEXTTS2_WORKER_PYTHON=C:\path\to\indextts2-env\Scripts\python.exe
#    INDEXTTS2_CHECKPOINTS_DIR=C:\path\to\indextts2\checkpoints
```

On an **8 GB card the peak sits right at the ceiling**, so two host settings are effectively
required:

- **NVIDIA Control Panel → Manage 3D settings → Program Settings → (your IndexTTS-2 `python.exe`)
  → "CUDA – Sysmem Fallback Policy" = "Prefer No Sysmem Fallback".** Without this, Windows spills
  to system RAM over PCIe and RTF collapses from ~4× to ~40×. With it, IndexTTS-2 fits the 8 GB
  3070 at ~4× realtime. Close other GPU apps during a render (no gaming).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fixes allocator fragmentation at the ceiling
  — the adapter sets it on the worker automatically, so no action is needed.

The adapter is **OOM-safe**: a CUDA out-of-memory (or a hung/dead worker) kills and restarts the
worker and retries the segment before failing loudly. `unload()` terminates the worker, so VRAM is
reclaimed by process death and the single-GPU discipline holds. Once configured, clone a voice with
`--engine indextts2` (or the Voice Studio engine picker) and render as usual; output rides the same
whisper validation and per-segment cost/cache paths as every other engine.

## Usage

### CLI (the whole pipeline)

```bash
# End to end (single-voice draft): EPUB → .m4b
uv run python -m seiyuu.cli convert book.epub

# Or run stages individually
uv run python -m seiyuu.cli ingest    book.epub
uv run python -m seiyuu.cli attribute <book_id>
uv run python -m seiyuu.cli characters <book_id>          # review detected cast
uv run python -m seiyuu.cli voice add-preset|blend|list|audition ...
uv run python -m seiyuu.cli assign    <book_id>           # auto-cast characters → voices
uv run python -m seiyuu.cli render    <book_id> --multivoice
uv run python -m seiyuu.cli validate  <book_id>
uv run python -m seiyuu.cli assemble  <book_id>
uv run python -m seiyuu.cli master    <book_id> --cover cover.jpg

# Money is always gated: estimate, then render with confirmation
uv run python -m seiyuu.cli estimate-cost <book_id>
uv run python -m seiyuu.cli render <book_id> --confirm-cost
```

### Web app

```bash
# Backend — exactly ONE uvicorn worker (the job runner & GPU manager are process-local)
uv run uvicorn seiyuu.api.main:app --reload

# Frontend (Vite proxies /api to the backend)
cd frontend && npm run dev
```

The API only answers requests whose `Host` header is on an allowlist (DNS-rebinding
guard — otherwise any website could resolve itself to `127.0.0.1` and drive the paid
endpoints). To reach it from another device on your LAN (e.g. the Listen screen on a
tablet), add that machine's address in `.env`:
`API_ALLOWED_HOSTS=localhost,127.0.0.1,testserver,192.168.1.20`

The UI has five screens: **Library** (books, status, estimates), **Listen** (real-audio
read-along with a word-karaoke highlight and chapter auto-advance), **Character Review**
(reassign / rename / merge the cast, spoiler-safe), **Voice Studio** (upload → curate →
consent → audition; Kokoro blend builder; cloud-slot view), and **Render & Jobs** (the
guided cost flow, progress, downloads, and validation diffs).

## Project layout

```
src/seiyuu/
  ingest/        EPUB → normalized JSON
  attribute/     LLM speaker attribution (providers/ holds the LLM SDKs)
  normalize/     pure, deterministic text normalization
  voices/        voice library, blends, cloud slot manager
  engines/       TTSEngine adapters (Kokoro, Chatterbox, IndexTTS-2, ElevenLabs) — the only SDKs
  render/        segment rendering + content-addressed cache + cost gate
  validate/      faster-whisper transcription validation
  assemble/      pauses, loudness, chapter MP3s, .m4b master
  gpu/           the single-GPU resource manager
  jobs/          single-flight durable job runner
  repository/    SQLite (WAL) + atomic file writes behind a seam
  api/           FastAPI app (routes/, uniform error envelope)
  services/      stage orchestration shared by CLI and API
  cli.py         the Click CLI
frontend/        React + Vite + TypeScript + TanStack Query
prompts/         versioned attribution prompts (prompt_version is part of the cache key)
SPEC.md          full architecture, data models, and milestones
CLAUDE.md        working agreements and hard rules for contributors
```

## Development

```bash
uv run pytest                 # default suite: CPU-only, fast, no live LLM/TTS (fixtures)
uv run pytest -m gpu          # TTS smoke tests (only when engine code changed)
uv run ruff check             # lint
uv run ruff format --check    # formatting

cd frontend && npm test       # vitest
cd frontend && npm run build  # tsc + vite build
```

The default test suite makes **no live LLM or TTS calls** — attribution tests replay
recorded fixtures (free to regenerate from the local model), and GPU work is behind
`@pytest.mark.gpu` (deselected by default).

## Status

Milestones **M1–M6 are complete**: the full CLI pipeline (ingest → attribute → normalize →
cast → render → validate → assemble → master), local + cloud attribution, the voice library
with cloning and blends, cloud TTS with the cost gate, a durable job/persistence layer, the
FastAPI backend, and the React web UI. **M7** (IndexTTS-2, a second local cloning engine with
native emotion control) is in progress — the out-of-process worker + adapter and its clone/render
surface are landed; see [IndexTTS-2 setup](#indextts-2-optional-second-local-engine). **M8** (PDF
ingestion) is planned. See [SPEC.md](SPEC.md) for the full design and the per-milestone design
notes.
