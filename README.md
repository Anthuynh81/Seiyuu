# Seiyuu — Multi-Voice Audiobook Creator

Turn an EPUB or PDF into a multi-voice `.m4b` audiobook: a local LLM attributes each line of
dialogue to a character, every character is cast with a voice (a preset, a blend, or one
cloned from a short sample), and the book is rendered, validated, and assembled into a
chaptered audiobook. Runs locally as a `localhost` web app, with a full CLI underneath.

> **Local-first, cloud-as-finisher.** Draft everything for free on a local LLM and local
> TTS; opt in to cloud intelligence (Claude) and cloud voices (ElevenLabs) as quality
> finishers, behind explicit cost gates. Designed for a single consumer NVIDIA GPU
> (target: 12 GB RTX 3080 Ti).

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
  model (a TTS engine *or* the local LLM) is loaded at a time, so the VRAM budget is never
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
| **Qwen3-TTS** | local | closest cloning; designs voices from a description | ✅ |
| **Breeze TTS 2** | local | fast cloning; can take a delivery instruction per line (non-commercial weights) | ✅ |
| **ElevenLabs** | cloud | premium final renders | ✅ (IVC) |
| Fish Audio | cloud | cheaper cloud renders (planned) | ✅ |

IndexTTS-2 runs **out-of-process** in its own CUDA env (it pins a torch/transformers combo that
conflicts with Chatterbox's), driven by a stdlib adapter over JSON stdio — see
[IndexTTS-2 setup](#indextts-2-optional-second-local-engine) below. It is opt-in and off the
default path.

**Attribution LLMs** (behind the `AttributionLLM` interface):

| Provider | Where | Role |
|---|---|---|
| **Ollama** (local) | local | default — `gemma4:12b-it-qat` (12 GB card); `qwen2.5:7b` for 8 GB |
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
ollama pull gemma4:12b-it-qat   # or qwen2.5:7b on an 8 GB card (set ATTRIBUTION_MODEL)

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

#### Qwen3-TTS (the default cloning engine)

Qwen3-TTS 1.7B (Apache-2.0) clones a voice from a short clip and can also **design** one from a
written description. It is the default engine for new clones (the Clone dialog and
`seiyuu voice clone`); pick `--engine chatterbox` / `indextts2` / `elevenlabs` for the others. Like IndexTTS-2 it runs as a worker in its **own** env (it pins torch
2.8/cu128 and transformers 5.15.1), through [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts):

```bash
mkdir qwen3-tts && cd qwen3-tts
uv venv --python 3.11
# constraints.txt pins torch==2.8.0+cu128, torchaudio==2.8.0+cu128 and transformers==5.15.1
# (transformers 5.22 breaks the speech tokenizer: "property upsample ... has no setter")
uv pip install --python .venv\Scripts\python.exe -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 faster-qwen3-tts
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

Then in `.env`: `QWEN3TTS_WORKER_PYTHON=...\qwen3-tts\.venv\Scripts\python.exe` and
`QWEN3TTS_MODELS_DIR=...\qwen3-tts\models`. `QWEN3TTS_BACKEND=stock` falls back to the reference
`qwen_tts` package (no CUDA graphs) if the fast path misbehaves. A voice is cloned in Qwen's ICL
mode when its `meta.json` binds a transcript (`settings.qwen3tts.ref_text_digest`, a digest of
`reference.wav` plus the `reference.txt` beside it, so the transcript is part of the segment cache
key); otherwise from the speaker embedding alone. A Qwen clone of a clip up to 30 s writes both:
the transcript is yours if you typed one (`--transcript`, or the dialog's "what the clip says"),
otherwise whisper's (CPU). Longer clips, silence, or a failed transcription clone from the sound
alone, because a long reference shares Qwen's generation budget and would cut long lines short.
`scripts/design_voices.py` writes both too. A `reference.txt` dropped in by hand is ignored, and
one that no longer matches its digest fails the render loudly rather than mixing clone modes in
one book.

```bash
uv run seiyuu voice clone "Elena" elena.wav --consent              # transcribed for you
uv run seiyuu voice clone "Elena" elena.wav --consent --transcript "Call me Elena."
```

**Describe a voice.** On the Voices screen, "describe a voice" designs one from a written
description: Qwen3-TTS VoiceDesign speaks a line in a new voice a few times (a `design` job on the
GPU: about a minute to load the model, then ~5 s per take), each take is scored for clarity and pace, and the one you keep becomes a
normal Qwen clone in your library. From the CLI:

```bash
uv run seiyuu voice design "Old janitor" --description "a gruff man in his sixties, slow and tired"
uv run seiyuu voice design "Old janitor" --description "..." --keep none   # just list the takes
```

For a whole cast at once, the batch script designs from a JSON file (nothing enters your voice
library; the staged clips land under `--out`):

```bash
uv run python scripts/design_voices.py cast.json --out output/designed/<book>
```

#### Breeze TTS 2 (cloning with voice direction)

[Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) (3B) clones a voice from a clip **and
its exact transcript**, and can take a natural-language delivery instruction per line ("calm, dry,
a little lower") on top of the clone. It also performs parenthesized sounds such as `(sigh)` or
`(laugh)` instead of reading them (whisper validation does not know that yet, and flags such a
line as missing a word). The code is Apache-2.0, but the **weights and any audio you render with
them locally are under the BreezeBlue Research and Non-Commercial License** (research and
non-commercial use only). It is a source checkout rather than a package, with its own pins (torch
2.9.1/cu128, transformers 4.57.3), so it too runs as a worker in its **own** env:

```bash
git clone https://github.com/breezeblue-ai/breeze-tts && cd breeze-tts
git checkout daa6cb2            # the revision the adapter was written against
uv venv --python 3.11
# constraints.txt (create it) pins the CUDA wheels: torch==2.9.1+cu128 and torchaudio==2.9.1+cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe "triton-windows<3.6"   # Windows only: the fast path compiles kernels
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
hf download BreezeBlue/breeze-tts-2 --local-dir checkpoints/breeze-tts-2   # 7.7 GB
```

Then in `.env`: `BREEZE_WORKER_PYTHON=...\breeze-tts\.venv\Scripts\python.exe`,
`BREEZE_REPO_DIR=...\breeze-tts` and `BREEZE_CHECKPOINT_DIR=...\breeze-tts\checkpoints\breeze-tts-2`.

- **Speed and memory** (RTX 3080 Ti): the default `BREEZE_FAST=decode` compiles the depth decoder,
  codec and backbone decode step: about RTF 0.45 for a plain clone and 0.54 with a delivery
  instruction, 8.8 GB peak. The first load compiles for 16-50 s; later starts reuse torch's
  compile cache. `BREEZE_FAST=off` runs plain eager code (about 4x slower, 8 GB), the fallback if
  compilation misbehaves. The two are not bit-exact, so each keeps its own segment cache.
- **Windows**: the worker sets `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` itself. Without it the
  first compiled kernel fails with `OverflowError: Python int too large to convert to C long`.
- **Transcript**: Breeze has no clone mode without one. A voice renders on Breeze only when its
  `meta.json` binds the transcript (`ref_text_digest`, the same digest of `reference.wav` plus
  `reference.txt` that Qwen clones use, so it is part of the segment cache key); a voice that
  binds none, or whose `reference.txt` was edited afterwards, fails the render loudly. A designed
  voice binds one when the take you kept passed the speech check; a Qwen clone binds one when the
  clip is at most 30 s.
- **Voice direction**: the engine speaks a line with a given delivery when its settings carry an
  `instruction`, using classifier-free guidance (`cfg_scale`, default 4), which costs about 20 %
  more time. Both are part of the segment cache key.
- **Delivery by kind of line**: a multi-voice render picks the instruction from the segment.
  Narration in a first-person book and thoughts get the *inner voice* text; narration in a
  third-person book gets the *narration* text (empty by default, a plain clone); spoken lines
  get the *dialogue* text. Chapter titles are never directed, and voices on other engines are
  untouched (their cached audio stays valid). A book counts as first person when the narrator's
  voice is also a character's voice, which is what "Narrate in X's voice" in the cast plan sets
  up. The three texts and an on/off switch are per book, in the `delivery` block of
  `output/<book_id>/cast_plan.json`: edit them under *Design voices* in Review or with
  `seiyuu delivery <book_id>` (`--off`, `--inner-voice`, `--narration`, `--dialogue`). An edit
  re-records the Breeze lines it directs on the next render. A book with no cast plan uses the
  built-in texts with direction on; single-voice renders are never directed.
- **Long paragraphs**: the clip, both transcripts and the generated audio share a window of 2048
  positions, so a whole long paragraph does not fit in one request. A line of more than 800
  characters is spoken in pieces: it is cut at sentence ends into pieces of even length, each
  spoken with the same clip, seed and delivery instruction, and the audio is joined with a short
  pause. Shorter lines are sent whole, which covers ordinary prose (the longest line of a
  68,000-line web novel here is 574 characters). The whole line is still one cached segment and
  is checked by whisper as one. A piece that still does not fit (a very long reference clip) is
  refused, which stops the render with the block named.
- **Safety net**: like the other worker engines it is validated by whisper, a line that never
  stops is cut at a per-line cap, and a CUDA out-of-memory or a dead or hung worker is killed,
  restarted and retried before the segment fails loudly. `unload()` terminates the worker, which
  frees all its VRAM.

Until the clone and design flows offer Breeze, try it on a voice that already binds a transcript.
A single-voice render becomes the book's active render, so send the trial to a scratch folder:

```bash
uv run seiyuu render <book_id> --engine breeze --voice <voice_id> --chapter 1 --output-dir <scratch dir>
uv run python scripts/benchmark_engines.py --engines breeze qwen3tts
uv run pytest -m gpu -k breeze_real -s    # end-to-end smoke test (a few minutes on the GPU)
```

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

### Hosting through Cloudflare Tunnel

`seiyuu serve` runs the API and the built UI together on `127.0.0.1:8767` (change it with
`API_PORT` or `--port`; the host can't be changed). Build the UI first with
`npm --prefix frontend run build`. The app is only reachable while this command is running.

To put it on a public hostname, on a PC that already runs a tunnel:

1. Create a Cloudflare Access app for the hostname, with an Allow policy listing specific
   emails. Everyone on the list shares one library and can spend any configured paid API key.
2. Add these to `.env`, then restart. Set all three or none: a partial set stops the app from
   starting, and until all three are set the public hostname is refused.
   ```
   API_PUBLIC_HOST=seiyuu.example.com
   API_ACCESS_TEAM=<the part before .cloudflareaccess.com>
   API_ACCESS_AUD=<that Access app's Application Audience tag>
   ```
3. Add a route on the tunnel from the hostname to `http://localhost:8767`.

Every request through the tunnel needs a valid Access token (`api/access.py`), including the
UI's own files and any future WebSocket. Requests from this PC don't. Requests from another
site are refused either way (`api/origin.py`). Keep the API docs at `/docs` local: they don't
answer on the public hostname.

The UI has seven screens: **Library** (books, status, estimates), **Listen** (real-audio
read-along with a word-karaoke highlight and chapter auto-advance), **Character Review**
(reassign / rename / merge the cast, casting, spoiler-safe), **Pronunciation** (the per-book
lexicon), **Voice Studio** (upload → consent → audition; Kokoro blend builder; cloud-slot
view), **Series** (carry character voices across books), and **Render & Jobs** (the guided
cost flow, progress, downloads, and validation diffs).

## Project layout

```
src/seiyuu/
  ingest/        EPUB / PDF → normalized JSON
  attribute/     LLM speaker attribution (providers/ holds the LLM SDKs)
  normalize/     pure, deterministic text normalization
  voices/        voice library, blends, cloud slot manager
  engines/       TTSEngine adapters (Kokoro, Chatterbox, IndexTTS-2, Qwen3-TTS, Breeze, ElevenLabs) — the only SDKs
  render/        segment rendering + content-addressed cache + cost gate
  validate/      faster-whisper transcription validation
  assemble/      pauses, loudness, chapter MP3s, .m4b master
  gpu/           the single-GPU resource manager
  jobs/          single-flight durable job runner
  repository/    SQLite (WAL) + atomic file writes behind a seam
  api/           FastAPI app (routes/, uniform error envelope)
  services/      stage orchestration shared by CLI and API
  cli/           the Click CLI, one module per stage
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

The default test suite makes **no live LLM or TTS calls** — attribution tests use
scripted fake providers and canned JSON, and GPU work is behind
`@pytest.mark.gpu` (deselected by default).

## Status

Milestones **M1–M8 are complete**: the full CLI pipeline (ingest → attribute → normalize →
cast → render → validate → assemble → master), local + cloud attribution, the voice library
with cloning and blends, cloud TTS with the cost gate, a durable job/persistence layer, the
FastAPI backend, the React web UI, IndexTTS-2 as a second local cloning engine with native
emotion control (see [IndexTTS-2 setup](#indextts-2-optional-second-local-engine)), and PDF
ingestion. **M9** (guided UX + engine refresh) is planned; its research and plans are in
`design/` (`ux-upgrade-plan-2026-10-22.md`, `tech-refresh-2026-10-22.md`). See
[SPEC.md](SPEC.md) for the full design and the per-milestone design notes.
