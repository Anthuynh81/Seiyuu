# Seiyuu: overview and technology refresh (2026-10-22)

> **Decision (2026-10-22):** the RTX 3080 Ti 12GB is the target hardware. Read the
> 12GB column in the ranking table as binding; the 8GB column is informational. The rule of
> one heavy model resident on the GPU at a time still holds.

> **Check this first: the hardware assumption.** On this machine, `nvidia-smi` reports an **NVIDIA GeForce RTX 3080 Ti with 12288 MiB, driver 617.42**. I re-ran it during this review and got the same result. CLAUDE.md, SPEC.md and the M7 benchmarks all assume an RTX 3070 with 8GB. If this card is now the render box, several verdicts improve:
> - `qwen3.5:9b` fits fully on the GPU.
> - Qwen3-TTS 1.7B and VoxCPM2 become comfortable.
> - The CUDA 13 driver floor (R580+) is already met.
>
> The recorded IndexTTS-2 RTF and VRAM figures need re-measuring either way. **Even on 12GB, the rule of one heavy model resident at a time stays a hard rule** until you change CLAUDE.md. None of the recommendations below assume a TTS model and the LLM share the GPU. Where the verdict differs between the two cards, the table gives both. If you still want Seiyuu to run on 8GB cards, treat the 8GB column as binding.
>
> Other local facts checked:
> - ffmpeg is 8.1.1 (gyan.dev full build), which is current.
> - **The Ollama client is already 0.40.0.** The 0.31.2 and 0.34.4 structured-output fixes are already in place. Only the Windows orphaned-runner fix in 0.40.6 is missing.
> - Chatterbox-Turbo (`chatterbox/tts_turbo.py`) is present in the installed `.venv`.

---

## 1. Where Seiyuu stands

All milestones M1–M8, the quality roadmap (F1–F5) and the v1.1 follow-ups are merged to `main` (457eb8f, PR #26, 2026-07-22). No branch has had a commit since 2026-07-22.

| Stage | What is built (key modules) |
|---|---|
| **Ingest** | `ingest/__init__.py` picks a parser by file suffix (`.epub`, `.pdf`) and produces a `NormalizedBook` with stable `chNNN_bNNNN` block ids. `ingest/common.py` holds the shared chapterization, front/back-matter stripping and scene-break collapsing. `epub.py` uses ebooklib + BeautifulSoup and has a zip-bomb guard and cover extraction. `css_italics.py` resolves CSS-class italics for thought detection. `pdf.py` (M8) uses PyMuPDF, strips headers/footers, reflows text, builds chapters from the outline or a heuristic, and refuses scanned or encrypted PDFs. |
| **Normalize** | `normalize_text` is pure and memoized, with per-engine profiles (`profiles.py`). The per-book lexicon (`lexicon.py`, F3) supports IPA on the Kokoro profile only. `respell.py` is an opt-in LLM respell suggester that only advises. |
| **Attribute** | Reconstruction-by-construction. `spans.py` splits on quotes deterministically, with book-wide UK single-curly detection, and the LLM only names speakers. `chunking.py` gives each block one owner plus overlap context. The `AttributionLLM` providers are Ollama native `/api/chat` (default `qwen2.5:7b`), an OpenAI-compatible transport, and Anthropic via a forced-tool schema (default `claude-opus-4-8`, settings.py:173). Prompt v5 adds per-quote speakers (F1) and a closed 7-label emotion set (F2); v6 adds opt-in thought segments. `validate.py` is the reconstruction guard. The SQLite WAL cache is keyed by (book, chapter, chunk_hash, provider, model, prompt_version). Identity is handled by the registry, the deterministic alias post-pass `aliases.py`, and opt-in LLM adjudication in `adjudicate.py`. |
| **Voices / engines** | The `TTSEngine` template method (`engines/base.py`) forces canonical 24 kHz mono output. Engines: Kokoro-82M (28 presets plus blends), Chatterbox 0.1.7 (zero-shot clone, conds cache), **IndexTTS-2 as a cu128 subprocess worker** (M7, 8-dim emotion vector), and ElevenLabs (M5, LRU voice-slot manager). The voice library is file-first, with a consent attestation bound to the sha256 of `reference.wav`. Casting, series links and an LLM caster are in place. |
| **Render / validate / assemble** | `render/pipeline.py` does single and multi-voice renders, keeps a manifest per mode, and uses the content-addressed `SegmentKey` cache. `render/gate.py` issues HMAC-signed, single-use cost quotes. faster-whisper small/int8 on CPU runs validation and seed retries (`validation_min_ratio` 0.85). Whisper word timestamps drive read-along (`render/align.py`). `assemble/pipeline.py` inserts pauses, applies two-pass loudnorm per chapter or book, and writes MP3 plus a chaptered m4b. `gpu/manager.py` arbitrates one resident model and holds a cross-process `gpu.lock`. |
| **API / jobs / frontend / CI** | FastAPI with 65 route decorators. A durable single-worker SQLite job runner, a repository layer, and a services layer shared by the CLI and the API. The frontend is React 19.2, Vite 8, Tailwind 4.3 and TanStack Query 5 ("Talkback v4", 7 screens). Roughly 1,026 collected backend tests and about 200 vitest cases; both counts come from grep and PR notes, not a fresh run. GitHub Actions runs ruff, xdist pytest, oxlint, tsc and vitest. |

**In flight, not merged:**
- **PR #27, `render-wallclock`** (5 commits): voice-grouped multivoice rendering, whisper validation overlapped with synthesis, a Chatterbox conds memo, warm engine reuse and lazy GPU release. The GPU smoke suite was **not** run on it.
- **PR #28, `attribution-fastpath`** (c0db144): skips the LLM for chunks that are pure narration.

**Main remaining gaps (verified against the code):**
- **Attribution quality:**
  - There is no attribution accuracy eval. `qwen2.5:7b` has not been re-benchmarked since M2.
  - Hybrid escalation fires only when a chunk fails outright, which almost never happens. Escalated results are also cached under the primary provider's key.
  - Seam context blocks carry no speaker annotation.
  - `--chapter` re-runs start from an empty registry.
  - Thought Phase 2b (cue-verb thoughts) is deferred.
  - Guillemet and em-dash dialogue produce zero dialogue spans.
- **Render and audio:**
  - On `main`, renders still thrash between engines; only the unmerged #27 fixes that.
  - Voices are not loudness-matched to each other. Only whole-chapter loudnorm runs, so a quiet clone next to a loud narrator keeps the jump.
  - Reference clips are not curated. Uploads are copied as-is: no trim, normalize or length check, although SPEC promises them.
  - Per-voice engine tunables cannot be set through the CLI or API.
  - The segment cache is never pruned.
  - When a seed retry wins, the cache key still records the pinned seed, not the seed that produced the audio.
  - The IndexTTS-2 worker protocol version is not checked.
  - The CLI can use up a single-use cost token before preflight failures show.
- **Text:**
  - Normalization misses years ("1850" becomes a cardinal), times, units ("5km" becomes "fivekm"), fractions and negatives. "Ms." becomes "Miss".
  - EPUB ingest drops text in `div`, `li` and table cells, and reads footnote markers aloud. There is no OCR.
- **App and ops:**
  - No production serving: there is no StaticFiles mount, so it runs as two dev servers.
  - No auth beyond the Host allowlist, which matters on a LAN because there are paid-spend and delete endpoints.
  - No backup or export of truth data (reference clips, edits, assignments, lexicons).
  - Job progress is free text, with no units or ETA, and queued jobs are cancelled on restart.
  - `GET /api/books` re-parses every `normalized.json` on each poll. This hurts on the 27.5MB, 1,432-chapter book.
  - Review's VoicePicker has no audition button.
  - The voice dialogs hard-code engine lists instead of calling `GET /engines`.
  - The casting trait vocabulary is only {young, deep}.
- **Docs are stale:**
  - README: the Status section shows M7 in progress and M8 planned, it says "five screens", and it describes `cli.py` as a single file.
  - SPEC: the provider table still shows Qwen3.5 9B over `/v1`.
  - CLAUDE.md: it says attribution tests use responses "recorded" from a model, but they use scripted fakes and canned JSON.

---

## 2. What has changed in the tech landscape

**TTS (local)**
- **IndexTTS-2.5** shipped 2026-08-10 in the same repo ([HF](https://huggingface.co/IndexTeam/IndexTTS-2.5), [GitHub](https://github.com/index-tts/index-tts), [arXiv 2601.03888](https://arxiv.org/abs/2601.03888)):
  - Same 8-float emotion vector, in the same order.
  - About 6 GB in bf16.
  - The paper reports a 2.28× RTF improvement; the README only says "faster".
  - Adds `duration_factor` (0.5–2.0) and `<word|CMU>` pronunciation control.
  - v2 still loads from the same repo.
- **Chatterbox-Turbo (350M) is already installed.** `chatterbox/tts_turbo.py` ships in the pinned 0.1.7 with the same torch 2.6 and transformers 5.2 pins ([GitHub](https://github.com/resemble-ai/chatterbox)). Its source logs that it ignores `exaggeration`, `cfg_weight` and `min_p`. Resemble has moved Chatterbox to a "research archive", so expect slow maintenance.
  - Nano (110M, CPU) and Multilingual V3 require a master source install that keeps the same "0.1.7" version string.
  - On Python 3.11, master still pins torch==2.6.0. The torch>=2.9 marker applies only to Python ≥3.14.
- **Voice design from a text description** is now standard in Apache-licensed models: Qwen3-TTS 1.7B-VoiceDesign followed by Base cloning, an officially documented flow ([GitHub](https://github.com/QwenLM/Qwen3-TTS)), and VoxCPM2 ([GitHub](https://github.com/OpenBMB/VoxCPM)). Both upstreams warn that design output varies from run to run.
- **New benchmark leaders worth a bake-off:**
  - dots.tts: Apache, 2B, few-step distilled, 4.4 GB weights ([GitHub](https://github.com/rednote-hilab/dots.tts)).
  - LongCat-AudioDiT-1B: MIT, non-autoregressive ([GitHub](https://github.com/meituan-longcat/LongCat-AudioDiT)).
  - Qwen3-TTS-1.7B-Base: best independent English WER.
  - Fun-CosyVoice3-0.5B: Apache, Linux-first install.
- **CPU cloning tier:** Kyutai Pocket TTS, about 100M parameters, runs on 2 cores, clones from a wav; code MIT, weights CC-BY-4.0 behind an auto-approved gate ([GitHub](https://github.com/kyutai-labs/pocket-tts)). It can run auditions while Ollama holds the GPU.
- **Every new GPU engine pins torch ≥2.7 and cu128 or newer.** The IndexTTS-2 subprocess-worker pattern is now *the* integration path.
- **Whole-scene dialogue models** (MOSS-TTSD, VibeVoice, Fish S2/Drama 3, Higgs v3) are mostly too big, non-commercial or withdrawn. They also break the per-segment cache and validation model.

**Attribution LLM**
- **Qwen3.5 small models** (4B and 9B, released 2026-03-02) are the frontier at 8–12 GB ([HF](https://huggingface.co/Qwen/Qwen3.5-4B)). The 4B tag is 3.3 GB including vision. The hybrid attention design keeps the KV cache small.
- **Ollama:** 0.31.2 fixed `format` being ignored under `think:false` for qwen3.5 ([#14645](https://github.com/ollama/ollama/issues/14645)). 0.34.4 applies structured outputs on thinking models in a single pass. `/api/chat` now exposes **logprobs** ([releases](https://github.com/ollama/ollama/releases)). The installed 0.40.0 already has both fixes. Official tags are re-published in place, so the weights behind a `model_id` can change on the next pull.
- **Claude Haiku 5.5** (2026-10-07) costs **$0.10/$0.50 per MTok**, or $0.05/$0.25 on batch. It accepts `thinking:{type:"disabled"}` together with forced tool use. **Opus 5.5 and Sonnet 5.5 reject a forced `tool_choice` with a 400**, so pointing the current provider at either one would break it ([pricing](https://platform.claude.com/docs/en/about-claude/pricing), [thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)).
- **Research:**
  - ModernBookNLP (ModernBERT joint scoring) reaches 94.5% on PDNC against 89.8% for an 8B LLM, and is more than 1000× faster ([arXiv 2608.02359](https://arxiv.org/abs/2608.02359)). The 94.5% assumes a gold character list.
  - NAACL 2025 work supports Seiyuu's design of an alias list in the prompt plus JSON output ([ACL](https://aclanthology.org/2025.naacl-short.62/)).
  - LibriQuote shows that dialogue-tag speech verbs and adverbs ("he whispered") are high-precision emotion cues ([arXiv 2509.04072](https://arxiv.org/abs/2509.04072)).

**Validation and alignment**
- **Parakeet TDT 0.6B v2 int8** runs **without torch** through `onnx-asr` (MIT) or `sherpa-onnx` (cp311 Windows wheels) ([HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [onnx-asr](https://github.com/istupakov/onnx-asr)):
  - Weights are CC-BY-4.0.
  - LibriSpeech-clean WER is 1.69.
  - CPU RTF is about 0.03–0.12.
  - Its transducer decoder is less prone than Whisper to inventing text.
  - Both runtimes return token timestamps only, so we would group tokens into words ourselves.
- **Native engine timestamps:**
  - ElevenLabs `/with-timestamps` gives character alignment on the same paid call ([docs](https://elevenlabs.io/docs/api-reference/streaming-with-timestamps)).
  - Kokoro's `KPipeline` reportedly exposes per-token `start_ts`/`end_ts`. This comes from community code and has not been checked against the official repo.
  - Either would remove the post-hoc whisper alignment pass for that engine.
- **Qwen3-ForcedAligner** (Apache, about 0.9B parameters, about 1.8 GB in bf16, GPU) has about 43 ms mean shift, against about 130 ms for NFA and WhisperX ([HF](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)).

**Cloud TTS: prices have dropped**
- **ElevenLabs:** the API list price is **$0.08 per 1k characters** for Multilingual v2, v3 and v4, and $0.04 per 1k for Turbo and Flash ([pricing](https://elevenlabs.io/pricing/api)).
  - Seiyuu's `elevenlabs_price_per_1k_chars` defaults to **$0.30** (settings.py:180), which is about 3.75× the API list price.
  - v4 (2026-09-28) exposes only Stability and Similarity: no Style, Speed or SSML. It is also "under continuous development", so the output is not pinned.
- **Fish s2.1-pro** costs $15 per million UTF-8 bytes, about $9 per novel ([docs](https://docs.fish.audio/developer-guide/models-pricing/pricing-and-rate-limits)).
  - It sends **inline reference audio on every request**, so there are no voice slots to manage.
  - There is no seed.
  - There is a $0 fair-use tier with no service guarantees.
- **Inworld TTS-2 / TTS-2 Flash** costs $15–25 per million on demand, about $9–15 per novel ([pricing](https://inworld.ai/pricing)).
  - Free instant cloning from 3–30 s of audio.
  - LINEAR16 output at 24 kHz.
  - No seed; `deliveryMode=STABLE` is the closest substitute.
  - Cloned voice IDs live in a workspace, so the engine needs recreate-on-404 handling.
- **Gemini 3.8 Flash TTS** produces native 24 kHz mono PCM at about $0.81 per audio hour, half that on batch. Prices double on 2027-01-01 ([docs](https://ai.google.dev/gemini-api/docs/speech-generation)). It does voice design, but cloning needs a recorded consent statement from the speaker, so it cannot clone arbitrary character clips.
- **OpenAI's batch TTS** shuts down on 2027-01-06.

**Tooling**
- torchaudio is in maintenance mode, and 2.11 is its last release ([PyPI](https://pypi.org/project/torchaudio/)). Seiyuu's only direct use is `engines/audio.py:60`, and `soxr` is already in `uv.lock`.
- cu126 wheels stop after torch 2.14 ([notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)). The wheels already hosted keep the current lock working.
- uv documents extras plus `conflicts` with a torch index per extra ([docs](https://docs.astral.sh/uv/guides/integration/pytorch/)).
- Competitors ship features Seiyuu lacks. ebook2audiobook and Abogen push to Audiobookshelf and export SRT/ASS subtitles. Alexandria creates voices from a text description.
- **Calliope** shows that EPUB3 Media Overlays generated from TTS timing have zero drift. Storyteller's post-hoc alignment drifted by up to about 35 s ([arXiv 2602.10735](https://arxiv.org/abs/2602.10735)).
- The frontend stack is current, and no migration is needed.

---

## 3. Ranked improvement opportunities

How to read the table:
- **Seam** means the change goes through an existing extension point. **Arch** means it is an architectural change.
- **⚠** marks an area that needs sign-off under CLAUDE.md: the voice model, the normalized JSON schema, the Segment schema, or a cache-key format.
- **$** means a paid call. Every $ item stays behind the existing cost-confirmation and paid-call gates and never runs on an automatic path.

| # | Adopt | Seam / module | Main gain | GPU fit (8GB / 12GB) | Effort | Risk | Sign-off |
|---|---|---|---|---|---|---|---|
| 0 | Merge #27 + #28 after the GPU smoke run | render/, attribute/ | Speed (already built) | ✓ / ✓ | S | Low | No |
| 1 | IndexTTS-2.5 in the existing worker | Seam: `indextts2_worker.py` | Speed about 2× (paper), pacing, phoneme control | ✓ (~6 GB) / ✓ | S–M | Low | No (new `model_version`) |
| 2 | Attribution eval harness + local LLM A/B | New CLI/script; providers | Objective quality; unblocks #6/#9 | ✓ / ✓ | M | Low | No |
| 3 | Inter-voice loudness matching | assemble/pipeline.py | Removes volume jumps between voices | CPU | S–M | Low | No (assembly only, cache untouched) |
| 4 | ElevenLabs: per-model price, `eleven_v4`, native timestamps | Seam: elevenlabs_engine.py, settings | Honest quotes; quality; no alignment pass | cloud $ | S–M | Low | Words sidecar: propose |
| 5 | Haiku 5.5 as `anthropic_model` (thinking disabled) | Seam: providers/anthropic.py config | Cloud attribution about $4.5 → about $0.09 per novel | cloud $ | S | Low | No |
| 6 | Default local model → Qwen3.5-4B or 9B, only if it wins #2 | Seam: `attribution_model` setting | Quality | 4B ✓, 9B borderline / 9B ✓ | S | Med | No (`model_id` is already in the key) |
| 7 | Chatterbox-Turbo engine id | Seam: new class in engines/ | Fast cloning tier, no new dependency | ✓ / ✓ | S | Low | No |
| 8 | Parakeet TDT validator (onnx-asr) | Seam: Validator transcribe injection | Stricter, faster validation; no torch | CPU | M | Med (thresholds need retuning) | No |
| 9 | Confidence-based hybrid escalation | attribute/pipeline.py + providers | Quality where it matters | ✓ / ✓ | M | Med | No |
| 10 | Generic subprocess-worker base | Arch-lite: extract from indextts2_engine | Cheaper path for every new engine | — | M | Low | No |
| 11 | Deterministic reference-clip prep | voices/, api/routes/voices.py clone route | Better clones; delivers what SPEC promised | CPU | S–M | Low | Re-attestation flow (see notes) |
| 12 | Deterministic gate checks + speaker similarity | render gate / validate/ | Catches failures WER misses | CPU | M | Low | No |
| 13 | Fish s2.1-pro cloud engine | Arch: generic paid-engine hook first | Cheapest cloud cloning | cloud $ | M | Med | No |
| 14 | CPU preview/audition engine (Pocket TTS) | New engine, worker env | Auditions while Ollama holds the GPU | CPU | M | Low | No |
| 15 | Read-along from source text | render/align.py | Highlighted words match the book exactly | CPU | M | Low | SegmentWords format: propose |
| 16 | Exports: EPUB3 Media Overlays, Audiobookshelf/OPF, SRT | New export module after assemble/ | Distribution, differentiation | CPU | M–L | Low | Needs a block → XHTML id plan |
| 17 | ML reference curation (isolation, restoration, scoring) | voices/ via worker | Clones from poor sources | ✓ / ✓ | M | Med | Keep the original upload |
| 18 | Voice design → frozen `reference.wav` | voices/ + new worker | Cast minor characters without clips | likely ✓ (unmeasured) / ✓ | L | Med | ⚠ Voice model |
| 19 | TTS bake-off: dots.tts, Qwen3-TTS-Base, LongCat-1B, CosyVoice3 | Workers via #10 | Possible quality step | varies | M | Low (spike) | No |
| 20 | Context-aware rendering (previous text/audio) | engines + SegmentKey | Prosody continuity | — | L | High | ⚠ Cache key |
| 21 | ModernBookNLP cross-check | attribute/ via worker | Second-opinion confidence | unmeasured / ✓ | L | High | No |
| 22 | Remote rented-GPU worker (optional) | Worker transport over the network | Fast renders without the local GPU | remote $ | L | Med | No |

**Details**

**0. Merge the in-flight work.** This is not new technology, but it is the highest-value change available. `main` still reloads engines on every voice alternation and validates serially. Run `uv run pytest -m gpu` first. It is a multi-minute GPU job, so ask before starting it.

**1. IndexTTS-2.5.** Pull the repo and run `uv sync` in the existing cu128 worker env. This is a separate venv, so the main env's torch is untouched.
- **What stays the same:** the emotion mapping in `voices/emotion.py`, because the vector order is identical. The checkpoint fingerprint produces a new `model_version`, so v2 cached segments stay valid.
- **Code changes:**
  - Switch the worker import to `indextts.infer_v2_5.IndexTTS2`.
  - Make the version selectable, so v2 and v2.5 can be A/B tested side by side.
  - Expose `duration_factor` as a tunable; it goes into `settings_hash`.
  - Run in bf16, which Ampere supports.
  - Skip DeepSpeed and vLLM on Windows.
- **Before trusting the cache:** verify that seeding gives deterministic output with `use_random=False`.
- **Later:** `<word|CMU>` could let `lexicon.py` emit real phoneme overrides for this engine. The whisper reference must stay the plain respelling.
- **License:** bilibili Model Use License. Commercial use is allowed below 100M MAU or RMB 1B annual revenue.

**2. Measure before switching the local model.** The model-card IFEval numbers (89.8 vs 71.2) are from thinking mode, and Seiyuu runs with `think:false`, so they settle nothing.
- **Gold labels:** Character Review `edits.json` from real books, plus PDNC and deezer's post-cutoff "Dark Corners" novel ([repo](https://github.com/deezer/llms_quotation_attribution)) for an offline harness. Keep the CI suite on scripted fixtures.
- **Metrics:** per-quote speaker accuracy (explicit, anaphoric and implicit), schema failures, alias drift, and seconds per chunk.
- **Arms:**
  - `qwen2.5:7b` (current).
  - `qwen3:4b-instruct-2507` (2.5 GB, the non-thinking control).
  - `qwen3.5:4b` with think off.
  - `qwen3.5:9b` on 12 GB. On 8 GB, use a text-only GGUF imported through a Modelfile with `RENDERER qwen3.5` / `PARSER qwen3.5`, and confirm 100% GPU in `ollama ps`. Ask before the roughly 5 GB pull.
- **Same harness, extra variables:**
  - `attribution_chunk_tokens` at 800, 1600 and 2000.
  - A registry-id enum for the speaker field on the local path.
  - Feeding earlier overlap-block speaker calls back into the prompt.
  - Compact registry rendering. Today the full registry is dumped as indent-2 JSON in every chunk.
- **Ollama:** the client is already 0.40.0, so upgrade to ≥0.40.6 for the Windows orphaned-runner fix. Add a startup probe: call `/api/show`, check the thinking values, then make one tiny `think:false` + `format` call and confirm the JSON is schema-valid. Fail loudly if any step fails.
- **Reproducibility:** record the model digest from `/api/tags` in the run manifest now. Putting it in the cache key would be a ⚠ change.

**3. Inter-voice loudness matching.** Today one linear loudnorm gain is applied per chapter, so a quiet clone next to a loud Kokoro narrator keeps the jump. Fix it in assembly:
- Measure integrated loudness per segment, or per voice per chapter, with ffmpeg `ebur128` or `pyloudnorm`.
- Apply a bounded gain toward a common target before the existing chapter loudnorm.
- `pyloudnorm` is currently only a transitive entry in `uv.lock`. Declare it if you use it, then re-run the torch CUDA check.

Leave `to_canonical` and cached WAVs alone so the cache is unaffected. Add a fixture test with two segments at different levels.

**4. ElevenLabs.**
- Replace the single $0.30 default with a per-model rate ($0.08 per 1k for v2/v3/v4, $0.04 per 1k for Turbo and Flash), settable for users on plan pricing. This is the cheapest fix in the list.
- Add `eleven_v4` and `eleven_v4_turbo`. v4 drops `style` and `speed`, so `emotion.py` needs a v4 mapping, possibly onto inline audio tags.
- Use `/with-timestamps` so the lazy whisper alignment for this engine goes away. Check on a live call that `pcm_24000` is returned on that endpoint.
- Defer `previous_text` / `next_text`. Context changes the output, so it belongs with #20.

**5. Haiku 5.5.**
- **Model and parameters:** change the default `anthropic_model` from `claude-opus-4-8` and send `thinking:{type:"disabled"}`. Keep the current forced-tool path, which Haiku accepts. Remove any temperature, top_p or top_k parameters.
- **Premium models:** expose Sonnet 5.5 and Opus 5.5 only after migrating to `output_config.format`, because both reject forced tools. Their per-novel estimates of about $1.8 and $3.6 exclude always-on thinking tokens.
- **Cost reductions:**
  - A Message Batches path (50% off) suits whole-book runs.
  - Prompt caching needs `render_prompt` split into static and dynamic parts. A per-book registry enum would invalidate the cache, so on Claude keep a string speaker field and validate it against the registry afterwards.
- **Gating:** all of this runs only when hybrid or Anthropic is explicitly enabled.

**6. Switch the default only on evidence from #2.** `model_id` is already part of the cache key, so old results stay valid. Re-record fixtures if any are model-derived.

**7. Chatterbox-Turbo.** Add it as its own engine id (for example `chatterbox_turbo`) so the cache key, casting and the GPU manager treat it separately.
- It ignores `exaggeration` and `cfg_weight`. Map only sound-type emotions to its tags (`[laugh]`, `[sigh]`, `[gasp]`), and keep the 0.5B model for emotional lines.
- Test whether style tags such as `[angry]` work on the PyTorch path. Issue #557 concerns the MLX port.
- The first run downloads new weights.
- Do not install master (Nano or V3) into the main env; trial it in a throwaway venv.

**8. Parakeet validator.** Write an adapter to the whisper-shaped `transcribe()` seam:
- Load the model once and reuse it.
- Resample 24 kHz to 16 kHz.
- Group BPE tokens into words.
- Retune `validation_min_ratio` on recorded renders.

Record the validator id in `.validation.json` so verdicts made by whisper can be recognized and re-checked. Keep faster-whisper as a configurable fallback (non-English, second opinion). Adding onnxruntime to the main env should not touch torch, but run the CUDA check anyway. This is driven by quality, not urgency.

**9. Escalation.** Add an `attribution_escalate_below` setting.
- **Start:** trigger on the per-segment confidence that already exists, with review threshold 0.7.
- **Then spike:** Ollama logprobs on the speaker field. How logprobs behave under grammar-constrained decoding, and with several per-quote speaker fields, is unverified.
- **Tiers:** local 4B → optional local 9B → Haiku 5.5, only when explicitly enabled ($).
- **Bug fix:** cache escalated results under the provider that actually produced them. This uses the existing key format, so no ⚠.

**10. Generic worker base.** `SubprocessTransport` and `WorkerError` are hard-coded to IndexTTS-2. Extract a small base: NDJSON over stdio, ready handshake, an **enforced** protocol-version check (currently missing), synchronous-kill unload, OOM restart, and an injectable transport for CPU tests. After that, each future engine is one adapter plus one worker script.

**11. Deterministic reference-clip prep.** SPEC promises trim, loudness normalization, resampling and a noise warning, and none of it exists. Add these to the clone route with ffmpeg and soundfile:
- Trim leading and trailing silence.
- Cap the length at about 30 s and warn above that. An M7 spike once used a 20-minute reference.
- Resample.
- Normalize loudness.
- Run a simple SNR or clipping check.

`reference.wav` is the consent-attested source of truth (sha256-bound), so curation must run **before** attestation and keep the original upload beside it. Re-curating an existing voice is a re-clone: re-attest, purge conds, cached segments and the IVC handle.

**12. Gate checks.** These are pure functions run inside the existing seed-retry loop. They apply to local engines only; paid engines flag for review and never auto-retry.
- Duration per normalized character against the voice's own running rate, which catches loops and truncation.
- The minimum ASR token log-prob.
- A Silero VAD check for audio after the last aligned word, which catches tail babble.

Speaker similarity against `reference.wav` should use an ONNX embedding model through sherpa-onnx, which keeps it out of torch; the exact model list there is unverified. SpeechBrain ECAPA (Apache) is the fallback in the torch-2.6 env. Start it as a soft review flag, calibrated per voice, and make it a retry trigger only once the data separates cleanly. Cache reference embeddings per voice, keyed by embedding-model version.

**13. Fish.** The engine is simple. Inline `references` means sending `reference.wav` plus its transcript on each request, with no cloud voice IDs. The work is in the seams:
- **Generalize the paid engine.** "elevenlabs" is hard-coded as *the* paid engine in `api/routes/render.py`, `api/routes/voices.py`, `render/pipeline.py:820`, `cli/voices.py` and the frontend dialogs. Add a generic paid-engine and cloud-voice hook first.
- **Consent.** Re-verify consent before every upload, as the ElevenLabs path does.
- **Validation.** CLAUDE.md requires whisper validation for Fish, but **failed validation must not trigger automatic paid retries**. Flag the segment for review, or allow a retry budget that the signed quote and `max_paid_usd` explicitly include. ElevenLabs sets `requires_validation=False` for exactly this reason.
- **Free tier.** `s2.1-pro-free` is for auditions only. It is still a cloud upload, with no DPA.
- **Live checks.** Confirm that `sample_rate=24000` is accepted for wav and pcm.
- **Alternative.** Inworld TTS-2 Flash is the second option if Fish quality or terms disappoint.

**14. CPU preview tier.** Pocket TTS (about 100M parameters, 2 cores, clones from a wav) as an audition and draft engine that never competes for VRAM. It needs torch ≥2.5. Put it in a worker env via #10, or install it in the main env only if a dry run shows torch is untouched, then re-run the CUDA check. Chatterbox-Nano is the alternative once a master install is acceptable.

**15. Read-along.**
- Kokoro and ElevenLabs: use their native timings.
- Chatterbox and IndexTTS: align Parakeet token timings to the *source* segment text with difflib, so highlighted words always match the book.
- Premium option: Qwen3-ForcedAligner in a worker through the GPU manager.
- Avoid MMS_FA (CC-BY-NC) and WhisperX (torch 2.8 pin).

**16. Exports.**
- **EPUB3 Media Overlays:** build SMIL from manifest timings, segment-level first and word-level later. This is the most distinctive feature available. It needs a plan for mapping blocks back to source XHTML ids.
- **Audiobookshelf:** write the `{Author}/{Series}/{N - Title}` layout plus an **OPF** sidecar, which is documented and portable, not ABS's internal `metadata.json`. Upload and scan are optional, behind a settings URL and token. Implement them from the server source, because the API docs are officially unmaintained. Add new endpoints to `test_api_m6b6.py`.
- **Tags and subtitles:** mutagen for m4b narrator and series tags. SRT/ASS generated from word timings.

**17. ML reference curation.** Every candidate conflicts with the main env, so run curation in a worker:
- clearvoice pins librosa 0.10.
- audio-separator needs numpy ≥2.
- Sidon needs torch ≥2.8.

Order: vocal isolation (MelBand-RoFormer), then restoration (Sidon, MIT; or LavaSR, Apache, about 500 MB VRAM), then a DNSMOS/NISQA score. The same consent rule as #11 applies.

**18. Voice design.**
1. An LLM writes a casting note from registry facts.
2. Qwen3-TTS 1.7B-VoiceDesign (Apache, its own Python 3.12 worker) renders about 15 s of audio.
3. A human auditions it and picks a take.
4. The chosen take is frozen as `reference.wav` with the seed pinned.

A designed voice has no consent subject, so `VoiceMeta` needs a new provenance or kind (⚠ voice model). Memory is about 4–4.5 GB by third-party estimates, which is fine under the one-model rule on either card but unmeasured. VoxCPM2 is the alternative at about 8 GB; its FP8 build needs Ada or newer, so it does not help an Ampere card. Gemini voice design ($) is a paid alternative; store its `sample_audio` as `reference.wav`, because designed IDs expire.

**19. Bake-off.** Do this after #10. Measure RTF, peak VRAM and whisper retry rate against IndexTTS-2.5 on real chapters:
- dots.tts: top claimed Seed-TTS SIM.
- Qwen3-TTS-1.7B-Base: best independent English WER.
- LongCat-AudioDiT-1B: non-autoregressive, so no looping.
- CosyVoice3-0.5B: Linux-first.

**20. Context-aware rendering.** This is the frontier. The simplest form is ElevenLabs `previous_text` / `next_text`; the local form is VoxCPM2-style continuation. Context changes the output, so it has to enter the TTS cache key (⚠). Leave it for later.

**21. ModernBookNLP.** Experimental.
- It pins transformers 4.57.6, so it needs a worker.
- The license and weight hosting are not stated.
- The shipped checkpoint peaks at about 9.2 GB: fine on 12 GB, unmeasured on 8 GB.
- Its quotes must be aligned to `spans.py` offsets, and agreement with the LLM must be calibrated before it auto-accepts anything.

**22. Remote GPU worker.** Running the existing IndexTTS or Chatterbox worker on a rented 3090 or 4090 ($0.22–0.89/h) keeps arbitrary-clip cloning, seeds and the emotion vector. The per-book cost is probably under $5, but that is unmeasured. It is paid, it needs a network transport for the worker protocol, and it uploads `reference.wav` to third-party hardware. It must be explicitly confirmed per job, and it is a lower priority now that the local card may be 12 GB.

**Where the current choice is still best:**
- **Kokoro:** keep it. No successor weights exist.
- **Ollama native transport:** keep it.
- **Reconstruction-by-construction:** keep it. The literature supports it.
- **Python 3.11:** keep it. Kokoro caps at <3.13 and IndexTTS at <3.12.
- **ffmpeg 8.1:** keep it.
- **faster-whisper:** keep it as the fallback.
- **Frontend:** no migration. Optional: bump TanStack Query to 5.104, and use `<Activity>` to preserve unsaved Review edits. React Compiler is low priority. Its safety lints need ESLint (the repo uses oxlint), and it does not stop PlayerTimeContext consumers from re-rendering on every tick.
- **Skip:**
  - Fish S2 Pro and Drama 3 run locally.
  - ZONOS2, Higgs v3, Voxtral, VibeVoice, Dia2, FireRedTTS-2 (9 GB in bf16).
  - OmniVoice. Its HF card says the weights are CC-BY-NC; the repo code is Apache.
  - MiniMax (more expensive than ElevenLabs).
  - Hume (pricing withdrawn).
  - OpenAI TTS (shutting down), Azure MAI (preview, gated cloning), xAI Grok TTS (passphrase-gated cloning).
  - gpt-oss-20b, Qwen3.6/3.8 and Gemma 4 12B (none fit).
  - WhisperX, MMS_FA.

---

## 4. Ways to make the project simpler

1. **Drop torchaudio from Seiyuu-owned code.** `engines/audio.py:60` is the only use, and `soxr` 1.1.0 is already in `uv.lock` via librosa. Add a parity fixture test for 24 kHz mono 16-bit output. *Tradeoff:* torchaudio stays installed transitively while Chatterbox runs in-process.
2. **Declare the CPU/CUDA torch split in the lockfile.** Use uv extras `cpu` and `cu126` with `[tool.uv] conflicts`, and declare torch and torchaudio directly in each extra so the index mapping overrides chatterbox's transitive pin. CI then runs `uv sync --extra cpu` instead of the hand-written reinstall step. *Tradeoff:* a one-time pyproject rework, and Windows-resolved lock behavior must be re-checked on Linux. Re-run the CUDA check afterwards.
3. **One worker protocol for every out-of-process engine (#10).** It replaces per-engine transport code and makes engines removable.
4. **Long term, and only with your approval: a main env without CUDA torch.** Move Chatterbox, with its current torch 2.6 pins, into its own worker venv. Combined with Parakeet validation (no torch) and possibly `kokoro-onnx`, the app env could stop depending on CUDA torch. That would remove the "CPU wheel silently replaced CUDA torch" failure class and make the cu126 end-of-life irrelevant.
   - *Tradeoff:* `engines/base.py` and `engines/audio.py` import torch at module top and would need reworking.
   - Kokoro blending would need re-validating on ONNX.
   - Every engine pays a process boundary and a WAV handoff.
   - It changes the CLAUDE.md environment and GPU rules, so propose it first.
   - Don't use Chatterbox's Python 3.14 path: torchaudio stops at 2.11, so that branch caps around torch 2.11, and the cp314 Windows wheels for its other dependencies are unverified.
5. **Collapse IndexTTS-2 into 2.5** after the A/B: same env, one fewer model.
6. **Remove the post-hoc whisper alignment for Kokoro and ElevenLabs** by using native timestamps (#4, #15).
7. **No new cloud voice-slot logic for Fish.** The `voices/cloud.py` LRU and recreate logic stays ElevenLabs-only. Inworld, if added, would reuse it.
8. **Consider retiring Chatterbox 0.5B** if IndexTTS-2.5 covers emotional cloning and Turbo covers fast cloning. *Tradeoff:* Chatterbox is MIT, IndexTTS is under the bilibili license, and Chatterbox is the lighter VRAM fallback. Decide after the bake-off, not before.
9. **Opt-in cloud attribution on Haiku 5.5 at about $0.10 per book** makes a local 9B escalation tier less necessary. *Tradeoff:* it is a paid call and goes against the offline default, so it stays strictly opt-in.

---

## 5. Suggested sequencing

**Phase 0: housekeeping (days)**
- Confirm which GPU is the target. If it is 12 GB, decide whether CLAUDE.md's "assume 8GB" stays as a portability floor; the one-resident-model rule stays either way.
- Run `pytest -m gpu` (ask first), then merge #27 and #28.
- Upgrade Ollama 0.40.0 to ≥0.40.6, add the `/api/show` probe, and start recording the model digest in the manifest.
- Fix the ElevenLabs default price (#4, price part only).
- Fix the stale docs:
  - README: Status shows M7/M8 as done; 7 screens; `cli/` package; "grouped by voice" only after #27.
  - SPEC: provider table now shows qwen2.5:7b on native `/api/chat`; drop the "frontend has no tests" gap.
  - CLAUDE.md: describe tests as scripted fakes, not recorded fixtures.
- Swap resampling to soxr.

**Phase 1: measure (propose, then approve; GPU jobs need a go-ahead)**
- Build the attribution eval harness and run the LLM A/B (#2).
- Write a TTS bake-off script recording RTF, peak VRAM and retry rate, and run IndexTTS-2, 2.5 and Chatterbox-Turbo first, on the actual card.

**Phase 2: drop-ins through existing seams**
- IndexTTS-2.5 (#1), loudness matching (#3), ElevenLabs v4 and timestamps (#4), Haiku 5.5 (#5).
- Local model switch if the data supports it (#6), the Chatterbox-Turbo engine (#7), the Parakeet validator with retuned thresholds (#8), deterministic reference prep (#11).

**Phase 3: architectural (a short plan each, approval before coding)**
- Generic worker base (#10), then the generic paid-engine hook and the Fish adapter (#13), then the Pocket TTS preview engine (#14).
- Escalation plus the cache-attribution fix (#9).
- Gate checks (#12).
- Source-text read-along (#15; propose the SegmentWords format).
- Exports (#16; plan the block → XHTML id mapping).
- Remaining debt:
  - Registry seeding on `--chapter` runs.
  - Seam speaker annotation and the v7 prompt.
  - Stop `free_all()` after every attribution run.
  - Normalization gaps (years, times, units, fractions).
  - EPUB `div`/`li`/footnote handling.
  - CLI render preflight.
  - Per-voice tunables via the API.
  - Segment-cache GC that keeps paid entries.
  - StaticFiles production serving.
  - Export or backup of truth data.

**Phase 4: experiments and gated features**
- dots.tts, Qwen3-TTS and LongCat spikes (#19); ML reference curation (#17).
- Voice design (#18, ⚠ voice model); per-emotion reference variants (⚠ voice model).
- Context-aware rendering (#20, ⚠ cache key); ModernBookNLP (#21); remote GPU worker (#22).

---

## 6. Caveats: unverified claims and what to spike

- **Hardware:**
  - `nvidia-smi` shows a 12 GB RTX 3080 Ti.
  - All recorded RTF and VRAM figures (IndexTTS-2 RTF about 1.3×, peak about 7.8 GB) come from the 3070.
  - None of the new-engine RTFs in this report has been measured on either card.
  - No Chatterbox or Kokoro RTF has ever been recorded.
- **IndexTTS-2.5:**
  - Seed determinism with `use_random=False` is unverified.
  - So is the speedup on Windows/Ampere without DeepSpeed or vLLM. The 2.28× figure is from the paper.
- **Chatterbox-Turbo:**
  - Style tags on the PyTorch path are untested.
  - The installed code uses 2 CFM steps, not the 1 step the README claims.
- **Qwen3.5:**
  - Card scores are from thinking mode, so the `think:false` gap against qwen2.5:7b is unknown.
  - The text-only 9B Modelfile import (RENDERER/PARSER) is unverified.
  - FA + q8_0 KV has an open crash report (#17434), so leave the KV cache at f16. Qwen3.5's full-attention KV is only about 256 MB at 8K anyway.
- **Ollama logprobs:** their meaning under grammar-constrained decoding is unverified.
- **Claude:**
  - Cost estimates assume about 0.6M input and 60K output tokens per novel.
  - Haiku 5.5's tokenizer counts about 30% more tokens than Haiku 4.5.
  - The Opus and Sonnet 5.5 estimates exclude thinking tokens.
- **Fish:**
  - `sample_rate=24000` for wav/pcm is undocumented; the docs show 44100.
  - There is no seed.
  - The S2 bracket-tag syntax was not on the TTS page checked.
  - The SDKs' default models are unverified. The docs say the Python SDK defaults to s2-pro.
- **ElevenLabs:**
  - Request stitching on v4 is ambiguous; the docs say v3 is unsupported.
  - Voice-slot counts per tier are unverified (the help page returned 403).
  - `/with-timestamps` with `pcm_24000` is unverified.
- **Kokoro native timings:** these come from community code, not the official docs.
- **Inworld:** the TTS-2 launch date and the consent terms are unverified.
- **Gemini 3.8 TTS:**
  - GA status is not stated.
  - Whether the seed is honored is unverified.
  - Replication is unavailable in some regions (including the EEA and UK).
  - 600k characters is about 13.8 h of audio by AWS's own example, not 11 h. Plan on about $11–12 standard or about $6 batch in 2026, and roughly double in 2027.
- **Parakeet:**
  - Word grouping is our own code.
  - Whether onnx-asr exposes full frame posteriors, which a true CTC Viterbi needs, is undocumented.
- **Speaker drift:** the cited drift paper uses general encoders plus an LLM judge. Its "F1 0.62 at a fixed threshold" figure is unverified. Calibrate per voice.
- **ModernBookNLP:**
  - License and weight hosting are unconfirmed.
  - Fit on 8 GB with the shipped T=2000 checkpoint is unmeasured.
  - The 94.5% assumes a gold character list.
- **Bake-off candidates:**
  - dots.tts, LongCat and Ming-3.0 quality claims are self-reported.
  - TADA is under the Llama 3.2 Community License, and its 3B model is about 9 GB.
  - MOSS-TTS's "8B fits in 8 GB" llama.cpp path is a vendor claim and Linux-oriented.
- **Reference curation:**
  - clearvoice's SpeechScore dependencies may need source builds on Windows.
  - deepfilternet-slim is an unvetted repackage from a single maintainer.
- **Exports:**
  - The Audiobookshelf upload field names must come from the server source.
  - Storyteller's license and aligner model are unverified.

---

## Critic notes

- **Ollama:** checked locally. The client is already **0.40.0**, so I replaced "upgrade to ≥0.34.4" with "upgrade to ≥0.40.6 for the Windows orphaned-runner fix" and noted that the format and think fixes are already present.
- **Hardware:** re-ran `nvidia-smi` and confirmed the RTX 3080 Ti 12GB. I made explicit that the one-heavy-model-resident rule stays binding on 12GB, and that the 8GB column is the floor if portability matters.
- **Turbo and the price default:** confirmed `tts_turbo.py` is in the installed venv, and the ElevenLabs price default is $0.30 (settings.py:180). I corrected "about 4×" to "about 3.75×", noted plan-based pricing, and cited the settings line.
- **Constraint violation fixed (Fish, #13):** the draft made whisper validation with seed retries mandatory for a paid engine. That would trigger automatic paid calls. Paid engines now flag failures for review, or retry only within a budget the quote explicitly includes. The same rule applies to the gate checks (#12). I also added a consent re-check before every reference upload.
- **Omitted candidates and gaps added to the ranking:**
  - Inter-voice loudness matching (#3), a verified audio defect that is cheap to fix.
  - Deterministic reference-clip prep (#11), which SPEC promises. It must run before consent attestation, because `reference.wav` is sha256-bound.
  - Pocket TTS CPU preview tier (#14).
  - Remote rented-GPU worker as an optional paid item (#22).
  - Inworld TTS-2 as a cloud alternative.
- **Section 1 gaps added:** seam annotation, registry seeding, guillemet and em-dash dialogue, per-voice tunables, the retry-seed mismatch, the unenforced IndexTTS protocol check, the CLI preflight token burn, no StaticFiles or auth, no backup, job-progress limits, `/api/books` re-parsing, the VoicePicker without audition, the hard-coded engine lists, and the {young, deep} trait vocabulary.
- **Ranking:**
  - Moved the eval harness to #2, ahead of Haiku. It is the prerequisite for the model switch and escalation, and Anthropic is opt-in.
  - Inserted loudness matching high.
  - Renumbered the table, the details and the sequencing to match.
- **Escalation (#9):** now starts from the existing per-segment confidence (review threshold 0.7). Logprobs became a spike, since their behavior under constrained decoding is unverified.
- **Parakeet (#8):** added recording the validator id in `.validation.json`, so whisper-era verdicts can be recognized. Removed the confusing "current choice stays defensible" sentence. Added the CUDA check after installing onnxruntime.
- **Speaker similarity:** now prefers ONNX via sherpa-onnx (no torch) over SpeechBrain in the pinned env.
- **IndexTTS-2.5 (#1):** added the concrete worker changes (the `infer_v2_5` import, a version selector, `uv sync` in the separate venv only) and corrected the license threshold wording.
- **Haiku (#5):** added the tokenizer inflation, the conflict between prompt caching and a registry enum, and that Sonnet and Opus estimates exclude thinking.
- **Voice design (#18):** the 8GB fit went from "borderline" to "likely ✓ (unmeasured)", consistent with the 3.8–4.5GB estimates under the one-model rule. Added Gemini design as a paid alternative, and noted that VoxCPM2's FP8 build doesn't help Ampere.
- **"Simplify" item 4 (CUDA-free main env):** marked as long-term and approval-gated, because `engines/base.py` and `engines/audio.py` import torch and it changes the CLAUDE.md environment rules.
- **Caveats added:** Kokoro native timings (community-sourced), `/with-timestamps` with `pcm_24000`, Ollama logprob semantics, Inworld terms, Gemini regional limits, and the Haiku tokenizer.
- **Vague advice made specific:**
  - Exactly which stale doc lines to fix.
  - The concrete deterministic reference-prep steps.
  - The eval metrics (explicit, anaphoric and implicit accuracy, schema failures, alias drift, seconds per chunk).
  - The remaining-debt list in Phase 3.
- **OmniVoice license conflict:** the repo code is Apache and the HF weights are CC-BY-NC. Both are now stated.