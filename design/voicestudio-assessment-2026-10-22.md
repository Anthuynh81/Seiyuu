# VoiceStudio: should Seiyuu use it? (assessment, 2026-10-22, target RTX 3080 Ti 12 GB)

Seiyuu files this is based on: `src/seiyuu/engines/base.py` (the `TTSEngine` template; `synthesize()` always runs `to_canonical()`), `engines/__init__.py` (the lazy string registry), `engines/indextts2_engine.py` and `indextts2_worker.py` (NDJSON-over-stdio worker; `unload()` kills the process), `render/cache.py` (`SegmentKey`, which is frozen), `assemble/pipeline.py` (two-pass loudnorm to -18 LUFS) and `design/tech-refresh-2026-10-22.md`.

## 1. What VoiceStudio is, relative to Seiyuu

**It is a competitor app that also bundles models. It is not a model, and it would be a poor backend.** debpalash/VoiceStudio (AGPL-3.0, with a closed "Pro" build sold alongside) is an Electron desktop app plus a FastAPI server. Its default engine, which it brands "VoiceStudio", is k2-fsa/OmniVoice 0.6B. It also wraps about 15 other engines.

- **As a competitor.** Its Audiobook workspace overlaps Seiyuu's whole surface: EPUB/PDF import, chapters from the TOC, a multi-voice cast, a lexicon, seeds, ACX loudness and chaptered M4B with resume. Its auto-cast (`electron/src/shared/utils/autoCast.js`) is a rule-based `parseScript`. It has no LLM attribution, no reconstruction invariant, no alias resolution and no ASR validation gate. Seiyuu's attribution stack plus its validation gate is still the differentiator.
- **As a source of models.** Most of what it wraps is already on the tech-refresh list: IndexTTS 2.5, VoxCPM2, CosyVoice 3, dots.tts, Pocket TTS and Parakeet. The only notable new one is OmniVoice, and the refresh deliberately skipped it because its weights are CC-BY-NC.
- **As a backend.** It is technically callable. It runs on `127.0.0.1:3900` with OpenAI-compatible `POST /v1/audio/speech`, native `/generate`, `GET /engines` and explicit `POST /model/unload/tts`. But the OpenAI path does not return raw engine audio, as section 3(d) explains.

## 2. Models it has that Seiyuu does not, compared with the tech-refresh picks (12 GB, Windows native)

| Model | Better than the refresh pick? | 12 GB / Windows fit | License | Seed / determinism | Native SR | Validation |
|---|---|---|---|---|---|---|
| **OmniVoice 0.6B** | **No for English audiobooks.** It has no emotion vector (only non-verbal tags such as `[laughter]`), so it loses to IndexTTS-2.5 on emotional narration. It also has no `duration_factor`. Its voice design is a fixed attribute vocabulary, coarser than Qwen3-TTS VoiceDesign's free text. Its real advantages are claimed speed (RTF 0.025 on an H100; unmeasured on a 3080 Ti) and 600+ languages. As a fast tier, Chatterbox-Turbo already gives speed with no new dependency and an MIT license. | Comfortable. VoiceStudio gives a 6 GB floor; upstream publishes no figure. Windows CUDA works. It needs `transformers>=5.3` against our 5.2.0, so **worker isolation is required**. It also cannot share the IndexTTS env, which pins transformers 4.52. | Code Apache-2.0. **Weights CC-BY-NC**, and the audio tokenizer carries extra Boson Higgs Audio 2 and Llama community terms. Not usable for audiobooks that are sold. | Sampling-based iterative decoding. Seed torch per request, as the IndexTTS worker does, and verify byte-stable output before trusting the cache. Unverified. | **24 kHz mono**, the canonical rate, so no resampling | Treat as `requires_validation=True` until retry-rate data says otherwise |
| VoxCPM2 | Already in the refresh (#18 alternative). Nothing new. | ~8 GB fits. The FP8 build needs Ada or newer, so it does not help an Ampere card. | Apache | Design output varies between runs; freeze the result to `reference.wav` | 48 kHz (`to_canonical` resamples) | yes |
| IndexTTS 2.5 | Already the refresh's #1 pick. VoiceStudio's isolated Python 3.11 venv shows it **works on Windows**. | ~6 GB fits; 12 GB of disk | bilibili | Seeded in our worker | 22.05 kHz | yes |
| CosyVoice 3 | Already in the #19 bake-off. Nothing new. | Linux-first | Apache | — | — | yes |
| dots.tts | In the refresh bake-off. VoiceStudio documents **no Windows install path**, so on native Windows it moves to the bottom of #19. | ~9 GB VRAM would fit, but no Windows path | Apache | — | — | yes |
| MOSS-TTS v1.5 (8B) / Nano, Confucius4-TTS | Not evaluated. There are no independent numbers, and their licenses were not checked. The 8B model in bf16 will not fit 12 GB. | 8B ✗; Nano unknown | unverified | unverified | — | — |
| KittenTTS, Supertonic-3, sherpa-onnx TTS | No. Kokoro already covers preset voices, and Pocket TTS (#14) covers CPU cloning. | CPU | various | deterministic | — | no |
| GPT-SoVITS | No. It is fine-tune based and runs as an external server, which does not fit "`reference.wav` → zero-shot". | — | — | — | — | — |
| Breeze-TTS-2 (audio.cpp) | No: the weights are non-commercial. | — | research/NC | — | — | — |
| ASR: WhisperX, Moonshine, FunASR | No. The refresh avoids WhisperX (torch 2.8 pin). Parakeet (#8) beats Moonshine on English WER. | — | — | — | — | — |
| Demucs (vocal isolation) | Could stand in for MelBand-RoFormer in #17 reference curation. It would run in a worker. | small | MIT (upstream) | deterministic | — | — |
| AudioSeal (watermark) | Optional provenance mark only. Chatterbox already applies Perth. | CPU/GPU, small | MIT (upstream, unverified here) | — | — | — |

Net result: OmniVoice is the only new model of interest. It is not better than the refresh picks for Seiyuu's main use (emotional English narration from cloned `reference.wav`), and its license rules it out unless the books are strictly for personal use.

## 3. Integration options, ranked

### Rank 1: (b) Borrow ideas and UX. Recommended.

These are features VoiceStudio has and Seiyuu lacks, in priority order. None of them needs its code, and copying its code would bring AGPL into Seiyuu.

1. **One-click isolated engine installers.** VoiceStudio creates a per-engine sidecar venv from the UI and checks disk space first. Today Seiyuu needs a hand-built cu128 env plus `indextts2_worker_python` / `indextts2_checkpoints_dir` settings. This pairs with refresh #10 (the generic worker base). Effort M, risk Low.
2. **An engine capability catalog.** Each engine would report `max_ref_seconds`, languages, emotion support and tag support, and the UI would show emotion controls only where the engine supports them. This would extend `api/routes/engines` (any new route must be added to `test_api_m6b6.py`). Effort S, risk Low.
3. **Best-window reference selection.** VoiceStudio accepts up to 75 s and auto-picks the best 15 s. This extends refresh #11. It must run **before** consent attestation, and the original upload must be kept, because `reference.wav` is bound to the sha256 used for consent. Effort S–M, risk Low.
4. **An idle-unload timer for the resident TTS model.** Today Seiyuu frees VRAM only when another consumer acquires the GPU. A timer would add a `gpu/manager.py` idle unload. VoiceStudio's 900 s default is a reference point; Seiyuu should pick something shorter. Effort S, risk Low.
5. **A VRAM budget pre-flight**, modelled on `/generate/budget`. Check free VRAM against the engine's needs before loading, and fail loudly instead of OOMing. On 12 GB, a browser or game can quietly hold 1–2 GB. Effort S–M, risk Low.
6. **An ACX compliance preset plus a report** (RMS -23 to -18 dB, peak ≤ -3 dB, noise floor ≤ -60 dB). Seiyuu only does EBU R128 loudnorm to -18 LUFS and has no ACX check. This belongs in `assemble/` with refresh #3. Effort S–M, risk Low.
7. **Chapter cue sheets** alongside the M4B. Fold this into refresh #16 (SRT and Media Overlays). Effort S.
8. **Non-verbal tags** (`[laughter]`, `[sigh]`) emitted from the emotion taxonomy for engines that support them (Chatterbox-Turbo #7, OmniVoice). The tags go through `settings` so they land in `settings_hash`. They must never appear in the normalized text or the whisper reference. Effort M, risk Med.
9. *(Low priority)* An MCP server or OpenAI-compatible `/v1/audio/speech` over Seiyuu's voice library. It may expose only local engines, so no paid call can be reached from it. This is scope creep, so defer it.

Do not borrow dubbing, dictation, Twilio, the marketplace or cloud sync. They are off-mission and widen the privacy surface.

### Rank 2: (c) Adopt OmniVoice through our own subprocess worker. Conditional: optional bake-off arm only.

- **Effort.** S–M after refresh #10 (the generic worker base); M without it. The work is a new uv env (`omnivoice==0.2.1`, cu128 torch), an `omnivoice_worker.py` and an adapter with `clones_from_library=True`, `requires_validation=True` and `native_sample_rate=24000`. `model_version` would be a checkpoint fingerprint, as `weights_fingerprint()` does for IndexTTS.
- **Fit with Seiyuu's rules.**
  - *GPU discipline:* `unload()` kills the process, so the one-heavy-model rule is automatic, as with IndexTTS-2.
  - *Voice truth:* `VoiceClonePrompt` `.pt` files are a disposable conds cache keyed by engine + model version, regenerable from `reference.wav`.
  - *Reference transcript:* pass `ref_text` from a CPU faster-whisper transcription of `reference.wav`, cached alongside it. Otherwise OmniVoice auto-loads Whisper inside the worker, onto the GPU.
  - *Cache key:* the `SegmentKey` format is untouched; only a new engine id and `model_version` value appear.
  - *Paid gates:* not applicable; it is local and free.
- **Risk.** Low technically. **License risk is high:** the weights are CC-BY-NC and the tokenizer has extra terms. It would need an explicit "non-commercial weights" flag in the engine catalog and in SPEC's license table.
- **Constraint.** Emotion is limited to tags; `emotion.py` would return `{}`, as it does for Kokoro. There is no reason to adopt it ahead of IndexTTS-2.5 or Chatterbox-Turbo. Add it to the #19 bake-off only if multilingual books or raw speed become goals.

### Rank 3: (a) Ignore it as a runtime dependency. Acceptable default.

- **Effort.** Zero.
- **Risk.** Very low. Its other models are already in the refresh plan, so ignoring it as software loses nothing material. The only cost is missing the UX ideas in (b).

### Rank 4: (d) Drive the VoiceStudio app as an HTTP backend. Not recommended.

- **Effort.** M. The adapter would be stdlib HTTP with `uses_gpu=True`, `unload()` calling `POST /model/unload/tts`, and `warm()` plus profile lifecycle management. Tests would run against a fake server.
- **Risk.** High:
  - **GPU discipline would be cooperative only.**
    - Seiyuu's `GpuResourceManager` cannot arbitrate VoiceStudio's own UI jobs, its auto-sized GPU workers (1 per 5 GB free, about 2 on 12 GB) or its WhisperX ASR.
    - After an unload, its CUDA context probably stays resident (unmeasured).
    - The idle timeout defaults to 900 s. You would have to set `OMNIVOICE_IDLE_TIMEOUT_S` low, keep `OMNIVOICE_SINGLE_ENGINE_RESIDENT=1`, pin workers to 1, and still check `nvidia-smi`.
  - **Output is not raw engine audio.** `_run_tts` trims silence, runs a mastering compressor, normalizes to -2 dBFS and applies an AudioSeal watermark if a preference is set. That stacks with Seiyuu's loudnorm. The flags are global preferences, not per-request, so the adapter would have to read them at boot and refuse to run, or fold them into `model_version`. Native-rate WAV is fine, because `to_canonical()` already resamples.
  - **The model version is a moving target, which can poison the cache.** The project pushes daily and the Electron app auto-updates, so engine code, weights or post-processing can change without notice. `model_version` would have to fingerprint the app version, the backend engine and model, and the post-processing preferences. Even then it can only be as honest as what `/engines` and `/v1/models` report.
  - **`reference.wav` truth.** The OpenAI path takes a `voice` profile ID, so profiles become disposable handles keyed by the reference's sha256, recreated when not found, as with ElevenLabs IVC. Consent must be re-verified before every upload, and `cloud_sync` and the marketplace must be confirmed off so cloned voices never leave the machine.
  - **Supply chain and privacy.** The installer is unsigned, the documented install is `irm | iex`, there is a CLA to a company and a closed Pro build, PostHog analytics are opt-in (hard-disable with `OMNIVOICE_ANALYTICS_DISABLED=1`), and it ships network routers (cloud sync, marketplace, Twilio).
  - **Licensing.** Calling over HTTP does not bring AGPL into Seiyuu. The default engine's weights are still CC-BY-NC.
  - **Other limits.** Input is capped at 4096 characters per request (Seiyuu's segments are well under that, but add a guard). Seed is an extension field and its determinism is unverified.
- **Constraints recap.** `SegmentKey` stays unchanged. No paid gate is involved, since the cost is 0. The one-model rule cannot be guaranteed from Seiyuu's side.

## 4. Recommendation

Treat VoiceStudio as a competitor to study, not as a dependency or backend. Its audiobook path has no LLM attribution or validation gate, and the models it wraps are already on the tech-refresh roadmap or ruled out by OmniVoice's CC-BY-NC weights. Borrow the UX ideas in 3(b): one-click worker-env installs, a capability catalog, best-window reference prep, an idle unload, a VRAM pre-flight and an ACX check. They slot into refresh items #3, #10, #11 and #16. Add OmniVoice as an opt-in bake-off worker only if non-commercial use and non-English books become goals.


---

# Appendix: verified facts


**Verdict:** the identification is correct, and most of the report holds up. There are several factual errors, a few important omissions (open-core Pro build, opt-in PostHog analytics, post-processing on the API path, an extra licence on the OmniVoice tokenizer), and one wrong technical reason, about the torch requirement. Corrected account below. Labels: **[V]** verified from a primary source today, **[C]** corrected, **[U]** unverifiable or secondary-only.

## 1. VoiceStudio: debpalash/VoiceStudio (the one the user means)

- **Existence and identity [V].** https://github.com/debpalash/VoiceStudio. The GitHub description reads "VoiceStudio is the open-source, fully-local ElevenLabs alternative — voice cloning, voice design, video dubbing, dictation, transcription & audiobook creation in 646 languages." The homepage is voicestudio.sh. The old URL `debpalash/OmniVoice-Studio` returns a 301 to the same repo ID. The repo was created 2026-04-09 and is not archived.
- **Activity [V].** 81,305 stars, 8,983 forks, 308 open issues, last push 2026-10-22T14:20Z. Contributor counts: debpalash 3,416 commits, velixio 433, then a long tail (the anon-inclusive contributor list runs to about 100 pages at 1 per page). The "+12.8k stars per week" and "gitnova says organic" claims rest only on aggregators **[U]**. Aggregator articles date back to **2026-09-04** (aitoolly), so it was getting coverage before "mid-September" **[C, minor]**.
- **What it is [V].** An Electron desktop app plus a Python FastAPI backend that wraps existing models. It is not a new model. The README says: "Start with **VoiceStudio** (default, powered by k2-fsa/OmniVoice)."
- **Releases [C].**
  - The GitHub releases REST API returns an **empty list** today (anonymous), so the report's "per the GitHub releases API" is not reproducible.
  - The tags API and the releases web page both show v0.4.0 through **v0.5.6**. Tag commit dates: v0.5.6 on 2026-09-23 (`3915a62c`), v0.5.5 on 2026-09-21, v0.5.4 on 2026-09-20. The report gave 09-22 and 09-21 for v0.5.5 and v0.5.4; publish dates could not be confirmed.
  - **Tauri versus Electron is wrong in the report.** v0.5.3 (17 Sep) already shipped Electron as "the primary desktop app on Windows, macOS, and Linux". It was the *final* Tauri release, with legacy Tauri assets alongside the Electron ones. v0.5.4 onward is Electron only. The README now says "Electron is the only desktop app."
  - The v0.5.6 notes confirm: OpenAI API fixes (`GET /v1/models`, `/v1/audio/translations`, real mp3/aac/opus output, `instructions` forwarded to the engine), the MCP path changing to `/mcp/`, "Long chapters finish on 8 GB GPUs", and Twilio phone answering (off by default).
- **Licence [V, with additions].**
  - The application is AGPL-3.0-only (LICENSE plus LICENSE-NOTICE.md). Commercial use under AGPL is free.
  - **Open-core [missed]:** voicestudio.sh/pro sells a "**closed-source Pro build**: a Commercial License for client work, more devices and early access. $99 a year, or $299 once."
  - **CLA [missed detail]:** contributions are licensed to **Yupcha Softwares Private Limited** under `.github/CLA-1.0.md`.
  - The bundled `omnivoice/` package is Apache-2.0.
  - **Model weights:**
    - OmniVoice weights are CC-BY-NC.
    - **[missed]** OmniVoice's `audio_tokenizer/LICENSE` carries separate **Boson Higgs Audio 2 and Meta Llama community terms**, per VoiceStudio's own LICENSE-NOTICE.
    - IndexTTS 2.5 uses the bilibili Model Use License. A separate licence is needed above 100M MAU or RMB 1B revenue, and there are also derivative, attribution and prohibited-use obligations.
    - Breeze-TTS-2 (audio.cpp) is research/non-commercial.
    - dots.tts is Apache-2.0 for code and checkpoints.
    - VoiceStudio states that "No commercial clearance is asserted."
  - The report's AGPL analysis is reasonable as an opinion, not legal advice: HTTP use does not propagate AGPL, but copying code does.
- **Engines [V].**
  - Speech generation, per the feature catalog: VoiceStudio/OmniVoice, omnivoice-subprocess, omnivoice-gguf, CosyVoice 3, KittenTTS, MLX-Audio, VoxCPM2, MOSS-TTS-Nano, MOSS-TTS-v1.5, gpt-sovits, sherpa-onnx, IndexTTS 2.5, supertonic3, dots.tts, Confucius4-TTS, pockettts, audiocpp.
  - Transcription: WhisperX (default), Faster-Whisper, MLX Whisper, PyTorch Whisper, Parakeet TDT (plus v3 on MLX), Moonshine, FunASR, sherpa-onnx, and an "OpenAI-compatible configured server".
  - **No Chatterbox or Qwen3-TTS anywhere in the repo tree [V].** Kokoro appears only via MLX-Audio, which is Apple-only.
  - dots.tts: "no Windows install path" [V]. IndexTTS 2.5 runs in an isolated Python 3.11 venv; the installer checks for **12 GB free disk** [V].
- **Audiobooks [V].**
  - Auto-cast is `electron/src/shared/utils/autoCast.js`. It calls `parseScript(text)` and cycles voice profiles, with **no LLM, alias resolution or ASR validation gate**. This confirms Seiyuu's differentiator.
  - EPUB import exists (tests `test_longform_import_epub_matter.py` and `test_epub_member_uris.py`).
  - Note that `backend/services/audiobook.py` still says "epub/pdf ingestion … deferred", so docs and code are partly out of sync.
  - The M4B ffmpeg mux, a chapter cache, and ACX two-pass mastering exist (`loudness.py`, spec 28).
- **Windows [V].**
  - Installer: `VoiceStudio-Electron-<ver>-win-x64.exe`, per-user, unsigned or ad-hoc signed (the release notes warn about trust prompts), with `SHA256SUMS.txt` published.
  - `irm https://voicestudio.sh/install | iex` downloads, verifies and opens the wizard. It is pipe-to-shell, so review it before running.
  - About 10 GB disk. The runtime ships **PyTorch 2.8** (CUDA or CPU). ROCm is ignored on Windows. ARM64 is experimental (x64 Python under emulation, CPU only). Intel Mac is UI-only.
  - torch.compile is probe-based and is skipped on typical Windows installs because there is no Triton wheel.
- **VRAM [V].**
  - OmniVoice has a "Recommended VRAM floor: **6 GB**", the only measured floor.
  - GPU workers: 1 per 5 GB free (max 4), and "Do not raise this on ≤10 GB cards". On a 12 GB card the result depends on *free* VRAM, so "about 2 workers" is plausible but not guaranteed.
  - `OMNIVOICE_SINGLE_ENGINE_RESIDENT=1` is the default.
  - **[missed]** The default `OMNIVOICE_IDLE_TIMEOUT_S` is **900 s**, so the TTS model stays resident for 15 min. Sidecars (IndexTTS and others) idle out after 300 s.
  - The benchmarks table: "No verified rows yet."
  - dots.tts needs about 9 GB VRAM; "12–16 GB … realistic target".
- **API [V, with additions].**
  - Default port `127.0.0.1:3900` (`OMNIVOICE_PORT`).
  - `POST /v1/audio/speech`:
    - `model` defaults to `omnivoice`; the listed IDs are omnivoice, voxcpm2, cosyvoice, mlx-audio, kittentts and moss-tts-nano, and OpenAI model names alias to the active engine.
    - `voice` takes a profile ID or a unique profile name.
    - `seed` is an extension field.
    - `input` is capped at **4096 characters**.
    - The other routes: `/v1/models`, `/v1/audio/transcriptions`, `/v1/audio/translations` and `/v1/audio/voices`. The native routes are `POST /generate`, `GET /generate/budget` and `GET /engines` (plus `/engines/tts`, `/engines/asr` and `/engines/llm`). Streaming ASR is a WebSocket at `/v1/audio/transcriptions/stream`. MCP is served at `/mcp/` and via stdio with `python -m backend.mcp_shim`.
  - Loopback calls need no key; remote calls use a bearer key `OMNIVOICE_API_KEY`.
  - **[missed, matters for Seiyuu]** `_run_tts` on the OpenAI path **trims silence, applies a mastering compressor chain (unless the engine opts out), normalises to −2 dBFS, and applies a pref-gated AudioSeal watermark**. Output is therefore not raw engine audio. `wav` comes out at the engine's native rate (for example VoxCPM2 at 48 kHz); only `pcm` is resampled to 24 kHz.
  - **[missed, useful]** Explicit unload endpoints exist: `POST /model/unload/tts` and `POST /system/flush-memory?unload_model=true`. These would let Seiyuu enforce its one-model GPU rule.
  - Docker: `ghcr.io/debpalash/voicestudio` and Docker Hub `palashdeb/omnivoice-studio`, linux/amd64 only.
- **Telemetry [missed].** `backend/core/analytics.py` contains **PostHog** analytics with a committed publishable key that sends to `us.i.posthog.com`. It is **opt-in** (`analytics_enabled` defaults to False), uses an allowlist of event properties, has no exception autocapture, and can be hard-disabled with `OMNIVOICE_ANALYTICS_DISABLED=1`. The README also embeds a supermetry view-counter pixel.
- **Other surface area [missed].** Routers for `cloud_sync` (with device management), `marketplace` (publish and install voice profiles), `community`, `telephony_twilio` and `calls`. They are optional, but they widen the attack and privacy surface well beyond "fully local".
- **Red-flag summary.** Not vaporware and not abandoned. The flags are: unsigned installers, `irm|iex` install, open-core closed Pro build, CLA to a company, opt-in telemetry, network features, and non-commercial default weights.

## 2. OmniVoice: k2-fsa/OmniVoice (the model under VoiceStudio)

- **Repo [V].** 15,580 stars, 2,289 forks, created 2026-03-31, last push 2026-10-19.
- **PyPI [V, the report left this unverified].** `omnivoice` **0.2.1** (2026-07-16).
- **Licence [V].** Code Apache-2.0. The HF card says "The pre-trained model is licensed under the CC-BY-NC due to constraints from its training data (e.g., Emilia)." Add the separate audio-tokenizer terms noted in section 1.
- **Base model [missed].** `Qwen/Qwen3-0.6B`, per the HF card.
- **Claims [V, upstream].**
  - "RTF as low as 0.025", measured on an H100 with fp16.
  - Use a 3–10 s reference; `ref_text` is optional because Whisper auto-transcribes.
  - CUDA, MPS, Intel XPU and CPU are supported, and FlashInfer is optional.
- **[C] Torch requirement.** The report says OmniVoice needs torch 2.8+cu128. That is only the README's *example* install line. `pyproject.toml` requires `torch>=2.4` and **`transformers>=5.3.0`**. Seiyuu's lock has torch 2.6.0+cu126 (pinned by chatterbox-tts 0.1.7) and **transformers 5.2.0**. The real conflict is **transformers**, not torch. Subprocess isolation is still the prudent route, but for that reason.
- **VRAM [U for upstream].** The 6 GB floor comes from VoiceStudio's docs. Upstream publishes no VRAM figure.

## 3. Voicebox: jamiepine/voicebox (the closest lookalike)

- **Repo [V].** MIT, 58,233 stars, 7,307 forks, **522** open issues (the report said 510), created 2026-01-25, last push 2026-10-07. Latest tag and release **v0.5.0** (2026-04-25), with nothing tagged since.
- **Engines [V].** Qwen3-TTS (0.6B and 1.7B), Qwen CustomVoice, Qwen VoiceDesign, LuxTTS, Chatterbox Multilingual, Chatterbox Turbo, HumeAI TADA (1B and 3B), and Kokoro. Whisper for STT, plus a bundled local LLM. Supports 23 languages.
- **Windows [V].** MSI and NSIS installers (Tauri updater `.sig` files). CUDA archives `cuda-libs-cu128-v1.tar.gz` (about 2.1 GB) and `voicebox-server-cuda.tar.gz` (about 1.0 GB). DirectML is listed for any Windows GPU.
- **API [V, the report lacked detail].** `127.0.0.1:17493` with `/generate`, `/speak`, `/transcribe` and `/profiles`, MCP at `/mcp`, and docs at `/docs`. **No OpenAI-compatible speech route is advertised.**
- **Maintenance.** Directed day to day by a "Capy Captain" AI-agent thread, with Jamie merging. The Capy promotion is prominent in the README.
- No EPUB to M4B pipeline is documented.

## 4. Voice Studio: msrbuilds/voice-studio (minor, same name)

- **Repo [V].** MIT, 218 stars, 47 forks, created 2026-06-17, last push 2026-07-31. Dormant for about 12 weeks.
- **[C] Windows** is explicitly listed ("OS: Windows 10/11, Linux, macOS"); the report said it was not verified.
- **[C] VibeVoice-1.5B "~5.4 GB" is disk, not VRAM.** The README gives VRAM as about 3 GB for VibeVoice fp16 and up to about 8 GB for VoxCPM2.
- Serves the UI and API on `:8880`.
- **Red flag [new]:** the README labels OmniVoice "Apache-2.0" without mentioning the CC-BY-NC weights.

## 5. Candidates the report missed (audiobook-relevant, closest to Seiyuu)

| Repo | Licence | Stars | Last push | Note |
|---|---|---|---|---|
| [DrewThomasson/ebook2audiobook](https://github.com/DrewThomasson/ebook2audiobook) | Apache-2.0 | 20,390 | 2026-10-06 | Ebook to audiobook, "50+ TTS engines with voice cloning"; the most direct open-source audiobook competitor |
| [prakharsr/audiobook-creator](https://github.com/prakharsr/audiobook-creator) | GitHub reports NOASSERTION (description says GPL-3.0) | 531 | 2025-11-17 (stale) | Does LLM character attribution with Kokoro/Orpheus, the closest feature match to Seiyuu |
| [erew123/alltalk_tts](https://github.com/erew123/alltalk_tts) | AGPL-3.0 | 2,449 | 2026-01-09 | Older Coqui-based TTS server with JSON API |

Engine/model detail on these was not re-verified **[U]**.

## Corrected takeaways for Seiyuu (12 GB RTX 3080 Ti)

1. VoiceStudio is an app built around OmniVoice. It adds no attribution quality, and its auto-cast is rule-based.
2. Using it as an HTTP engine is feasible, with these conditions:
   - Call `/model/unload/tts` (or set `OMNIVOICE_IDLE_TIMEOUT_S` low) to keep the one-heavy-model GPU discipline. The 900 s default conflicts with it.
   - Disable mastering and watermarking, or accept them.
   - Resample to 24 kHz mono yourself, since `wav` comes back at the engine's native rate.
   - Keep each request under 4096 characters.
3. Using OmniVoice directly needs isolation because of **transformers>=5.3** against Seiyuu's 5.2.0, not because of torch. Its weights are non-commercial, and its tokenizer has extra licence terms.
4. 12 GB comfortably fits OmniVoice, IndexTTS 2.5 or VoxCPM2 one at a time. dots.tts would fit too, but it is not supported on Windows. No published RTF or VRAM benchmarks exist for VoiceStudio.

## Evidence
- https://api.github.com/repos/debpalash/VoiceStudio · /tags · /contributors · https://github.com/debpalash/VoiceStudio/releases/tag/v0.5.6 · /releases/tag/v0.5.3
- https://raw.githubusercontent.com/debpalash/VoiceStudio/main/README.md · LICENSE-NOTICE.md · docs/feature-catalog.md · docs/install/windows.md · docs/install/docker.md · docs/performance.md · docs/benchmarks.md · docs/speech-platform.md · docs/engines/omnivoice.md · docs/engines/indextts.md · docs/engines/dots-tts.md · docs/engines/audio-cpp.md
- https://github.com/debpalash/VoiceStudio/blob/main/backend/api/routers/openai_compat.py · backend/core/analytics.py · backend/services/audiobook.py · electron/src/shared/utils/autoCast.js
- https://voicestudio.sh/pro
- https://api.github.com/repos/k2-fsa/OmniVoice · https://github.com/k2-fsa/OmniVoice/blob/master/pyproject.toml · https://huggingface.co/k2-fsa/OmniVoice · https://pypi.org/project/omnivoice/
- https://api.github.com/repos/jamiepine/voicebox · https://api.github.com/repos/jamiepine/voicebox/releases/tags/v0.5.0 · https://github.com/jamiepine/voicebox/blob/main/README.md
- https://api.github.com/repos/msrbuilds/voice-studio · https://github.com/msrbuilds/voice-studio/blob/main/README.md
- https://api.github.com/repos/DrewThomasson/ebook2audiobook · https://api.github.com/repos/prakharsr/audiobook-creator · https://api.github.com/repos/erew123/alltalk_tts
- Secondary only: https://alternativeto.net/software/voicestudio/ · https://aitoolly.com/en/ai-news/article/2026-09-04-voicestudio-the-open-source-and-localized-powerhouse-challenging-elevenlabs-in-ai-voice-synthesis
- Seiyuu lock check: `F:\Projects\Seiyuu\uv.lock` (torch 2.6.0+cu126, transformers 5.2.0, chatterbox-tts 0.1.7)

Nothing was installed. Fetched text was saved only under `C:\Users\Cyber\AppData\Local\Temp\claude\F--Projects-Seiyuu\e7979d64-7136-46d9-bfec-352f7910845e\scratchpad\vs`, and the repo was not modified.