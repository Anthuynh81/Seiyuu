# TTS engine options — November 2026

*Decision brief for chapter 34 of "Got Dropped into a Ghost Story". Researched 2026-11-07. "Unverified" means nobody has measured or confirmed it yet; treat it as a lead.*

**Glossary.** **Clone**: copy a voice from a short audio clip. **Voice design**: create a new voice from a written description such as "gruff old janitor, slow and tired". **Real time**: 3x real time means a 9-minute chapter renders in 3 minutes. **VRAM**: graphics-card memory; yours has 12 GB. **Worker**: a separate program with its own Python setup, like the one IndexTTS already uses. **Hallucination**: the model says words that are not in the text.

---

## 1. Bottom line

1. **Re-test IndexTTS-2.5 with its pace setting first.** The `duration_factor` setting is already built on branch `indextts25-engine` (commit `cc7bdd1`) but has not been run yet. On chapter 34, 2.5 ran 8–13% longer than IndexTTS-2 for almost every voice; the narrator was 393 s vs 362 s. A setting of 0.90 should bring the chapter back to about 9.4 min. Cost: about 12 min of GPU, no downloads.
2. **Then A/B Qwen3-TTS 1.7B.** Design each chapter-34 voice from a text description, then render with Qwen's clone model. Why Qwen:
   - It tends to speak fast rather than slow.
   - It had the best naturalness score of the cloning engines in the one independent test.
   - It needs about 5 GB of VRAM and is Apache-2.0 (commercial use allowed).
3. **Run the same designed voices through IndexTTS too.** That shows whether the "robotic" feel comes from the engine or from cloning Kokoro clips.
4. **No engine will fix the dash/ellipsis pauses or the bystanders who share one voice.** Those are Seiyuu fixes (section 5).
5. **The other candidates:**
   - F5-TTS: non-commercial license and no voice design.
   - Pocket TTS: about Kokoro quality.
   - Neither is worth an A/B slot. VoxCPM2 is the backup if Qwen disappoints.

---

## 2. Comparison table

Speeds for chapter 34 (9.4 min of audio) are measured where marked; the rest are estimates for your 3080 Ti on Windows.

| Engine | License | Fits 12 GB? | Time for ch. 34 (9.4 min) | Windows | Clone / design / built-in voices | Pace control | Pauses at … and — | Known robotic / hallucination issues | Integration effort |
|---|---|---|---|---|---|---|---|---|---|
| **IndexTTS-2.5** (current, for reference) | bilibili Model Use License (custom; [card](https://huggingface.co/IndexTeam/IndexTTS-2.5)) | Yes, ~6 GB | **12.3 min measured** (incl. load + retries) | Works (our worker) | Clone only | `duration_factor` 0.5–2.0, not yet tried | Long text is split at 120 tokens with 200 ms gaps; "prosody is not modelled across a segment boundary" ([card](https://huggingface.co/IndexTeam/IndexTTS-2.5)) | 4 lines flagged on ch. 34 vs 2 for v2 | **S**: already wired; the pace setting is on a branch |
| **Qwen3-TTS 1.7B** (VoiceDesign + Base) | Apache-2.0 | Yes, ~4.6–4.9 GB ([tts-bench](https://5uck1ess.github.io/tts-bench/speed.html)) | ~3–5 min with [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (est.); 12–30 min with the stock package (est.) | Not official, but community-proven. Use SDPA instead of FlashAttention. [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/26254) ships Windows CUDA builds (clone only) | Clone + **voice design** + 9 presets (only 2 English, both male) | No number; written instructions only. Leans **fast**, not slow ([#290](https://github.com/QwenLM/Qwen3-TTS/issues/290)) | Ellipses and dashes make "absolutely no difference" ([#75](https://github.com/QwenLM/Qwen3-TTS/discussions/75)) | Faint ~40 ms echo in all output ([#386](https://github.com/QwenLM/Qwen3-TTS/issues/386), open). Wrong voice on ~20% of lines with some references ([#80](https://github.com/QwenLM/Qwen3-TTS/issues/80)). Adds laughs/fillers ([#16](https://github.com/QwenLM/Qwen3-TTS/issues/16)) | **M**: copy of the IndexTTS worker; output is already 24 kHz. Upstream is dormant; fixes live in community forks |
| **VoxCPM2** | Apache-2.0 | Tight: ~8 GB official; a 12 GB card crashed in the demo app ([#300](https://github.com/OpenBMB/VoxCPM/issues/300)) | ~9–14 min (est.; a 3060 on Windows ran ~1.45x slower than real time) | No Windows docs. Compile needs Triton + a C compiler ([#419](https://github.com/OpenBMB/VoxCPM/issues/419)) | Clone + **voice design**; no built-in voices | No number ([#56](https://github.com/OpenBMB/VoxCPM/issues/56) closed with no feature); writes speed up rather than slow down | "Ellipsis can produce hesitation"; "... —" gives only a subtle pause. "(pause)" gets **read aloud** ([#276](https://github.com/OpenBMB/VoxCPM/issues/276)) | Crackle at high settings ([#404](https://github.com/OpenBMB/VoxCPM/issues/404)); extra words on 1-word lines, 7–33% ([#357](https://github.com/OpenBMB/VoxCPM/issues/357)) | **M–L**: heavy, loosely pinned dependencies; 48 kHz output; Windows compile problems |
| **F5-TTS v1** | Code MIT, **weights NON-COMMERCIAL** (CC-BY-NC-4.0) | Yes, <1 GB | ~3–4 min (est.) | Works with torch <2.9. Pin 1.1.22, because the main branch crashes on our torch 2.6 | Clone only | Real knobs: `speed` and `fix_duration` (exact seconds) | Pause requests still open ([#1034](https://github.com/SWivid/F5-TTS/issues/1034)) | Short lines are stretched to 3.3x on purpose. "!" lines go unstable / "a bit robotic" ([#1263](https://github.com/SWivid/F5-TTS/issues/1263)). Worst word-error rate in tts-bench | **M**: worker + reference clips ≤12 s with exact transcripts + duration math |
| **Pocket TTS 4.0.1** (Kyutai) | Code MIT, weights CC-BY-4.0 (cloning weights behind an auto-approved sign-up) | Yes, tiny; runs on **CPU** | ~2–5 min on CPU; ~1 min on GPU (est.) | CPU install works; a Windows binary also exists | Clone (no transcript needed) + 21 English presets + hundreds of donated human voices; **no design** | None ([#15](https://github.com/kyutai-labs/pocket-tts/issues/15)) | No pause tags ([#6](https://github.com/kyutai-labs/pocket-tts/issues/6)); adds ~240 ms tail per line | Repeated sentence endings ([#347](https://github.com/kyutai-labs/pocket-tts/issues/347), improved). "Prosody worse than Kokoro" with presets ([#115](https://github.com/kyutai-labs/pocket-tts/issues/115)) | **S–M**: CPU worker; needs its own setup because of a numpy clash |
| **Phonon 31M** (scout; built on Pocket) | Apache-2.0 (the 40M version is non-commercial) | Yes, 65 MB; CPU | ~1 min (est. from the vendor's 11.6x on one laptop core) | Probably (`pip install ptts`, CUDA/Vulkan); Windows not named | ~100-voice catalog ([card](https://huggingface.co/gradium/phonon-31m-en)); cloning and design are paid cloud only | Global speed setting (unverified) | No pause tags | Arena score 1112 vs Kokoro 1061 ([Artificial Analysis](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights)) | **S–M**: no torch conflict on the `ptts` path |
| **Ming-3.0-tts-0.5B** (scout) | MIT | Yes, ~3 GB of weights | Unknown (only tested on datacenter GPUs) | Tested on Linux only; pins an old transformers | Clone + design + only 5 English built-in voices | Explicit speed setting, but modes can't be combined | No pause token | Card admits its design "does not yet match the strongest evaluated system" ([card](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)) | **L**: new, small community, Linux-first |
| **MOSS-VoiceGenerator 1.7B** (scout; design tool only) | Apache-2.0 | Probably (~4–6 GB, unverified) | n/a (only makes ~15 s clips) | Undocumented; pins torch 2.9.1 | **Design only**; its clips feed any clone engine | In the description | n/a | Unknown | **S–M** as a one-off clip-making script |

**Scout corrections:**
- **OmniVoice weights are non-commercial** (CC-BY-NC, per its [HF card](https://huggingface.co/k2-fsa/OmniVoice)), not Apache as the scout said, so it is left out.
- **Breeze TTS 2** and **Higgs v3** are non-commercial and Linux-first.
- **Fish Drama 3** needs 48 GB.

---

## 3. Engine by engine

### IndexTTS-2.5 with a pace setting (cheapest test)
- **Good at:** it already works in Seiyuu. The slowdown looks uniform: 2.5 was 8–13% longer than v2 for the narrator and most named characters (from the ch. 34 manifests in `output/ab/.../ch034/`). That suggests one setting can fix it.
- **Would bother you:**
  - Pace partly follows the reference clip, and today's clips are Kokoro audio.
  - Long narration paragraphs get chopped at 120 tokens with a 200 ms gap and no carried-over intonation. That is a plausible cause of "robotic in some sections" (my hypothesis, unverified).
  - Its license is a custom bilibili one, not Apache.
- **Unknowns:** whether speeding it up makes it sound rushed. Whether the robotic patches line up with the long paragraphs.

### Qwen3-TTS (recommended main A/B)
- **Good at:**
  - Voice design from a description, with an officially documented "design once, save the clip, clone it for every line" workflow ([README](https://github.com/QwenLM/Qwen3-TTS)). That makes a distinct voice for every bystander cheap.
  - It speaks fast, not slow ([#290](https://github.com/QwenLM/Qwen3-TTS/issues/290); [PR #362](https://github.com/QwenLM/Qwen3-TTS/pull/362) measured drift toward ~195 words/min).
  - Best clone naturalness score in [tts-bench](https://5uck1ess.github.io/tts-bench/scores.html): 4.23 vs IndexTTS-2 3.71 and IndexTTS-2.5 3.62. That bench is small (5 prompts).
  - VoxCPM's own listening test rated Qwen's designed voices *more* natural than VoxCPM's: 4.61 vs 4.48 ([arXiv 2606.06928](https://arxiv.org/abs/2606.06928)).
- **Would bother you:**
  - It ignores ellipses and dashes.
  - It may rush: "periods sound like commas" in long passages. Short per-line segments plus its non-streaming mode should help.
  - A reported faint echo could read as "synthetic" ([#386](https://github.com/QwenLM/Qwen3-TTS/issues/386)).
  - It sometimes speaks a dialogue line in the wrong voice, especially lines starting with a quote mark ([#80](https://github.com/QwenLM/Qwen3-TTS/issues/80)). Our whisper check would not catch that.
  - The official package is dormant: issues get auto-closed.
- **Unknowns:**
  - Real speed on your card. No 3080 Ti or Windows-with-fast-path numbers exist; a 4060 on Windows reached 1.83x real time.
  - Whether the echo is audible on your book.
  - Whether a cloned designed voice keeps its range.
  - Whether runs repeat exactly with the same seed. The cache makes that matter less.

### VoxCPM2 (backup)
- **Good at:**
  - Voice design plus cloning in one model.
  - Apache-2.0, and its maintainers say reusing designed clips is fine ([#407](https://github.com/OpenBMB/VoxCPM/issues/407)).
  - Leans brisk.
  - Naturalness ties IndexTTS-2 in its own test (4.78 vs 4.78).
- **Would bother you:**
  - ~8 GB is tight on 12 GB; one 12 GB Windows user crashed.
  - Likely about as slow as IndexTTS-2 on Windows unless compile works.
  - No pause control, and "(pause)" gets read aloud.
  - Extra words appear on one-word lines.
  - Its "Ultimate" clone mode copies the reference's rhythm, so cloning Kokoro clips there would copy Kokoro.
- **Unknowns:** speed on your card; whether the crackle report ([#404](https://github.com/OpenBMB/VoxCPM/issues/404)) affects English; how quote marks change short-line errors.

### F5-TTS (not recommended)
- **Good at:**
  - Real pace knobs; you can set exact seconds.
  - Under 1 GB.
  - Fast.
- **Would bother you:**
  - **Non-commercial weights.**
  - No voice design and no built-in voices.
  - Flat delivery, and it copies the flatness of Kokoro references.
  - Short lines are deliberately stretched 3.3x unless Seiyuu overrides the length ([commit 45868ab](https://github.com/SWivid/F5-TTS/pull/1332)).
  - Lines ending in "!" go "a bit robotic".
  - Your ~15 s clips would be cut to 12 s while keeping the full transcript, which garbles timing.
  - No new model since March 2025.
- **Unknowns:** whether correct duration math gets an exact 9.4-min match.

### Pocket TTS (not for this A/B)
- **Good at:**
  - Runs on CPU, so it can render while Ollama holds the GPU.
  - Hundreds of real human donor voices you can clone (CC0/CC-BY).
  - Brisk pace (~5 words/s).
- **Would bother you:**
  - About Kokoro-level naturalness; one open issue says worse with presets ([#115](https://github.com/kyutai-labs/pocket-tts/issues/115)).
  - No design.
  - No pause control.
  - The default model changed several times in 2026, so the model must be pinned.
- **Unknowns:** whether the current model still repeats sentence endings, and how its longer between-sentence pauses feel. Its successor **Phonon 31M** (Apache, ~100 voices, arena score above Kokoro) is the better pick if you want a Kokoro replacement. Phonon's evidence is vendor/scout-level and unverified on Windows.

### Scout finds, briefly
- **MOSS-VoiceGenerator:** Apache-2.0, designs voices from free text, and the card names audiobooks as a target use. Worth a look later as a second design tool if Qwen's designed voices don't fit your characters.
- **Ming-3.0-tts-0.5B:** strong leaderboard results, MIT, but Linux-only testing and only 5 English built-in voices. Revisit in a few months.

---

## 4. Recommended next A/B (chapter 34 again)

**Why designed voices instead of Kokoro clips:**
- A clone copies its reference's delivery, pacing and acoustics. Qwen's maintainer says this outright ([#16](https://github.com/QwenLM/Qwen3-TTS/issues/16)), and Kyutai says the same of Pocket.
- So every IndexTTS voice today inherits Kokoro's synthetic rhythm.
- Designed clips remove that, and they also give each character a voice made to fit them.

**Step 0: IndexTTS-2.5 at pace 0.90**
- No download; ~12 min GPU; already built.
- Command: `uv run python scripts/ab_indextts.py got-dropped-into-a-ghost-story-still-33b6f285 --chapter 34 --consent --arms indextts25 --pace 0.9`
- This adds an `indextts25-d0.90` row to `output/ab/.../ch034/ab.md`.
- Listen for: length near 9.4 min; rushed vs natural; whether the robotic patches remain.
- If they remain at normal speed, slowness was not the cause, which points at the references or the internal chunking.

**Step 1: design the 9 chapter-34 voices with Qwen3-TTS 1.7B-VoiceDesign**
- I draft one description per voice from the book; you edit them.
- For each voice: 3 takes, you pick one, and it is saved as a ~10–15 s clip.
- The voices are narrator, Eun Ha-je, Kwak Jae-kang, the new employee, Lee Ja-heon, Park Min-seong, Kim So-leum, `character_auto` and `staff_auto`.
- The clips are staged inside the A/B folder, exactly like today's Kokoro clips. Nothing enters your voice library.
- GPU: ~5–25 min, depending on whether the fast path works on Windows.

**Step 2: render chapter 34 with the designed clips**
- **Arm A:** Qwen3-TTS 1.7B-Base clone with faster-qwen3-tts, non-streaming mode, one Seiyuu line per call. GPU ~6–12 min (est.), or 12–30 min if only the stock package works.
- **Arm B:** IndexTTS-2.5 at the best pace from step 0, using the **same** designed clips. ~12 min. This separates "engine" from "reference".
- **Optional Arm C:** Phonon 31M on CPU with catalog voices, as a Kokoro replacement test. ~1–3 min, no GPU, ~65 MB. Needs a small worker first.
- The existing Kokoro / IndexTTS-2 / IndexTTS-2.5 MP3s stay as the baseline.

**What to listen for (write down timestamps):**
1. Total length vs Kokoro's 9.4 min, and whether speech feels rushed.
2. Pauses at sentence ends vs at "…" and "—" (expect them to still be off; see section 5).
3. Robotic patches. Note whether they fall on long narration paragraphs.
4. A faint doubled-syllable echo (Qwen).
5. A dialogue line spoken in the wrong voice (Qwen; the speech check won't flag it).
6. Unrequested laughs or "OK"s.
7. Does each voice fit its character?
8. Number of lines flagged by the speech check.

**Totals:**
- **GPU time:** roughly 30–60 min across steps 0–2, in short sessions. Ollama must be stopped.
- **Downloads:** about **11 GB**, estimated from parameter counts; I'll confirm on Hugging Face before pulling.
  - Qwen3-TTS 1.7B-VoiceDesign ~3.9 GB
  - 1.7B-Base ~3.9 GB
  - Tokenizer ~0.7 GB
  - A new worker setup with CUDA torch ~3 GB, partly shared with the IndexTTS setup's cache

**Needs your OK:**
1. Running step 0 (GPU time only).
2. The ~11 GB download and a new, separate Python setup for Qwen. The main setup is not touched; I'll still re-run the torch CUDA check afterwards.
3. ~30–60 min of GPU time for steps 1–2.
4. **Later, only if Qwen wins:** a new voice kind, "designed".
   - It would store the description, engine, seed and chosen take instead of a recorded speaker.
   - The consent step would change, because there is no real person behind the voice.
   - This changes the voice model, so it needs your sign-off before coding.
   - The A/B itself needs none of this.

**After the A/B:**
- **Qwen wins:** build the Qwen worker engine (M) plus designed voices (needs sign-off).
- **IndexTTS with designed clips sounds just as good:** keep IndexTTS and add voice design only as a voice-creation tool.
- **Neither:** try VoxCPM2 next, with the same design-then-clone approach.

---

## 5. What a new engine will NOT fix

- **Bystanders sharing one voice.** Attribution folds unnamed speakers into buckets: chapter 34 has `character_auto`, `staff_auto` and `new_employee_auto`. Every line in a bucket gets the same voice on any engine. The fix belongs in attribution/casting, for example:
  - telling unnamed speakers apart by description ("the tall guard"), or
  - rotating a small pool of bystander voices per scene.

  Voice design makes that pool cheap to create, but it doesn't decide who is who.
- **Pauses at "…" and "—" inside a line.** Seiyuu only controls silence *between* blocks today: 0.6 s paragraph, 0.35 s dialogue beat, 1.8 s scene break, plus a silent beat for lines with nothing to say. Pauses inside a line are left to the engine, and none of the candidates handle them reliably (Qwen [#75](https://github.com/QwenLM/Qwen3-TTS/discussions/75), VoxCPM [#276](https://github.com/OpenBMB/VoxCPM/issues/276), F5 [#1034](https://github.com/SWivid/F5-TTS/issues/1034), Pocket [#6](https://github.com/kyutai-labs/pocket-tts/issues/6)).
  - The fix is a Seiyuu rule: split a line at "…" and "—" before synthesis and insert our own short silence.
  - It would help every engine, including Kokoro.
  - Because it changes how lines become audio clips, it may touch the render cache key, so I would propose it separately for sign-off.
- **Matching pace across engines.** No engine promises Kokoro's speed. Each needs its own pace setting, plus a words-per-minute check in the A/B report.
- **Character fit.** An engine can't read the book. Someone has to write each voice description; later the attribution step could draft them for you.
- **Wrong voice or wrong tone.** The speech check confirms the words, not who said them or how. Catching wrong-voice lines would need a separate voice-similarity check.
- **Long-paragraph chunking.** Engines split long text internally and lose intonation across the split. Seiyuu splitting long narration at sentence boundaries is a likely improvement on any engine (unverified until the A/B shows where the robotic patches fall).
