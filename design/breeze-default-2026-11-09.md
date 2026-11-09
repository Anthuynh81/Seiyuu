# Breeze TTS 2 as the default engine — plan for sign-off (2026-11-09)

The owner listened to the A/B (A = Qwen today, B = Breeze cloning the same cast, C = Breeze with
voice direction) and prefers B/C. Goal: **Breeze renders every new book by default**. Qwen stays
installed: it still designs voices, and it remains a selectable engine and fallback.

## What the test established

| | Qwen3-TTS (today) | Breeze TTS 2 |
|---|---|---|
| Speed (3080 Ti, fast path) | RTF ~0.6 | **0.45** clone · **0.54** direction (cfg 4) |
| VRAM peak | — | 8.8 GB of 12 |
| Whisper flags, 48 lines | 1 | 2 (both runs share the "Huuu." sigh) |
| Emotion / delivery control | none | natural-language instruction (+ `(sigh)`, `(laugh)`…) |
| Env | `qwen3-tts/` (torch 2.8) | `breeze-tts/` (torch 2.9.1+cu128, triton-windows) |

Windows needs `triton-windows<3.6` in the Breeze env and
`TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` (else `OverflowError: Python int too large to convert
to C long`). Fast config = depth decoder + codec + backbone decode (not the text encoder: +1.5 GB
for nothing). First compile 16–50 s; later worker starts reuse the inductor cache.

## Design

### Engine `breeze` (same shape as `qwen3tts`)

- `engines/breeze_engine.py`: stdlib-only adapter reusing `SubprocessTransport` (newline JSON).
  `requires_validation = True`, `clones_from_library = True`, 24 kHz mono (already canonical).
- `engines/breeze_worker.py`: runs in the Breeze env, never imports seiyuu; imports the Breeze
  repo from `BREEZE_REPO_DIR`, sets the inductor env var **before** torch is imported, loads once,
  warms the fast graphs, then serves `synthesize {text, reference_wav, ref_text, seed,
  instruction?, cfg?}`. Prototype (works): `breeze-tts/seiyuu_proto/breeze_synth.py` (A/B render) and `breeze_speed.py` (fast-path configs).
- Breeze cloning needs the reference's **exact transcript**. Voices that already bind one
  (`reference.txt` + digest; every designed voice does) work as-is; others get one from the
  existing whisper transcription service (`services/clone_transcript.py`) when switched.
- `model_version` = checkpoint fingerprint (offline, same recipe as Qwen) + fast-config name +
  cfg recipe, so the cost estimate builds the same SegmentKey as the render.
- Settings: `BREEZE_WORKER_PYTHON`, `BREEZE_REPO_DIR`, `BREEZE_CHECKPOINT_DIR`,
  `BREEZE_FAST=decode|off`, timeouts/restarts like Qwen. Normalization profile: start from
  Qwen's (`punct: comma`; same tokenizer lineage and the A/B used Qwen-normalized text).
- Same OOM / dead-worker kill + restart + retry as Qwen. `unload()` kills the worker.

### Delivery direction (what C did)

Breeze takes an instruction per line. The render passes a `delivery` setting chosen from the
segment, so it rides `settings_hash` (no cache-key format change):

| Segment | Instruction (book-level, editable) |
|---|---|
| Narration, first-person book | the protagonist's inner voice: calmer, drier, more intimate than aloud |
| Narration, third-person book | none (plain clone) or "clear, steady audiobook narration" |
| Thought (italics) | same inner-voice instruction |
| Dialogue | "natural, expressive spoken delivery that fits the line" |

One helper computes it for both render and estimate (the `_emotion_settings` parity rule).
Later, emotion verdicts (F2) can add a phrase ("angry, raising his voice"): Breeze is the first
local engine where that works without a numeric knob.

### Defaults switch

- Cast design keeps **Qwen VoiceDesign** to make the takes (proven; Breeze's own design mode is
  untested). The kept take becomes a CLONED **breeze** voice (same reference.wav + transcript).
- Clone dialog / `seiyuu voices clone` default → `breeze`; Qwen stays in the list.
- Existing Qwen voices: a **"Use Breeze"** action (CLI + Voices screen) switches a cloned voice
  between local cloning engines. See decision D2.
- README: setup section for the Breeze env.

## PRs

1. **Engine**: adapter + worker + settings + registry/bench/health wiring, fake-transport tests,
   one GPU smoke test (ask before running), README setup, `/breeze-tts/` in `.gitignore`.
2. **Delivery direction**: segment-type instruction in render + estimate, book-level narrator
   instruction in the cast plan, tests for key stability.
3. **Default switch**: design-keep → breeze voice, clone default, "Use Breeze" action, UI.
4. **Ghost-story re-render** on Breeze (~2.5 h + 15 min assemble; asked first), compared against
   the current Qwen m4b, which is backed up first.

## Decisions for the owner

- **D1 — DECIDED 2026-11-09: C (direction on)** is the default. It is the only option that
  gives the narrator Kim's voice *and* a distinct inner-voice delivery. Costs ~20 % more time
  than B. A per-book switch keeps B available.
- **D2 — switching existing voices.** Today a voice's engine is immutable ("a new voice").
  Recommend allowing the switch **only for cloned voices between local cloning engines**: the
  engine is already in the SegmentKey so no stale cache can be reused, consent binds to the
  reference bytes (unchanged), and every assignment keeps pointing at the same voice_id.
  Alternative: create a `<id>_breeze` copy and re-point assignments (more clutter, no gain).
- **D3 — voice design stays on Qwen** (recommended) until Breeze's design mode is tested.

The "who is I?" process plan (first-person protagonist detection, casting defaults, pre-render
cast check) is independent and feeds PR 2: it decides which book is first-person and who the
narrator is.
