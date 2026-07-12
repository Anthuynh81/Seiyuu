# Improvement Audit — 2026-07-11

Multi-agent verified audit (6 analyst lenses -> adversarial verifiers -> completeness critic; 13 agents).
Surviving ideas: 44 · Killed by verification: 2 · Critic additions: 7

Impact/effort grades are the verifier-adjusted ones. Every item was checked against the actual code; "VERIFIER" notes record what held up and what was corrected.

## Render / TTS wall-clock

### Implement the deferred voice-grouped (engine-grouped) synthesis in render_book_multivoice
*Impact: high · Effort: medium*

The SPEC-deferred optimization is confirmed still unimplemented: the multivoice loop synthesizes strictly in reading order, resolving the engine per segment and calling gpu.acquire per segment; when a segment's engine differs from the resident one the manager unloads it, and each reload is catastrophic (Chatterbox unload sets _model=None so the next segment pays a full from_pretrained; IndexTTS-2 unload terminates the worker process, and reboot is a fresh cu128 process + checkpoint load budgeted up to 600s). A narrator-on-kokoro / dialogue-on-chatterbox chapter alternates engines every few segments, so load/unload can dominate wall-clock. Fix stays inside pipeline.py: pass 1 computes SegmentKeys and partitions UNCACHED segments by engine, pass 2 synthesizes each engine group under one residency, pass 3 emits the manifest in reading order via cache.get — the docstring at pipeline.py:458 already guarantees the per-segment cache key makes synthesis order-independent, so no cache-key or schema change is needed (paid gating unaffected: elevenlabs has uses_gpu=False).

Evidence:
- `SPEC.md:258-260 — 'mixing engines (kokoro<->chatterbox) within a chapter thrashes the single GPU — voice-grouped synthesis is a deferred optimization'`
- `src/seiyuu/render/pipeline.py:530-540 and 588-592 — engine chosen per segment in reading order; gpu.acquire(engine) per uncached segment`
- `src/seiyuu/gpu/manager.py:39-43 — acquire() unloads the resident consumer whenever a different consumer acquires`
- `src/seiyuu/engines/chatterbox_engine.py:136-141 — unload() drops the model; next synth re-runs ChatterboxTTS.from_pretrained`
- `src/seiyuu/engines/indextts2_engine.py:358-366 — unload() terminates the worker; settings.py:121 indextts2_worker_load_timeout=600.0`
- `src/seiyuu/render/pipeline.py:456-459 — 'the per-segment cache key makes that order-independent for caching' (grouping is safe by design)`

> **Verifier:** Every evidence citation checks out: render_book_multivoice (pipeline.py:513-641) picks the engine per segment in reading order and gpu.acquire()s per uncached segment; manager.acquire (gpu/manager.py:39-41) unloads the resident consumer on any different acquire; ChatterboxEngine.unload drops _model (line 137) forcing from_pretrained next time; IndexTTS2.unload terminates the worker (indextts2_engine.py:358-366) with indextts2_worker_load_timeout=600.0 (settings.py:121). SPEC.md:258-260 confirms it is a deferred optimization and no grouping/partition code exists under render/. The pipeline.py:458-459 docstring genuinely guarantees order-independent caching, and elevenlabs uses_gpu=False keeps paid gating untouched (the cumulative _gate_paid cap is order-insensitive). Effort is real medium: the restructure must preserve broker lending (F1), cancel-resume semantics, validation-on-cache-hit, and the reading-order manifest, in a ~200-line function plus tests.

### Overlap CPU whisper validation with GPU synthesis (currently strictly serial per segment)
*Impact: high · Effort: large*

SPEC's 'validation runs on CPU concurrently with a GPU TTS model' refers only to memory non-contention — execution is serial: _synthesize_validated does engine.synthesize (GPU), audio.save to a tmp wav, then validator transcription (CPU) before the next segment starts, all inside the gpu.acquire with-body. With Chatterbox/IndexTTS-2 at warm RTF ~1.3x and whisper small/int8 adding roughly 0.3-0.5x realtime on CPU, validation adds ~25-40% to validated-engine render wall-clock that a one-slot pipeline would hide: hand attempt-1 audio to a single background validation thread, synthesize segment N+1 meanwhile, and re-queue a failed segment for its retry-with-seed+1 (the engine is still resident under grouping). As part of the same change, pass the in-memory float32 array to faster-whisper (it accepts np.ndarray) instead of the per-attempt tmp-wav write+read round trip.

Evidence:
- `src/seiyuu/render/pipeline.py:216-226 — per attempt: synthesize -> save(tmp) -> validate, sequentially, before the loop advances`
- `src/seiyuu/render/pipeline.py:357-362 and 593-598 — _synthesize_validated (including all whisper passes and retries) runs inside the gpu.acquire context`
- `src/seiyuu/validate/validator.py:73-80 — transcription takes a file path; settings.py:144 whisper_device='cpu' (no GPU contention, so true concurrency is safe)`
- `SPEC.md:273-274 — the concurrency claim is about contention, not pipelining`
- MEMORY.md m7 note — warm IndexTTS-2 RTF ~1.3x on the 3070, so whisper time is a material fraction of synth time

> **Verifier:** Confirmed strictly serial: _synthesize_validated (pipeline.py:216-226) does synthesize -> save(tmp) -> transcribe per attempt, entirely inside the gpu.acquire with-body (which also holds the manager lock through whisper time), and whisper_device='cpu' (settings.py:144) means true concurrency has no GPU contention — SPEC.md:273 is indeed only a non-contention claim. Both local cloning engines (chatterbox, indextts2) have requires_validation=True, so this hits the mainline local path, and the 25-40% overhead estimate is credible (whisper small/int8 ~0.3-0.5x RT vs warm synth ~1.3x RT). Two caveats temper it: faster-whisper's ndarray input expects 16 kHz mono while canonical audio is 24 kHz (needs a resample or in-memory wav bytes, not a raw array handoff), and the retry-with-seed+1 re-queue only avoids a reload if idea 1's engine grouping lands first — the proposal itself admits this dependency. Deferred verdicts also force the manifest-append flow to be restructured, justifying the large effort grade.

### Let multivoice renders reuse the warm EngineRegistry instances instead of constructing fresh engines
*Impact: medium · Effort: small*

render_book_multivoice's engine_for() calls get_engine() directly, building brand-new engine instances per render job; since GpuResourceManager compares consumers by identity, a model warmed by the warmup job, a prior render, or an audition is evicted and the fresh instance cold-loads the same multi-GB weights. The single-voice API path already fixed exactly this — handlers.py passes registry.get(engine_id) with the comment 'shared instance: warm re-acquire' — but the multivoice signature has no engine-provider seam. Add an optional engine_provider: Callable[[str], TTSEngine] parameter defaulting to the current get_engine behavior (CLI/tests unchanged) and pass registry.get from the API handler; the registry already constructs cloning engines with the consent-correct voices_dir. This converts every per-chapter iteration workflow (render ch1, listen, render ch2) from one cold load per job into a warm no-op re-acquire.

Evidence:
- `src/seiyuu/render/pipeline.py:490-494 — engine_for() -> get_engine(engine_id, **extra): fresh instances per multivoice render`
- `src/seiyuu/api/handlers.py:229-231 — single-voice path: 'registry.get(single.engine_id),  # shared instance: warm re-acquire'; handlers.py:207-227 — multivoice call has no engine source parameter`
- `src/seiyuu/api/registry.py:1-7 — registry exists so 'the GPU manager's identity comparison makes re-acquire a no-op instead of a multi-GB reload'`
- `src/seiyuu/gpu/manager.py:40 — 'self._resident is not consumer' identity check makes a new instance always evict the warm one`

> **Verifier:** The gap is real: engine_for() calls get_engine() fresh (pipeline.py:490-494) while the single-voice handler passes registry.get (handlers.py:231, 'shared instance: warm re-acquire'), and manager.py:40's identity check means a fresh instance always evicts a warm one. But the headline benefit is overstated: the 'render ch1, listen, render ch2' flow does NOT become warm from this change alone, because both render loops call gpu.free_all() in their finally (pipeline.py:405-408 and 650-651) — that flow additionally requires idea 5. What this alone delivers: a multivoice render stops cold-loading after a warmup job or audition (both deliberately leave the model resident — handlers.py:110-111, voices.py:623-624), and stops evicting the registry's warm instance (which flips is_resident false and makes the next audition cold). Consent invariants hold since registry._construct_kwargs uses the same settings.voices_dir the handler's VoiceLibrary uses. Impact downgraded to medium standalone; effort small is accurate (optional engine_provider param + one call-site change).

### Skip the per-segment conds torch.load in Chatterbox when the voice has not changed
*Impact: low · Effort: small*

_synthesize_native calls _ensure_conds on every synthesis, and when the conds .pt exists it unconditionally does model.conds = Conditionals.load(path, map_location=device) — a disk read + host-to-device transfer per segment even when the same voice renders thousands of consecutive segments (the entire single-voice path, and long same-speaker runs in multivoice; validation retries re-enter synthesize and reload again per attempt). Memoize the currently-loaded conds identity (voice_id + conds path, which already embeds model_version and the reference hash) and skip the load when unchanged, resetting the memo in unload(). Small fixed cost times every segment of a book, for a ~5-line change with no behavioral or cache-key impact.

Evidence:
- `src/seiyuu/engines/chatterbox_engine.py:120-124 — _synthesize_native -> _ensure_conds on every call`
- `src/seiyuu/engines/chatterbox_engine.py:106-110 — 'if path.is_file(): model.conds = self._load_conds(path)' with no check of what is already loaded`
- `src/seiyuu/render/pipeline.py:216-220 — each validation retry calls engine.synthesize again, repeating the reload`
- `src/seiyuu/engines/chatterbox_engine.py:92-99 — conds_path already keys model_version + reference hash, so (voice_id, path) is a sound memo key`

> **Verifier:** Evidence exact: _ensure_conds runs on every _synthesize_native call (chatterbox_engine.py:120-124) and unconditionally Conditionals.load()s when the .pt exists (lines 106-110), including on every validation retry; no memo exists, and conds_path (lines 92-99) already keys model_version + reference hash so the proposed memo key is sound (memo must also be set by the prepare_conditionals branch and reset in unload()). But impact is overstated: the conds object is a few MB, so the load+H2D transfer is tens of milliseconds against seconds-per-segment autoregressive synthesis — well under ~1% of render wall-clock even over thousands of segments. Correct, safe, cheap, but low impact.

### Make render-end GPU release lazy in server mode instead of unconditional free_all()
*Impact: medium · Effort: small*

Both render loops call gpu.free_all() in their finally block, unloading the resident model at the end of every render job — yet the manager is explicitly designed for lazy release ('a model stays resident after its work so back-to-back use is cheap, and is freed only when a competitor acquires'), and the audition path already deliberately leaves the model resident. Back-to-back render jobs (or render followed by audition of the same engine) therefore pay a full reload each time for no VRAM benefit, since any competitor (attribution providers acquire through the same manager) evicts lazily anyway. Gate the free on a flag (e.g. release_gpu=True default for CLI where an out-of-process ollama run may follow, False from the API handler whose warmup/eviction lifecycle the manager already owns); combined with the registry-reuse idea this makes sequential chapter renders fully warm.

Evidence:
- `src/seiyuu/render/pipeline.py:405-408 and 650-651 — unconditional gpu.free_all() at the end of both render loops`
- `src/seiyuu/gpu/manager.py:5-7 — 'Release is LAZY ... freed only when a competitor acquires (or free_all() at teardown)'`
- `src/seiyuu/api/routes/voices.py:617-623 — audition path: 'NO free_all(): the model stays lazily resident by design'`
- `src/seiyuu/services/attribution.py:205 — attribution acquires the manager on its provider, so an LLM job still evicts a lingering TTS model correctly`

> **Verifier:** All evidence confirmed: both render loops free_all() in finally (pipeline.py:405-408, 650-651); the manager module docstring (manager.py:5-7) documents lazy release as the design; the audition path (voices.py:623-624) and the warmup handler (handlers.py:110-112) both deliberately leave the model resident — renders are the only server path that unconditionally frees. Eviction correctness holds: local attribution providers acquire through the same manager (services/attribution.py:203-208), so a later LLM job lazily evicts a lingering TTS model, and lifespan teardown already free_all()s (main.py:96). The CLI-default-True flag respects the existing 'free the GPU for the next stage/process' rationale (out-of-process Ollama in a convert pipeline). This is the necessary second half of the warm-sequential-renders pair with idea 3; medium/small stands.

### Parallelize ElevenLabs segment synthesis with a small bounded worker pool
*Impact: medium · Effort: medium*

Cloud segments are synthesized one HTTP round trip at a time inside the reading-order loop (uses_gpu=False, so nothing but the loop serializes them); at typical 1-3s API latency a several-thousand-segment cloud render spends most of its wall-clock waiting on the network serially. A pool of 3-4 workers for elevenlabs-engine segments would cut that portion 3-4x. The cost gate needs care but no policy change: the quote is already verified and the run capped by max_paid_usd before synthesis, so pre-charge _gate_paid serially when dispatching each task (charging is already estimate-based, engine.cost_estimate(text), not response-based) and fail the pool loudly on the first refusal — no automatic path bills anything the gate did not already admit.

Evidence:
- `src/seiyuu/engines/elevenlabs_engine.py:26-29 — uses_gpu=False, requires_validation=False: no GPU manager or whisper coupling forces serialization`
- `src/seiyuu/render/pipeline.py:530-598 — one segment per loop iteration; cloud segments share the same serial path as GPU ones`
- `src/seiyuu/render/pipeline.py:123-155 — _gate_paid charges from engine.cost_estimate(text) before synthesis, so pre-charging at dispatch preserves the budget cap semantics`
- `src/seiyuu/api/handlers.py:185-197 — quote verified and consumed before the render loop starts`

> **Verifier:** Confirmed unimplemented and evidence accurate: uses_gpu=False / requires_validation=False (elevenlabs_engine.py:28-29) so nothing but the loop serializes cloud segments; they ride the same one-at-a-time loop (pipeline.py:530-598); _gate_paid charges from engine.cost_estimate(text) BEFORE synthesis (pipeline.py:138-155) so serial pre-charge at dispatch preserves the cumulative max_paid_usd cap exactly; the quote is verified+consumed before the loop (handlers.py:185-197); no ThreadPoolExecutor exists anywhere in src. Policy-compatible — no new automatic paid path, the run is already inside a verified quote. Effort is honestly medium, slightly heavier than sketched: ensure_cloud_voice mutates the cloud slot LRU registry and must be serialized at dispatch, and ElevenLabs per-tier concurrency limits need 429 handling plus loud fail-fast pool teardown on the first gate refusal. Impact medium: only cloud renders benefit, but for those the 3-4x cut of serial network latency is real.

### Parallelize per-chapter assembly (independent numpy concat + two ffmpeg passes per chapter)
*Impact: low · Effort: small*

assemble_book processes chapters strictly serially: read every segment wav, concatenate in numpy, write a temp wav, then spawn up to two ffmpeg processes (loudnorm measure + mp3 encode) per chapter. Chapters are fully independent (independent inputs, independent output files, results only appended to lists), and the work is a mix of GIL-releasing I/O and subprocesses, so a ThreadPoolExecutor of 3-4 workers gets a near-linear speedup on this stage; keep master_book serial (it streams into one book.wav). Worth doing because assemble runs on every convert and every pause/loudness re-tune, where it is the entire wall-clock.

Evidence:
- `src/seiyuu/assemble/pipeline.py:224-240 — serial chapter loop: _chapter_samples -> sf.write -> _measure_loudness -> _encode_mp3`
- `src/seiyuu/assemble/pipeline.py:150-163 and 166-188 — two subprocess.run ffmpeg invocations per chapter when loudness is set`
- `src/seiyuu/assemble/pipeline.py:86-134 — _chapter_samples touches only that chapter's segment files (no shared mutable state)`

> **Verifier:** Evidence accurate: assemble_book's chapter loop is strictly serial (assemble/pipeline.py:224-240), spawns _measure_loudness + _encode_mp3 subprocess.run per chapter (lines 150-188), and _chapter_samples touches only that chapter's files with results appended to lists — chapters are independent, and master_book is correctly excluded (it streams one book.wav). No parallelism exists in the stage. The work mix (GIL-releasing soundfile I/O + ffmpeg subprocesses) does parallelize, though 'near-linear' is optimistic on disk-bound machines and per-worker memory is one chapter of float32 (~170MB/30min — fine). Impact is honestly low: assembly is minutes against multi-hour renders, though it IS the whole wall-clock on pause/loudness re-tunes as claimed. Effort trimmed to small: a bounded ThreadPoolExecutor over an already-independent loop with order preservation and cancel checks at submission.

## Attribution throughput & caching

### Pure-narration fast path: skip the LLM call for chunks with zero quoted spans
*Impact: high · Effort: small*

In the dispatch loop, before calling the provider, check whether any owned block in the chunk has a quoted span (or a thought candidate when emit_thoughts is on). If none, synthesize the ChunkAttribution deterministically (one NARRATION segment per block) and skip the Ollama round trip entirely. This is provably output-equivalent for segments: _assemble_segments only emits DIALOGUE for quoted spans and THOUGHT for known candidate_ids — with neither present, every span degrades to narration regardless of what the model returns. The only loss is CharacterMention enrichment from narration-only chunks (registry descriptions/gender for non-speaking scenes), so gate it behind a config flag. Narration-heavy stretches (prologues, description-dense chapters) become free; each skipped chunk saves a full ~5-10s prefill+decode cycle.

Evidence:
- `src/seiyuu/attribute/pipeline.py:174-209 — every uncached chunk unconditionally goes through _attribute_chunk_validated → provider.attribute_chunk`
- `src/seiyuu/attribute/providers/base.py:203-207 — spans are computed deterministically BEFORE the LLM call, so quoted/candidate presence is known pre-dispatch`
- `src/seiyuu/attribute/providers/base.py:310-368 — quoted branch and verdict branch are the only non-narration paths; with no quoted spans and no candidate_ids, model output cannot affect segments`
- `src/seiyuu/attribute/registry.py:99-100 — the only thing lost by skipping is mention merging into the registry`
- `src/seiyuu/attribute/spans.py:27-43 — a block with no double quotes is a single prose span`

> **Verifier:** Not implemented: pipeline.py:174-209 sends every uncached chunk through _attribute_chunk_validated unconditionally. Output-equivalence verified in providers/base.py:289-369 — _assemble_segments emits DIALOGUE only for quoted spans and THOUGHT only for candidate_ids in the deterministic known_ids set; with neither present every span becomes NARRATION (default confidence 1.0) regardless of model output, so a synthesized ChunkAttribution is byte-equivalent for segments. spans.py is importable and deterministic so the pipeline can test quote/candidate presence pre-dispatch. Mention loss is even smaller than claimed: v5.md line 29 asks only for people who SPEAK in the slice, so a compliant model returns [] on quote-free chunks anyway. With M8 PDF non-fiction (zero dialogue) entire books skip the LLM. Reconstruction guard trivially passes. Kept high impact: content-dependent for fiction (a chunk qualifies only if ALL owned blocks are quote-free), but each skipped chunk saves a full ~5-10s call and the win is total on narration-heavy/non-fiction books.

### v7 prompt contract: omit pure-narration block entries and re-listed known characters to cut decode tokens
*Impact: medium · Effort: small*

Decode (~60-80 t/s locally) dominates per-chunk latency, and the current contract forces output fat: rule 1 demands one entry per block, so every narration block costs ~25 output tokens of {"speaker": null, "confidence": 1.0, "emotion": null}, and the characters section re-describes speakers already in the registry (~40 tokens each) every chunk. The assembler already tolerates omitted entries — blocks.get() returning None degrades that block to narration — so a v7 prompt saying 'omit blocks with no dialogue; list only characters NOT already in the registry (or with new information)' is a prompt-file-only change plus updating the retry reminder text. prompt_version is in the cache PK, so v5/v6 cached rows stay valid and the two contracts never collide. Expect roughly 20-35% decode reduction on mixed dialogue/narration chunks.

Evidence:
- `src/seiyuu/attribute/providers/base.py:291-307 — blocks.get(block_id) is None-tolerant; an absent entry yields narration, exactly what an omitted no-dialogue block should mean`
- `prompts/attribution/v5.md:80 — 'One entry per block' rule forces null entries; v5.md:29 — 'characters lists the people who speak in this slice' forces re-listing known speakers`
- `src/seiyuu/attribute/providers/base.py:225-228 — retry reminder restates one-entry-per-block and must change in lockstep`
- `src/seiyuu/attribute/cache.py:28 — prompt_version in the primary key isolates v7 rows from v5/v6`

> **Verifier:** Evidence checks out: base.py:292 blocks.get(block_id) is None-tolerant (absent entry -> block_speaker None -> all spans narration, exactly the intended meaning of an omitted no-dialogue block); v5.md rule 1 (line 80) forces one entry per block and line 29 makes the model re-list known speakers; retry reminder at base.py:225-228 must change in lockstep; prompt_version is in the cache PK (cache.py:28). One inaccuracy: this is NOT prompt-file-only — _PER_QUOTE_VERSIONS = frozenset({"v5","v6"}) at base.py:36 must gain "v7" or the F1 per-quote marker rendering silently turns off (a quality regression on multi-quote blocks). That fix is trivial but it is code. Real risk capping impact: a model that over-omits a dialogue-bearing block degrades its quotes to narration SILENTLY (reconstruction still passes, no flag), so the change needs a quality eval, not just a wall-clock one. Decode savings claim is plausible (~25 tokens per narration entry, decode at 60-80 t/s dominates).

### Chunk-size bake-off: raise attribution_chunk_tokens from 800 toward 1600-2400
*Impact: medium · Effort: small*

Each request resends a fixed overhead — the ~1,000-token static template (v5.md is 4,153 bytes), the full registry JSON, and 2+2 overlap context blocks — around only 800 owned tokens, so 60-80% of prefill is repeated boilerplate and the number of round trips is high. Doubling the budget halves the call count and the total boilerplate prefill. The 800 default was chosen for schema reliability, but the native transport now enforces the shape via Ollama structured outputs (the format grammar), which removes most of that risk; num_ctx 8192 leaves headroom for ~2400-token chunks plus registry and output. This is a config-only experiment — run one fixture book at 800 vs 1600 vs 2400 and compare wall clock plus flag rate. Note the new partition produces new chunk_hashes, so the first run at a new size is a full cache miss (one-time cost, no collision).

Evidence:
- `src/seiyuu/settings.py:57-59 — attribution_chunk_tokens=800 with the schema-reliability rationale`
- `src/seiyuu/attribute/chunking.py:49-86 — greedy budget chunker; smaller budget = more chunks = more per-call overhead`
- `src/seiyuu/attribute/providers/local.py:120-130 — 'format': schema constrains decoding server-side, mitigating the malformed-JSON concern that motivated 800`
- `prompts/attribution/v5.md measured at 4153 bytes (~1040 tokens) resent per chunk; settings.py:60 — overlap_blocks=2 context resent on both sides`

> **Verifier:** Evidence accurate: settings.py:59 defaults 800 with the schema-reliability rationale (lines 57-58); chunking.py:49-86 is the greedy budget walk; local.py:124 passes the JSON schema as Ollama's server-side format grammar, which does mitigate the malformed-output concern that motivated 800; v5.md measured at exactly 4153 bytes (~1040 tokens) resent per chunk plus registry plus 2+2 overlap. The knob already exists so the experiment is config-only as claimed. Honest caveats: savings are boilerplate PREFILL only (total decode is unchanged), a reconstruction failure now flags a whole 1600-2400-token chunk (bigger blast radius, measured by the proposed flag-rate comparison), and late-book registry + 2400 owned + ~1500 output gets tight inside num_ctx 8192 (truncation is retry-then-flag, not silent). New chunk_hashes = one-time full cache miss, correctly noted. Worth running; medium impact is fair for the 1600 step, 2400 likely too aggressive.

### Trim registry prompt fat: compact rendering and bounded fields in render_prompt
*Impact: medium · Effort: small*

render_prompt dumps the ENTIRE running registry into every chunk prompt as indent=2 JSON with all five fields including description. The registry grows monotonically, so late-book chunks carry a registry section that can exceed the 800-token owned payload several times over (60 characters at ~150 bytes each is ~2,250 tokens per request), making attribution measurably slower as the book progresses. Cheap levers, in order: render compact (no indent, drop null fields), truncate description to ~80 chars, and optionally cap the list to characters whose name/alias appears in the chunk+context text plus the N most recent speakers. The registry is deliberately NOT part of the cache key, so this changes no cache semantics, but it does change prompts — bump prompt_version so cached rows from the old render stay distinguishable.

Evidence:
- `src/seiyuu/attribute/providers/base.py:135-148 — json.dumps(all characters, indent=2) with name/aliases/gender/age_hint/description every chunk`
- `prompts/attribution/v5.md:90-93 — {registry_json} is embedded in every chunk prompt`
- `src/seiyuu/attribute/registry.py:60-72,108-115 — registry only ever grows as chunks resolve`
- `src/seiyuu/attribute/pipeline.py:8-9 — cache key intentionally has no registry component, so the render shape is free to change without cache migration`

> **Verifier:** Evidence accurate: base.py:135-148 dumps every character with all five fields at indent=2 into every chunk prompt; registry.py only ever appends (lines 61-73, 108-115); pipeline.py:8-9 documents that the cache key deliberately has no registry component, so the render shape is free to change (the prompt_version bump is conservative hygiene, not a cache-correctness requirement). Not implemented anywhere. Two cautions from code the proposer did not cite: the render shape was DELIBERATELY matched to the CharacterMention output shape because the model copied whatever shape it saw (base.py:131-134 comment) — dropping null fields needs a quick check that the format grammar still keeps output shape honest (it should, since Ollama enforces the schema server-side); and capping the list risks name-variant re-declarations, mitigated by the proposed appears-in-chunk-text filter. Compact + no-nulls + truncated description alone is riskless and cuts the late-book registry roughly in half; compounds with the chunk-size idea.

### Fixed-lag registry pipelining with OLLAMA_NUM_PARALLEL=2
*Impact: medium · Effort: large*

The dispatch loop is strictly one-chunk-at-a-time: the provider call blocks, then resolve_chunk mutates the registry before the next chunk's prompt is rendered. Plain client-side concurrency alone gains almost nothing (Ollama with default NUM_PARALLEL=1 serializes requests, and client-side gaps are milliseconds vs multi-second inference). The real win requires two coordinated changes: run Ollama with num_parallel=2 (decode is memory-bandwidth-bound, so batching two streams yields ~1.5-1.8x aggregate throughput) and dispatch with a deterministic fixed-lag registry — chunk N+1's prompt uses the registry snapshot after chunk N-1, with resolve_chunk still applied in order on completion. Determinism is preserved (lag is fixed, not timing-dependent) and cache semantics are unaffected since the registry is not in the key; the one-chunk-stale registry is absorbed by resolve_chunk's minimal-record path plus the alias post-pass. Caveats to validate: KV cache per slot (~0.5GB at num_ctx 8192 for qwen2.5:7b) must fit beside the 4.7GB weights on 8GB, and retry/escalation ordering gets more complex.

Evidence:
- `src/seiyuu/attribute/pipeline.py:174-218 — the dispatch loop: provider call at line 189, resolve_chunk at line 211 updates the registry before the next iteration renders its prompt`
- `src/seiyuu/attribute/providers/base.py:123-159 — render_prompt embeds the current registry, creating the sequential dependency`
- `src/seiyuu/attribute/cache.py:47-53 + pipeline.py:8-9 — cache key excludes registry state, so a bounded-lag snapshot is consistent with existing cache semantics`
- `src/seiyuu/attribute/providers/local.py:120-137 — single blocking HTTP POST per chunk, no concurrency anywhere in the provider`

> **Verifier:** Evidence accurate: the dispatch loop is strictly sequential (provider call pipeline.py:189, resolve_chunk at 211 mutates the registry before the next prompt renders), render_prompt embeds the live registry (base.py:135-155), the provider is one blocking urllib POST (local.py:117-144), and grep confirms zero concurrency primitives anywhere in src/seiyuu/attribute. The cache-semantics claim holds — registry is excluded from the key, and note the same key can already store different payloads across runs (retry temperature varies), so a one-chunk-stale registry is consistent with existing semantics. KV math checks out (~0.46GB/slot at 8192 ctx for qwen2.5:7b's GQA layout; 4.7GB weights + 2 slots ≈ 5.7GB — feasible but tight on an 8GB card also driving a desktop, must be validated as the proposal says). Honest about the real costs: threading the dispatch loop, retry/escalation/flag ordering, cancellation, and per-chapter boundaries make this genuinely large. The ~1.5-1.8x aggregate decode claim for batched memory-bound decoding is plausible. High risk/complexity relative to the safer wins in ideas 1-4; do those first.

### Stop force-unloading Ollama at the end of every successful attribution run
*Impact: low · Effort: small*

run_attribution calls gpu.free_all() in its finally block on every run, which POSTs keep_alive:0 and then polls /api/ps in 0.5s steps — even on success. This defeats the GPU manager's documented lazy-release design (a competitor acquire already evicts the resident consumer safely), so back-to-back attribution work — per-chapter re-attributes from the UI, attribute followed by adjudicate, multi-book batches — pays an unload-poll plus a full qwen2.5:7b reload (~5-15s) per run. The warmup handler already does exactly the right thing for TTS engines ('the model STAYS resident so the next re-acquire is an identity no-op'). Change: on success, leave the LLM resident and let the manager evict it when a TTS engine acquires; keep the eager free only on failure/cancel if desired. Ollama auto-freeing after keep_alive 5m is benign — unload() of a non-resident model is a cheap no-op.

Evidence:
- `src/seiyuu/services/attribution.py:238-242 — finally: gpu.free_all() on every run including success`
- `src/seiyuu/gpu/manager.py:5-7,36-43 — 'Release is LAZY... freed only when a competitor acquires'; acquire() already unloads a different resident consumer`
- `src/seiyuu/api/handlers.py:110-112 — warmup deliberately leaves the engine resident, the exact pattern attribution is denied`
- `src/seiyuu/attribute/providers/local.py:186-205 — unload posts keep_alive 0 then polls /api/ps every 0.5s up to unload_poll_timeout`

> **Verifier:** Evidence is accurate (finally: gpu.free_all() on every run at services/attribution.py:238-242; lazy-release design in gpu/manager.py:5-7; warmup precedent at api/handlers.py:110-112) but the proposed mechanism only half-delivers: GpuResourceManager.acquire compares consumer IDENTITY (manager.py:40, `self._resident is not consumer`) and run_attribution builds a NEW OllamaProvider instance per run (line 193 via build_provider), so in the long-lived API server — the headline 'per-chapter re-attributes from the UI' scenario — the next run's acquire evicts the same-model resident anyway and the unload+reload merely moves from run N's end to run N+1's start (zero net gain, zero regression). The benefit is real only for cross-process CLI sequences (attribute then adjudicate, multi-book batches), where a fresh process's manager has no resident and hits a warm Ollama within its 5m keep_alive. To deliver the server scenarios the change must also key residency by the name string acquire already receives (or cache provider instances) — still small, but the proposal as written omits it. Downgraded to low impact.

### Anthropic prompt caching for the paid/hybrid path (cache_control on the static prefix)
*Impact: low · Effort: medium*

The Anthropic provider sends the forced tool schema plus the whole rendered prompt with no cache_control, so a full-anthropic book run re-bills the ChunkLabels tool schema, the static template, and the registry at full input price every chunk, seconds apart. Splitting the user message into two content blocks at the {registry_json} boundary (static template prefix + dynamic suffix) and marking the prefix — or the tool definition — with cache_control {type: ephemeral} gets 0.1x pricing on the repeated portion within the 5-minute TTL; forced tool_choice does not invalidate the tools/system cache tier. Honest caveats that cap the impact: the configured model claude-opus-4-8 has a 4096-token minimum cacheable prefix, so tool schema + template (~1,600-1,900 tokens) alone silently won't cache early-book — the breakpoint should sit after the registry section, which is stable between consecutive chunks mid/late-book; and this path only runs when the user explicitly enables the paid provider or hybrid escalation (sporadic escalations may fall outside the TTL).

Evidence:
- `src/seiyuu/attribute/providers/anthropic.py:86-99 — messages.create with tools and a single user string, no cache_control anywhere`
- `src/seiyuu/settings.py:169 — anthropic_model='claude-opus-4-8' (4096-token minimum cacheable prefix per current Anthropic docs, verified via the claude-api skill)`
- `src/seiyuu/attribute/providers/base.py:154-159 + prompts/attribution/v5.md:92 — the rendered prompt is static template text up to {registry_json}, giving a natural split point`
- `src/seiyuu/services/attribution.py:196-199 — hybrid escalation builds this provider; CLAUDE.md paid-gating is unchanged by caching (it only cheapens the already-explicit path)`

> **Verifier:** Not implemented (grep: no cache_control/ephemeral anywhere in src; anthropic.py:86-99 sends tools + one plain user string). Verified the two load-bearing API claims against the claude-api skill: minimum cacheable prefix for claude-opus-4-8 is exactly 4096 tokens (Opus 4.8 row in the caching minimums table), and tool_choice changes invalidate only the messages cache tier — tools/system stay cached — and tool_choice is constant across chunk calls here anyway. The proposal's own caveats are correct: tool schema + template (~1.6-1.9k tokens) alone silently won't cache on this model, so the breakpoint must sit after the registry section (stable between consecutive chunks, invalidated only when the registry grows), and the path only runs when the paid provider/hybrid is explicitly enabled — CLAUDE.md's paid gating is untouched since caching only cheapens already-authorized calls. Effort is medium not small: the prompt reaches _tool_call as a single rendered string through the _complete_json seam, so splitting at the {registry_json} boundary requires reshaping the template-method interface or re-splitting the string at a sentinel. Impact stays low — real dollar savings but only on the rarely-enabled paid path.

## Attribution & casting quality

### Unattributed quotes silently become confidence-1.0 narration and never reach the review queue
*Impact: high · Effort: small*

When the model returns speaker=null for a quote (or an unlabeled/out-of-range per-quote index), the quoted span degrades to a NARRATION Segment built without a confidence argument, so it gets the schema default of 1.0 — the model's own uncertainty (QuoteSpeaker.confidence) is discarded. The Character Review queue filters on `speaker !== null && confidence < threshold` and the API's low_confidence filter on `confidence >= threshold`, so a quote rendered in the narrator's voice — exactly the thing a reviewer should check — is invisible everywhere. Fix: propagate the model's confidence (or a fixed 0.0) on the degrade path, count narration-typed segments whose text starts with a quote glyph (spans.is_quoted_span already exists) in the characters overview, and add them to the review queue. No Segment schema change needed; confidence already exists on narration segments (the flagged-chunk fallback already uses 0.0).

Evidence:
- `src/seiyuu/attribute/providers/base.py:340-344 — degrade path constructs Segment(type=NARRATION) with no confidence`
- `src/seiyuu/attribute/models.py:63 — Segment.confidence defaults to 1.0`
- `frontend/src/screens/Review.tsx:471 — queue filter `s.speaker !== null && s.confidence < threshold` excludes all narration`
- `src/seiyuu/api/routes/books.py:489 — low_confidence filter also passes these at 1.0`
- `src/seiyuu/services/characters.py:50-55 — low_confidence counted only for speaker != None`
- `src/seiyuu/attribute/pipeline.py:101-106 — the flagged-fallback already uses confidence=0.0 narration, proving the pattern`

> **Verifier:** Every cited line verified. base.py:341-344 and 365-368 build Segment(type=NARRATION) without a confidence argument on the quote-degrade paths, so models.py:63's default 1.0 applies and QuoteSpeaker/BlockSpeaker confidence is discarded. Review.tsx:471 filters `s.speaker !== null && s.confidence < threshold` (narration always excluded), books.py:489 drops confidence>=threshold rows from the low_confidence view, and characters.py:50-52 does `if seg.speaker is None: narration += 1; continue` before counting low_confidence — so a narrator-voiced unattributed quote is invisible in all three surfaces. pipeline.py:101-106 proves the confidence-0.0-narration pattern already exists (flagged fallback), and is_quoted_span (spans.py:41) is referenced nowhere outside spans.py, confirming no other surfacing exists. No schema change needed; fix is a few lines each in the provider, characters service, API filter, and Review.tsx queue filter. Real, silent quality hole on the default local-model path.

### Hybrid escalation almost never fires — add opt-in confidence-based escalation (closes a SPEC open question)
*Impact: medium · Effort: medium*

Escalation to Anthropic triggers only when `outcome.attribution is None`, which since reconstruction became structural (spans sliced from source, 'reconstruction cannot be violated by the model') can only happen via MalformedOutputError — and Ollama structured outputs make that rare too. So paid hybrid mode buys almost no quality: low-confidence speaker calls, the actual local-model failure mode, never escalate. Add an opt-in `attribution_escalate_below` threshold: after a validated local chunk, if any owned dialogue segment's confidence falls below it, re-run that chunk through the (already explicitly-enabled, key-gated) escalation provider and keep the higher-confidence result. This directly answers SPEC's 'Hybrid escalation defaults: confidence threshold' open question; while there, note the cache currently stores an escalated result under the PRIMARY provider's key (pipeline.py:176-183 + 208), which is worth making explicit or fixing.

Evidence:
- `src/seiyuu/attribute/pipeline.py:190-194 — escalation only when attribution is None (reconstruction/malformed failure)`
- `src/seiyuu/attribute/spans.py:1-13 — reconstruction is structural; the model cannot violate it`
- `src/seiyuu/settings.py:63-65 — attribution_hybrid exists but only gates the dead-ish flag path`
- `SPEC.md:352-353 — 'Hybrid escalation defaults: confidence threshold, max local retries' still open`
- `src/seiyuu/api/routes/books.py:346-361 — paid gate already handles explicit consent for hybrid`

> **Verifier:** Confirmed: pipeline.py:190-194 escalates only when outcome.attribution is None, which requires all retries to fail via MalformedOutputError (non-dict output, base.py:230-233) or a reconstruction failure that the structural span-slicing design makes near-impossible (spans.py:1-13, base.py docstring 'reconstruction cannot be violated by the model'). Grep confirms no confidence-based escalation exists anywhere; attribution_confidence_threshold (settings.py:63) only feeds review surfacing. SPEC.md:352-353 open question and the books.py:346-361 confirm_paid gate both check out, and the cache observation is accurate: ChunkCacheKey is built from the PRIMARY provider (pipeline.py:176-183) and cache.put at line 208 stores an Anthropic-escalated result under the local provider/model key. Impact regraded to medium: it only benefits users who explicitly enable paid hybrid mode — though for them it converts a near-dead premium path into a functional one, and the consent surface (confirm_paid + key gate) already exists.

### Attribution eval harness using Character Review edits as gold labels, then re-run the stale model bake-off
*Impact: medium · Effort: medium*

There is no bake-off or accuracy harness anywhere (no scripts, no CLI command), and qwen2.5:7b was chosen at M2 — SPEC explicitly asks to re-run the bake-off, and by mid-2026 several stronger small non-thinking models fit 8GB. Manual reassigns are already ground truth (applied at confidence 1.0), so a book the user has corrected IS a labeled eval set: add a `seiyuu eval --model X` command that re-attributes with a candidate model (cache is keyed by model_id, so runs coexist without clobbering), diffs per-segment speakers against the effective (edited) report, and prints accuracy plus a disagreement list. This makes model swaps and prompt-version changes measurable instead of vibes-based, and it is free (local models only).

Evidence:
- `No bake/benchmark/eval harness found: scripts/ contains only Smoke.py, audition_indextts2_emotions.py, demo_section2.py`
- `SPEC.md:354-357 — 'Re-run the bake-off (incl. Gemma)...' open question; qwen2.5:7b chosen at M2`
- `src/seiyuu/services/edits.py:201-207 — 'a manual reassign is ground truth: confidence 1.0'`
- `src/seiyuu/attribute/pipeline.py:176-183 — ChunkCacheKey includes model_id, so candidate-model runs never collide`
- `src/seiyuu/settings.py:44-46 — attribution_model default 'qwen2.5:7b' with the M2-era rationale comment`

> **Verifier:** Confirmed missing: scripts/ contains only Smoke.py, audition_indextts2_emotions.py, demo_section2.py; no eval/bake/benchmark/accuracy match in cli.py. SPEC.md:354-357 explicitly leaves 'Re-run the bake-off (incl. Gemma)... cache is keyed by model_id so results don't clobber' open, and edits.py:201-207 confirms manual reassigns are confidence-1.0 ground truth. Segment boundaries are model-independent (deterministic span split), so per-segment diffing is well-founded, and attribute_book returns a report without writing attribution.json, so a candidate run needn't clobber state. Two honest caveats that temper the grade: only user-edited segments are true gold (unedited ones just measure agreement with the incumbent model), and speaker comparison must match by name, not id (different models mint different registry ids/aliases), which is the real work in the diff. Impact regraded to medium — it is instrumentation that enables quality gains rather than a direct gain.

### Single-quote (UK convention) dialogue books currently yield zero dialogue segments
*Impact: high · Effort: medium*

The span splitter recognizes only double quotes (straight + curly); single quotes are deliberately left alone because they are usually apostrophes. But an entire class of UK-published books uses ‘single quotes’ for all dialogue — such a book splits into pure narration, renders single-voice, and nothing flags it. Add book-level convention detection (count curly ‘…’ pairs at word boundaries vs double-quote runs across the normalized book); when single-quote dominant, switch _QUOTED_RUN to a guarded single-quote pattern (curly open ‘ is unambiguous; apostrophes are ’ inside words). Note the cache consideration: assembly happens before caching, so already-cached chunks stay quote-blind — a re-attribute needs a prompt_version bump or a convention component in the cache key (key-format changes need approval per CLAUDE.md, so the prompt bump is the cheap route). At minimum, detect the convention and WARN in the attribution report so the failure stops being silent.

Evidence:
- `src/seiyuu/attribute/spans.py:21 — _QUOTED_RUN matches only [“"]...[”"]`
- `src/seiyuu/attribute/spans.py:9-13 — 'single quotes/apostrophes are left alone' by design, no book-level convention check anywhere`
- grep for single-quote handling in ingest/normalize found nothing (only pdf.py terminal-char list)
- `src/seiyuu/attribute/providers/base.py:249-269 — segment type derives solely from span.quoted, so no dialogue spans means no dialogue segments`

> **Verifier:** Confirmed: spans.py:21 _QUOTED_RUN matches only straight/curly DOUBLE quotes and the module docstring (lines 9-13) documents leaving single quotes alone by design. Grep across src/seiyuu for single-quote/convention handling finds only the validate.py glyph-fold table, a lexicon token regex, and pdf.py's terminal-char list — no book-level convention detection anywhere in ingest or normalize, and nothing converts ‘…’ to double quotes upstream. Segment type derives solely from span.quoted (base.py:310-345), so a single-quote-convention book splits to pure narration, renders single-voice, and nothing warns. The proposal correctly handles the cache consequence (assembly precedes caching, so a prompt_version bump is the approval-free invalidation route per CLAUDE.md's cache-key freeze). Silent total failure for a common publishing convention justifies high impact; detection + guarded pattern + tests is medium effort (the warn-only floor alone would be small).

### Annotate seam-context blocks with their already-resolved speakers
*Impact: medium · Effort: medium*

Chunk context blocks (overlap_blocks=2) are rendered as raw text; the model re-derives who was speaking at a chunk seam from scratch, which is the classic failure for long alternating untagged dialogue. The pipeline has already resolved those blocks' speakers (they were owned by the previous chunk and sit in by_block), so pass them to the provider and render leading context as e.g. `[ch004_b0012] (speaker: Elizabeth) ...`. This mirrors how the running registry already threads into prompts without being part of the cache key (accepted by design, pipeline.py docstring), and needs a prompt v7 to invalidate old cached rows. Cheap tokens, targets the single hardest local-model case: conversational turn tracking across seams.

Evidence:
- `src/seiyuu/attribute/providers/base.py:81-82,130-133 — context blocks rendered as plain [id]\ntext with no speaker info`
- `src/seiyuu/attribute/pipeline.py:172-218 — by_block already holds resolved (speaker=character-id) segments for preceding chunks`
- `src/seiyuu/attribute/pipeline.py:8-9 — registry threads across chunks outside the cache key, the exact precedent for this`
- `src/seiyuu/settings.py — attribution_chunk_overlap_blocks: int = 2 (small window, so identity anchoring matters)`

> **Verifier:** Confirmed not implemented: context blocks render as plain `[id]\ntext` via _render_blocks (base.py:82) selected at base.py:131/157 with no speaker info. pipeline.py's by_block (lines 172, 217-218) holds resolved segments (speaker = character id) for all earlier chunks in the chapter by the time the next chunk runs — including cached chunks, since resolve_chunk runs on cache hits too — so leading-context speakers are available; ids map to names via the registry that is already passed to attribute_chunk. The registry-threading precedent (pipeline.py:8-9, cache key deliberately registry-free) is accurately cited, and overlap_blocks defaults to 2 (settings.py:60). One structural note the proposal implicitly handles: chunks carry leading AND trailing context (chunking.py:28) and only leading can be annotated, which is what it proposes. Requires plumbing a per-block speaker map into attribute_chunk plus a prompt v7 bump; medium effort, medium impact on the genuine seam turn-tracking failure mode.

### Thought Phase 2b: cue-verb candidate nomination for books that don't italicize thoughts
*Impact: medium · Effort: medium*

Thought candidates come exclusively from italic runs (emit_thoughts passes b.italic_spans; the v6 prompt only confirms pre-located italics), so books that set interior monologue in plain prose — very common — emit zero THOUGHT segments and the entire thought-voice render path (VoiceAssignment.thought_voice_id) stays inert. Add a second deterministic nominator: a prose sentence adjacent to a thought cue (', he thought', 'she wondered', 'X asked herself') becomes a candidate through the existing candidate_id/ThoughtVerdict confirm machinery — offset-preserving sentence sub-split exactly like the italic sub-split, model still only confirms and names the thinker, never slices text. Ships as prompt v7 so the cache key separates it cleanly; precision is preserved by the same confidence floor and prefer-false prompt rule.

Evidence:
- `src/seiyuu/attribute/providers/base.py:203-207 — candidates generated only from b.italic_spans`
- `src/seiyuu/attribute/spans.py:122-159 — thought_candidate_spans sub-splits prose only at italic runs`
- `prompts/attribution/v6.md:94-108 — cue verbs mentioned only as thinker evidence, never as a candidate source`
- `SPEC.md:345-351 — thoughts open question; memory: 'Phase 2b cue detection deferred'`
- `src/seiyuu/attribute/models.py:164-183 — ThoughtVerdict machinery is candidate-source-agnostic, ready to reuse`

> **Verifier:** Confirmed deferred, not built: candidates are generated exclusively from b.italic_spans (base.py:203-208), thought_candidate_spans sub-splits prose only at italic runs (spans.py:122-159), and prompts/attribution/v6.md:94-108 uses cue verbs solely as thinker EVIDENCE ('are good evidence for the thinker'), never as a candidate source. The ThoughtVerdict confirm machinery is genuinely candidate-source-agnostic — verdicts are keyed by candidate_id against the deterministically generated known_ids set (base.py:277-287, models.py:164-183) — so a cue-verb nominator slots in without touching the reconstruction guarantee (sub-splitting preserves the concatenation invariant). A prompt v7 cleanly separates the cache. Benefit is real but scoped: it activates the thought-voice path (opt-in emit_thoughts) for non-italicizing books; precision risk is higher than italics but the confidence floor + prefer-false rule are the existing mitigations. Medium/medium stands.

### Seed the registry from attribution.json on --chapter subset re-runs
*Impact: medium · Effort: small*

attribute_book always starts with an empty CharacterRegistry, so a `attribute --chapter 12` re-run prompts the model with an empty 'Known characters so far' section: it re-invents name forms ('Lizzy' vs 'Elizabeth Bennet'), mints duplicate registry entries, and loses gender/description metadata that guides both attribution and later casting. _merge_partial_attribution unions registries afterward but by then the damage (divergent ids/names) is done. When `chapters` is a subset and attribution.json exists, prime the registry from its stored registry (read-only seed) — cache keys are unaffected since the registry is deliberately not a key component. This also softens the adjudication-skip-on-subset limitation because fewer duplicates get minted in the first place.

Evidence:
- `src/seiyuu/attribute/pipeline.py:155 — registry = CharacterRegistry() unconditionally`
- `src/seiyuu/services/attribution.py:129-152 — _merge_partial_attribution merges only AFTER the run`
- `src/seiyuu/services/attribution.py:214-218 — adjudication skipped on subsets precisely because the registry is partial`
- `src/seiyuu/attribute/pipeline.py:8-9 — cache key intentionally has no registry component, so seeding is cache-safe`

> **Verifier:** Confirmed: pipeline.py:155 constructs `registry = CharacterRegistry()` unconditionally and attribute_book has no seed parameter; _merge_partial_attribution (services/attribution.py:129-152) unions registries only AFTER the run, by which point a subset re-run has already prompted with an empty 'Known characters' section and can mint divergent ids/name forms; adjudication is skipped on subsets for exactly this partial-registry reason (services/attribution.py:213-218). Cache safety checks out — the ChunkCacheKey deliberately has no registry component (pipeline.py:8-9), and the registry already varies run-to-run by design, so seeding changes nothing about key semantics. Not implemented anywhere; fix is loading the existing attribution.json registry in run_attribution when chapters is a nonempty subset and passing it through a new attribute_book parameter. Impact limited to the subset re-run workflow but prevents duplicate-character minting that is painful to clean up afterward; small effort is accurate.

### Grow the casting trait vocabulary beyond {young, deep}
*Impact: low · Effort: small*

The whole casting trait system — keyword scan, LLM caster hints, and voice-trait tags — knows exactly two traits, and only 8 of the pool presets carry any tag, so for most characters the tie-breaker is a no-op and casting degenerates to id-order pool assignment. Kokoro's own preset descriptions already encode a richer taxonomy (warm, breathy, bright, crisp, soft, energetic, formal), so adding 3-5 tags (e.g. warm/bright/soft/crisp) to KNOWN_TRAITS, the preset tables, the _wants keyword scan, and a caster prompt v2 is mostly data entry. It is structurally safe: hints only reorder the greedy pick among already-distinct candidates, so distinctness and determinism cannot break (documented invariant), and the drift test already asserts trait tables stay a subset of _POOLS.

Evidence:
- `src/seiyuu/voices/casting.py:46 — KNOWN_TRAITS = frozenset({'young', 'deep'})`
- `src/seiyuu/voices/casting.py:38-39 — _YOUNG/_DEEP tag only 8 presets total`
- `src/seiyuu/engines/kokoro_engine.py:28-56 — _DESCRIPTIONS already describe warm/breathy/bright/crisp/soft voices`
- `prompts/caster/v1.md:9-12 — LLM caster can only express the two tags`
- `src/seiyuu/voices/casting.py:69-81 — hints are tie-breaker-only, so expansion cannot affect collision-freeness`

> **Verifier:** Evidence accurate: KNOWN_TRAITS = frozenset({'young', 'deep'}) at casting.py:46, _YOUNG/_DEEP tag exactly 8 presets (casting.py:38-39), kokoro_engine._DESCRIPTIONS (lines 28-57) already describe warm/breathy/bright/crisp/soft/energetic/formal voices, prompts/caster/v1.md can only express the two tags, and the tie-breaker-only structural safety claim is correct (casting.py:69-91, 159-168 — hints only reorder greedy picks among already-distinct candidates). Nothing richer exists. But impact regraded to LOW: young/deep already cover the biggest audible fit gaps (age and depth); the new tags are subtler timbre refinements, character descriptions from attribution rarely contain deterministic timbre keywords (the benefit mostly flows through the opt-in F4 LLM caster), and changed tie-breaks reshuffle voice picks on any re-cast, churning settings_hash for previously cast books. It is safe, cheap data entry — just a modest aesthetic nudge, not an attribution-quality gain.

## Backend architecture & robustness

### GPU discipline is process-local: CLI + server concurrently can put two heavy models on the 8GB card
*Impact: high · Effort: medium*

GpuResourceManager serializes heavy consumers with a threading.Lock behind an lru_cache singleton, so it only protects ONE process. If uvicorn is running (a render/warmup leaves an engine lazily resident) and the user runs `seiyuu render`, `seiyuu attribute`, or `voice audition` from a terminal, the CLI process loads a second multi-GB model onto the same card — exactly the OOM/CPU-spill failure the one-heavy-model rule exists to prevent. The cross-process file_lock primitive already exists (used for cloud_voices.json and edits.json). Add a data_dir/gpu.lock acquired for the duration of each heavy stage run (run_attribution and the render loops have clean boundaries), failing with an actionable 'another seiyuu process holds the GPU' error; the nuance is lazy residency — a server holding VRAM idle after a render should still refuse the CLI (truthful refusal beats OOM), so the server-side hold should span gate.hold, not just the stage body.

Evidence:
- `src/seiyuu/gpu/manager.py:31-43 — threading.Lock only; acquire/unload has no cross-process awareness`
- `src/seiyuu/gpu/manager.py:68-70 — lru_cache get_gpu_manager() is a per-process singleton`
- `src/seiyuu/api/main.py:6-8 — docstring warns only about duplicate SERVER processes (--reload/workers>1), not a concurrent CLI`
- `src/seiyuu/cli.py:1182-1191 — CLI audition acquires its own process's manager and loads an engine unconditionally`
- `src/seiyuu/repository/lock.py:1-10 — cross-process msvcrt/flock file_lock already exists but is used only for cloud_voices.json / edits.json`

> **Verifier:** Every cited line checks out: gpu/manager.py:31-43 is a threading.Lock behind an lru_cache singleton (68-70), api/main.py:6-8 warns only about duplicate SERVER processes, cli.py:1182-1191 (audition) plus the CLI render/attribute paths acquire their own process's manager unconditionally, and repository/lock.py's msvcrt/flock file_lock is used only by voices/cloud.py and services/attribution.py (grep confirms no gpu.lock anywhere). No CLI code checks for a running server. The failure is real: a server that warmed or rendered leaves an engine lazily resident (handlers.py warmup explicitly documents 'the model STAYS resident'), and a concurrent CLI render/audition loads a second multi-GB model, OOMing or CPU-spilling the 8GB card — the exact violation CLAUDE.md's top rule exists to prevent, and both entry points are first-class in this project. One correction to the sketch: spanning the server hold across gate.hold is still insufficient, because warmup residency deliberately outlives gate.hold (handlers.py:110-112); the lock must track GpuResourceManager residency (acquire → free_all), which pushes effort to a solid medium.

### jobs.db has no retention: terminal rows (including never-reaped WARMUP rows) accumulate forever, and crashed uploads orphan data/uploads dirs
*Impact: low · Effort: small*

The only deletion in JobStore is delete_jobs_for_book, called solely when a book is deleted. Every render/attribute/assemble/master row for a book you keep lives forever, and WARMUP rows can NEVER be reaped because their book_id is the 'engine:{id}' overload no book deletion matches. Separately, POST /books writes uploads to data/uploads/{token} and cleans up only in a request-scoped finally with ignore_errors=True — a crash mid-ingest (or a Windows sharing violation) strands the directory permanently. Add startup housekeeping right after reconcile_startup: prune terminal job rows older than N days (or keep last K per book+kind) and sweep stale data/uploads entries. Both changes are contained to JobStore + the lifespan and require no schema change. (Related trivial cleanup: JobKind.INGEST is a dead enum value — nothing enqueues it and build_handlers registers no handler for it.)

Evidence:
- `src/seiyuu/repository/jobs.py:334-346 — delete_jobs_for_book is the only DELETE in the store, terminal-only and book-scoped`
- `src/seiyuu/repository/jobs.py:48-50 — WARMUP book_id is 'engine:{engine_id}', so book deletion never matches those rows`
- `src/seiyuu/api/routes/books.py:112-162 — upload dir data/uploads/{hex}; cleanup only via finally shutil.rmtree(ignore_errors=True); no startup sweep exists anywhere (grep for prune/sweep/cleanup in src/seiyuu finds none)`
- `src/seiyuu/api/handlers.py:282-288 — handler map lacks JobKind.INGEST; src/seiyuu/repository/jobs.py:43 defines it anyway`

> **Verifier:** Verified: delete_jobs_for_book (jobs.py:334-346) is the only DELETE in the store; WARMUP rows carry book_id='engine:{id}' (jobs.py:48-50, confirmed by _guard_any_active's own docstring in routes/books.py) so no book deletion ever reaps them; grep for prune/sweep/cleanup finds no startup housekeeping; JobKind.INGEST appears only at jobs.py:43 and is absent from the handler map (handlers.py:282-288) — dead enum confirmed. The upload claim is slightly overstated: the finally at routes/books.py:161-162 DOES clean up on ordinary exceptions, so orphans need a hard process kill or a silently-ignored rmtree failure (ignore_errors=True on Windows sharing violations — plausible, and voices.py:270 repeats the pattern). Impact honestly regrades to low: rows are tiny, GET /jobs caps at limit<=500 and book detail at 10, so accumulation is hygiene rather than a felt problem; the fix is genuinely contained (JobStore + lifespan).

### Queued jobs are canceled on restart instead of requeued, discarding durable user intent
*Impact: low · Effort: small*

reconcile_startup cancels every queued row with 'server stopped before the job started' because the queue handle lived in the dead process's memory — but the rows themselves are durable and carry the exact params the handlers re-parse. runner.start() could, after reconciling running rows, re-put still-queued job_ids into the worker queue in created_at order instead of canceling them, so a queued assemble/master behind a long render survives a restart. Caveat to design in: a queued RENDER's cost token (unconsumed until job start) may have expired across the restart — the job then fails loudly with the verbatim gate reason, which is acceptable, or requeue can be limited to free kinds.

Evidence:
- `src/seiyuu/repository/jobs.py:377-386 — queued rows are unconditionally CANCELED at startup reconcile`
- `src/seiyuu/jobs/runner.py:87-90 — start() reconciles then starts the worker; re-enqueueing surviving queued rows here is mechanical`
- `src/seiyuu/api/handlers.py:95,115,152 — handlers re-parse Job.params with the route's pydantic model, so a requeued row needs no in-memory state`
- `src/seiyuu/api/handlers.py:139-141 — the render handler already documents that verify/consume happens at job start, so a delayed start only risks a clean expiry refusal`

> **Verifier:** Evidence exact: reconcile_startup unconditionally cancels queued rows (jobs.py:377-386), runner.start() reconciles then starts the worker (runner.py:81-90), handlers re-parse Job.params with the route's pydantic models (handlers.py:95,115,152) so a requeued row needs no in-memory state, and the render handler verifies/consumes the cost token at job start (handlers.py:134-196) so a stale token refuses cleanly with the verbatim gate reason. Not already implemented — the current behavior is a documented deliberate tradeoff (runner.py docstring), which doesn't refute the improvement, but two things temper it: recovery today is one re-click (the canceled row is visible with a clear reason and its params intact), and restarts with a non-empty queue are rare in a single-flight single-user server. The paid-kind caveat brushes CLAUDE.md's no-automatic-paid-calls rule; the proposer's own free-kinds-only fallback handles it. Real but small convenience: impact regrades to low.

### Job handlers discard stage results: a finished job's progress_text is a stale mid-run line and the API never surfaces synthesized/cache-hit/timing stats
*Impact: low · Effort: small*

render_book/render_book_multivoice return RenderResult (synthesized, cache_hits, validation_failures, total seconds) and run_attribution returns the report — the CLI echoes rich summaries from these, but the API handlers drop the return values, so a succeeded job's progress_text is whatever mid-run tick came last ('chapter 12/40 ...'). The cheapest fix needs no schema change: have each handler emit a final ctx.progress summary line before returning (update_progress is guarded to state=running, which still holds at that point). That gives the UI per-job outcome stats and, combined with started_at/finished_at already on the row, a basic per-stage timing surface; a proper `result` JSON column would be an additive migration needing sign-off. Today /api/system exposes no timing or cache-hit-rate signal at all — the pipeline is otherwise a black box between enqueue and the manifest.

Evidence:
- `src/seiyuu/api/handlers.py:204-246 — render_book_multivoice/render_book return values are unassigned; same for run_attribution at handlers.py:120-132`
- `src/seiyuu/render/pipeline.py:159-167,423-428 — RenderResult carries synthesized/cache_hits/validation_failures/duration`
- `src/seiyuu/cli.py:242-249,370-377 — CLI prints these summaries; the API path loses them`
- `src/seiyuu/repository/jobs.py:325-332 — update_progress is guarded to running, so a final summary tick from inside the handler lands`
- `src/seiyuu/api/routes/system.py:49-97 — SystemStatus has no per-stage duration or cache-hit surface`

> **Verifier:** Verified end to end: handlers.py drops the return of run_attribution (120-132) and render_book/render_book_multivoice (204-246); RenderResult carries synthesized/cache_hits/validation_failures/total_audio_seconds (pipeline.py:158-168, built at 423-429); the pipeline's progress callback emits only per-chapter/per-segment lines (say() at 316/380/518/615 — no final summary), so a succeeded job's progress_text really is the last mid-run tick; the CLI prints the rich summary (cli.py:242-249, 370-377); update_progress is guarded to state=running (jobs.py:325-332) so a final ctx.progress line from inside the handler lands; routes/system.py exposes no timing or cache-hit signal. One overstatement: validation failures are NOT fully invisible on the API path — the manifest persists validation_failures and the review routes surface flagged segments — so the 'black box' framing is partly wrong, and synthesized/cache_hits/duration are the genuinely lost stats. A cheap, real observability win, but cosmetic: impact regrades to low.

### Segment cache is append-only with zero size visibility: stale WAVs/sidecars accumulate across every re-attribution, setting change, and lexicon edit
*Impact: medium · Effort: medium*

SegmentCache only ever puts; any change to text, voice, settings, seed, lexicon, or engine model version mints new key_hashes and permanently strands the old .wav/.json/.validation.json/.words.json quads in output/{book}/cache/. No endpoint or CLI reports cache size — book detail reports only m4b/mp3 bytes. Add (a) a storage report (per-book cache bytes + count of entries not referenced by the current manifest) on book detail or /api/system, and (b) an explicit opt-in prune (CLI + DELETE endpoint) that removes unreferenced FREE-engine entries. Critical constraint discovered in the code: detect_paid_artifacts treats the never-pruned cache sidecars as the AUTHORITATIVE paid-work signal for the deletion 402 gate, so a prune must either always keep paid-engine sidecars or route through the same confirm_paid flow.

Evidence:
- `src/seiyuu/render/cache.py:73-129 — put/get only; no eviction or size API`
- `src/seiyuu/services/deletion.py:66-70,97-101 — 'the cache is content-addressed and never pruned' and the paid-deletion gate RELIES on that property`
- `src/seiyuu/api/routes/books.py:186-211 — downloads report m4b/mp3/cover bytes only; nothing reports cache size`
- `src/seiyuu/render/pipeline.py:270,471 — cache lives at output/{book}/cache, one dir per book, shared across all renders/modes`

> **Verifier:** Verified: SegmentCache (render/cache.py:73-129) has only get/put/put_validation/put_words — no eviction or size API; any key ingredient change strands the old .wav/.json/.validation.json/.words.json quad; book detail reports only m4b/mp3/cover bytes (routes/books.py:186-211) and /api/system reports nothing; deletion.py:62-105 states verbatim that 'the cache is content-addressed and never pruned' and the paid-deletion 402 gate RELIES on the sidecars as the authoritative signal, so the proposer's critical constraint is accurate and load-bearing. One nuance the proposal missed (doesn't invalidate it): a targeted purge already exists — _purge_cached_segments (routes/voices.py:206-232) deletes one voice's entries on re-clone, including paid ones under an explicit 409-replace confirm — precedent for a confirm-gated prune, but no general size report or unreferenced-entry prune exists anywhere. Accumulation is material for this app: mono 24kHz 16-bit WAV is ~2.8MB/min, so each stranded full-book variant is roughly 1-2GB. medium/medium stands.

### Render preflight lives only in the API layer, so the CLI burns single-use cost tokens on doomed renders
*Impact: low · Effort: small*

_preflight_renderability (missing ELEVENLABS_API_KEY with paid segments, clone-consent failures, the ElevenLabs-library-voice-on-single-path trap) is a private helper in routes/render.py. The CLI's _pass_cost_gate calls verify_quote without a consume kwarg — the default is consume=True — so the token is spent immediately, BEFORE render_book hits the consent gate or a missing key; the user must then re-mint. Move the preflight into seiyuu/services (it is pure domain logic reading the library and settings) and call it from both the CLI gate path and the API, keeping the two entry points from drifting the way the module docstrings elsewhere warn about.

Evidence:
- `src/seiyuu/api/routes/render.py:111-161 — _preflight_renderability exists only in the API layer and is explicitly motivated by 'without this they surface only AFTER the handler burns the single-use approval token'`
- `src/seiyuu/cli.py:284-295 — CLI verify_quote passes no consume kwarg`
- `src/seiyuu/render/gate.py:159 — consume defaults to True, so the CLI consumes at gate time, before any consent/key failure`
- `src/seiyuu/render/gate.py:246-249 — a second use of the burnt token refuses; recovery is a full re-estimate + re-mint`

> **Verifier:** All evidence verified: _preflight_renderability is a private helper in routes/render.py:111-161 whose docstring gives exactly the cited motivation; the CLI's _pass_cost_gate calls verify_quote with no consume kwarg (cli.py:285-294, also reached by the multivoice path at 341-348) and gate.py:159 defaults consume=True with consumption as the last verification step (198-199), so the token is burned at gate time; the ElevenLabs client is lazy (_get_client raises on missing key only at synthesis, elevenlabs_engine.py:64-73) and the clone-consent gate runs inside render_book — both AFTER consumption on the CLI path; a reused token refuses (gate.py:246-250) forcing re-estimate + re-mint. Not already implemented for the CLI. Benefit is real but narrow: it only bites the estimate-cost --token → render --cost-token flow (the interactive --confirm-cost flow involves no token), and the loss is re-mint friction, never money. Effort small but note the helper raises ApiError, so the extraction needs a service-level error type plus mappings in both callers. Impact regrades to low.

### GET /api/books re-parses every book's full normalized.json on every poll
*Impact: low · Effort: small*

list_books -> get_book_status -> _read_book_meta json-parses the ENTIRE normalized.json (all chapters and blocks, megabytes for a novel) of every ingested book just to extract title/authors, and the library route runs this on each request — the repository docstring itself flags it as a deferred optimization. Write a tiny meta sidecar (title/authors/chapter count) at ingest time via write_normalized, or memoize per (path, mtime, size) in the repository, so library polling stays O(books) stat calls instead of O(total library bytes) of JSON parsing.

Evidence:
- `src/seiyuu/repository/books.py:11-13 — docstring: 'parses the whole normalized.json; a denormalized per-book summary is a deferred optimization'`
- `src/seiyuu/repository/books.py:80-86 — _read_book_meta does json.loads on the full file for two fields`
- `src/seiyuu/repository/books.py:116-124 — list_books calls get_book_status for every known id`
- `src/seiyuu/api/routes/books.py:75-89 — GET /api/books runs list_books per request, and the frontend refetches the shelf while jobs are active`

> **Verifier:** Verified: repository/books.py:11-13 itself flags the deferred optimization; _read_book_meta (80-86) json.loads the entire normalized.json for title+authors; list_books (116-124) runs get_book_status per known id; GET /api/books (routes/books.py:75-89) calls list_books per request, and no meta sidecar or memoization exists anywhere (write_normalized at ingest/epub.py:302 writes no summary; no lru_cache/mtime logic in the repository). Not implemented, evidence accurate. Impact honestly regrades to low for this single-user local app: a realistic shelf of a dozen books means tens of MB of JSON parse per poll at worst — wasteful CPU during active-job refetching but not user-visible latency until the library grows large. The (path, mtime, size) memoization variant is the right small fix since it avoids touching the write path or inventing a new on-disk format.

## Frontend UX & feature gaps

### Add the cover-art upload UI the app already promises
*Impact: medium · Effort: small*

PUT and DELETE /api/books/{id}/cover exist and are fully wired on the backend, but no component calls them — the only frontend use of covers is the GET in Listen's shelf tiles. Worse, the Listen shelf copy literally tells users to 'upload cover art in Render & Jobs', and RenderJobs.tsx contains no cover control at all. Add a small cover drop-zone/preview to the RenderJobs Outputs panel (or the Library card) using the existing BookDetail.cover field for current state.

Evidence:
- `src/seiyuu/api/routes/books.py:557 (PUT /books/{book_id}/cover), :591 (GET), :603 (DELETE)`
- `frontend/src/screens/Listen.tsx:254 — copy says 'upload cover art in Render & Jobs to make this prettier'`
- frontend/src/screens/RenderJobs.tsx — no reference to 'cover' anywhere in the file
- `frontend/src/api/types.ts:156 — BookDetail already carries cover: {content_type, bytes} | null, unused for upload`

> **Verifier:** Verified: PUT/GET/DELETE /books/{id}/cover exist at src/seiyuu/api/routes/books.py:557/:591/:603, and a repo-wide grep of frontend/src for 'cover' finds only the GET <img> in Listen's CoverTile (Listen.tsx:38), the promising copy at Listen.tsx:254 ('upload cover art in Render & Jobs'), and the unused BookDetail.cover field (types.ts:156). RenderJobs.tsx contains zero cover references. Not built anywhere. Impact is real beyond polish: the PUT docstring says it 'replaces the CLI's master --cover', so frontend-only users currently cannot embed cover art in the mastered m4b at all. Small effort — two mutations + file input + preview; the UI should also surface the existing 409 conflicting_job guard (books.py:545-554) when a master job is live.

### Let users HEAR voices inside the casting flow (Review screen)
*Impact: high · Effort: medium*

Casting is done completely blind: Review's VoicePicker is a plain select of 'name · engine' with no way to play a voice, so users must tab to Voice Studio, audition, memorize, and come back for every character. The backend already serves a cached last-take at GET /voices/{id}/audition.wav and instant kokoro previews at GET /engines/kokoro/preview; the gpu-borrow retry logic for live auditions already exists in Voices' AuditionControl and could be extracted. Add a ▶ button beside each picker (play last take if has_audition, else trigger the audition mutation), which also gives a cheap A/B compare — open two voices' last takes on the same standard audition line.

Evidence:
- `frontend/src/screens/Review.tsx:198-222 — VoicePicker renders only a TalkSelect, no audio affordance`
- `src/seiyuu/api/routes/voices.py:640 — GET /voices/{voice_id}/audition.wav (cached last take)`
- `src/seiyuu/api/routes/engines.py:157 — GET /engines/{engine_id}/preview for instant preset/blend demos`
- `frontend/src/screens/Voices.tsx:25-151 — AuditionControl already handles engine_cold/gpu_busy_retry/paid-confirm recourses, reusable`

> **Verifier:** Verified: VoicePicker (Review.tsx:198-222) is a bare TalkSelect of 'name · engine'; a grep of Review.tsx finds no audio element or audition/preview call (the only 'preview' hits are the smart-cast dry-run). GET /voices/{id}/audition.wav exists (voices.py:640-647, cached last take — free playback), kokoro preview exists (engines.py:157), and AuditionControl (Voices.tsx:25-151) already handles engine_cold/gpu_busy_retry/payment_confirmation_required recourses and the '▶ last take' cached player, so extraction is straightforward. VoiceOut.has_audition is already consumed (Voices.tsx:129). Constraint-clean: last-take playback touches no GPU, and the live-audition path already respects the borrow broker and paid-confirm gates. Casting is the core human loop, done blind today — high impact, medium effort stands.

### Structured render progress: segment counts, percent bar, and ETA
*Impact: high · Effort: medium*

A full-book render runs for hours, but JobOut carries only a free-text progress_text that the render loop updates once per chapter — inside a long chapter the transport bar can sit unchanged for 30+ minutes over an indeterminate CSS meter, and no percent or ETA exists anywhere. Emit per-N-segment progress (done/total is known upfront from speakable_blocks) as structured fields on the job row (done_units/total_units, plus started_at already exists), then render a real progress bar + rate-based ETA in TransportBar and JobRow. The jobs table is not one of the approval-gated schemas, but adding columns is still a migration — flag it. Bonus: the unreached GET /books/{id}/runtime-estimate could show '≈ X h of audio' in the ScopeRow.

Evidence:
- `src/seiyuu/api/schemas.py:42 — JobOut has only progress_text: str`
- `src/seiyuu/render/pipeline.py:316 — say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}") is the only in-loop progress call (next one is a validation failure at :380)`
- `frontend/src/app/TransportBar.tsx:104-105 — indeterminate <div className='meter'> + raw progress_text`
- `src/seiyuu/api/routes/books.py:297 — GET /books/{id}/runtime-estimate unreachable from any frontend code`

> **Verifier:** Verified: JobOut carries only progress_text (schemas.py:42), the DB jobs row likewise (repository/jobs.py:90/:112, update_progress at :325), and BOTH render loops emit progress once per chapter only (pipeline.py:316 single-voice, :518 multi-voice; the only other say() is the validation-failure line at :380). TransportBar renders an indeterminate striped meter + raw text (TransportBar.tsx:104-105). No done/total/percent/ETA exists anywhere. Jobs table is not on CLAUDE.md's approval-gated list, though the column add is a schema migration as the idea flags. Minor nit: the runtime-estimate bonus is half-wrong — BookDetail already carries runtime_estimate_seconds (types.ts:152), so no need to call the unreached GET /books/{id}/runtime-estimate; it just isn't displayed in ScopeRow. Effort is upper-medium: Job model + store + runner ctx (a second, structured progress channel beside the Callable[[str],None] sink) + both pipelines + TransportBar/JobRow.

### Validation-failure triage: filter/sort, jump-to-context, and a re-roll path
*Impact: high · Effort: large*

The Audio checks panel dumps every whisper failure in the whole book as one flat list — no chapter filter, no sort by score, no link to the segment in Review/Listen for context. Critically there is no fix path: the render pipeline reuses a cached wav even when its stored validation verdict is a failure, and the only cache purge that exists is purge-on-reclone, so a bad segment stays bad forever unless the user re-clones the voice or edits the lexicon. Add (a) small UI triage — group by chapter, sort by score, link to Review — and (b) a per-segment 'purge & re-synthesize' action (new endpoint deleting that SegmentKey's cache entry, then the existing chapter-scoped free render re-rolls it). Purging an entry does not change cache-key formats, so no approval-gated migration.

Evidence:
- `frontend/src/screens/RenderJobs.tsx:700-719 — flat validation.data.results list, no filter/sort/navigation`
- `src/seiyuu/render/pipeline.py:333-345 — cache hit returns the wav regardless of a failed stored verdict`
- `src/seiyuu/api/routes/voices.py:206 — _purge_cached_segments exists only for reclone; no per-segment purge route in render.py`
- `frontend/src/screens/RenderJobs.tsx:298-318 — ValidationFailure offers only expected-vs-heard text and a play button`

> **Verifier:** Evidence verified: flat unfiltered list (RenderJobs.tsx:700-719, ValidationFailure at :298-318 offers only diff + play); cache hits reuse the wav even with a stored failed verdict (pipeline.py:333-345 single-voice, :557-575 multi-voice); _purge_cached_segments exists only in the reclone path (voices.py:206); render.py's route list has no per-segment purge. BUT the proposed mechanism is incomplete: retry seeds are deterministic — attempt_seed = seed + i (pipeline.py:218) with voice seeds pinned in meta.json — so purge-and-re-render replays the exact same seed ladder the failed segment already lost with, reproducing the failure modulo GPU nondeterminism. A working re-roll needs a per-segment seed offset folded in before SegmentKey.build (the F2 emotion-override precedent at pipeline.py:545-548 shows value-level overrides are fine without touching the frozen key FORMAT), and that brushes the 'renders must use the pinned seed' rule, so it needs owner sign-off. The triage UI half is unconditionally real. Valid, but grade it knowing the fix path needs this extra design.

### Listen player table stakes: playback speed, ±15s skip, and resume position
*Impact: high · Effort: medium*

For an audiobook app the player lacks speed control (no playbackRate anywhere), skip-back/forward buttons, and resume — only volume and the spoiler frontier persist, so reopening Listen restarts the chapter from clip 0. Add a speed selector (0.8–2x, persisted like volume), ±15s keys on the transport, and persist {chapter, clip index, offset} per book to offer 'resume from where you left off' on load. All purely frontend: the clip model already supports seekClip(index, offset).

Evidence:
- `frontend/src/app/player.tsx:88-135 — full PlayerApi surface is load/toggle/seekClip/seekFraction/setVolume/clear; no rate, no relative seek`
- `frontend/src/app/player.tsx:30 — only volume is persisted (seiyuu.volume); Listen.tsx:167-169 persists only the frontier chapter number`
- `frontend/src/app/TransportBar.tsx:16-54 — transport has play/pause, seek meter, volume; no speed or skip controls`

> **Verifier:** Verified: the full PlayerApi is load/toggle/seekClip/seekFraction/setVolume/clear (player.tsx:88-135); grep confirms zero playbackRate anywhere in frontend/src; only seiyuu.volume (player.tsx:30/:128), the spoiler frontier (Listen.tsx:167-169), reading prefs, and theme are persisted; on reopen the chapter defaults to the first rendered chapter (Listen.tsx:70-71), so there is no resume at all. TransportBar's AudioTransport has only play/pause, seek meter, volume (TransportBar.tsx:16-53). Purely frontend, no constraint conflicts. Two implementation notes that keep effort at medium rather than small: playbackRate must be re-applied on every per-clip src swap (each clip reloads the single Audio element), and resume needs periodic persistence of {chapter, clip, offset} plus a restore path that overrides the chapter default.

### Review queue keyboard flow — and make 'next ▸' actually advance
*Impact: medium · Effort: small*

The review drainstrip's 'next ▸' always scrolls to lowConfInChapter[0], so clicking it repeatedly never moves past the first low-confidence segment — the review queue can't be walked. Track a cursor and cycle; then add keyboard shortcuts for the highest-friction loop in the app: j/k next-prev low-confidence segment, Enter to open the reassign popover on the current one, number keys for the most frequent speakers, u for undo-last. The only onKeyDown in the entire frontend today is the voice tag editor, so this is greenfield UI work with zero backend change (each fix is already a single record-edit POST).

Evidence:
- `frontend/src/screens/Review.tsx:481-487 — jumpToLowConf() always takes lowConfInChapter[0]; no cursor state`
- `frontend/src/screens/Review.tsx:540 — the 'next ▸' button is the only queue navigation`
- `grep of frontend/src for keydown handlers: only frontend/src/screens/Voices.tsx:191 (tag editor)`
- `src/seiyuu/api/routes/review.py:104-141 — reassign/set_emotion edits are cheap single POSTs suitable for a keyboard loop`

> **Verifier:** Code claims verified: jumpToLowConf always targets lowConfInChapter[0] (Review.tsx:481-487), next ▸ at :540 is the only queue navigation, and the sole keydown handler in the frontend is Voices.tsx:191. One framing correction: the queue is DESIGNED to drain — a manual reassign stamps confidence 1.0 (services/edits.py:201-207, comment literally says 'so the review queue drains'), so next ▸ does advance after each fix. The real gap is narrower but genuine: you cannot walk past a segment you judge CORRECT without recording a no-op edit, and there is zero keyboard support for the app's highest-frequency loop. Cursor + j/k/Enter/number-key shortcuts are greenfield frontend work confined to Review.tsx, with cheap single-POST edits confirmed at review.py:104-141. Valid at medium impact, small effort.

### Drive the voice dialogs from GET /engines instead of hardcoded lists
*Impact: medium · Effort: small*

GET /api/engines reports per-engine facts the UI never reads: paid, supports_cloning, weights_cached, resident. AddVoiceDialog and CloneDialog hardcode their engine options, so ElevenLabs is offered even with no key configured (the render screen already does this right for Anthropic), and a cold GPU engine surfaces only as an engine_cold refusal after the user tries to audition. Fetch the catalog once, disable/annotate unavailable engines (key missing, weights not yet downloaded → offer the existing warmup job proactively), and show a 'resident' dot so users know which engine is warm.

Evidence:
- `src/seiyuu/api/routes/engines.py:57-74 — EnginesOut with weights_cached + resident per engine; no frontend caller (grep shows only /engines/kokoro/voices, /engines/{id}/warmup, /engines/kokoro/preview)`
- `frontend/src/screens/Voices.tsx:574-583 — CloneDialog hardcodes chatterbox/indextts2/elevenlabs options`
- `frontend/src/screens/Voices.tsx:418-426 — AddVoiceDialog hardcodes kokoro/elevenlabs; no key-configured gating`
- `frontend/src/screens/RenderJobs.tsx:592-596 — precedent: anthropic option disabled when keys.anthropic_configured is false`

> **Verifier:** Verified: GET /api/engines returns paid/supports_cloning/weights_cached/resident per engine (engines.py:57-74) and has no frontend caller — hooks.ts hits only /engines/kokoro/voices (:544) and /engines/{id}/warmup (:617), plus the kokoro preview URLs in Voices.tsx. AddVoiceDialog hardcodes kokoro/elevenlabs (Voices.tsx:418-426 region, options at :422-426) and CloneDialog hardcodes chatterbox/indextts2/elevenlabs (:574-582) with no key gating, while RenderJobs already gates anthropic on keys.anthropic_configured (:590-597) and SystemStatusOut.keys.elevenlabs_configured already exists (types.ts:493, useSystem at hooks.ts:255). engine_cold is handled only reactively after a failed audition (Voices.tsx:77-86). All plumbing exists; this is wiring one query into two dialogs plus disabled/annotated options. Valid at medium/small.

### Responsive pass, prioritizing the Listen screen
*Impact: medium · Effort: medium*

The entire app has exactly one media query (prefers-reduced-motion); the shell is a fixed 220px-rail grid at 100vh and the main screens use fixed two-column grids (review: 440px + page, rjgrid: 380-460px + jobs), so on a narrow window or tablet the console overflows unusably. Since the backend is a LAN server, the realistic mobile use case is Listen — listening with read-along from a couch device. Add breakpoints that collapse the rail to a top bar, stack rjgrid/review columns, and make the Listen page single-column with the margin chips inline; the transport bar is already a flex row that mostly survives narrowing.

Evidence:
- `frontend/src/index.css:134 — the only @media rule is prefers-reduced-motion`
- `frontend/src/index.css:140 — .shell { grid-template-columns: 220px 1fr; height: 100vh }`
- `frontend/src/index.css:236 (.rjgrid minmax(380px,460px) 1fr) and :301 (.review 440px 1fr) — fixed desktop-only columns`
- `frontend/src/index.css:324 — .page grid with fixed 190px margin column used by Listen's read-along`

> **Verifier:** Verified: the only @media rule in the entire frontend is prefers-reduced-motion (index.css:134); .shell is a fixed 220px/100vh grid (:140); .rjgrid minmax(380px,460px) 1fr (:236); .review 440px 1fr (:301); .page is a 66ch + 190px margin-column grid (:324); and a grep for Tailwind responsive prefixes (sm:/md:/lg:/xl:) across frontend/src returns nothing — the app is genuinely desktop-only. The use case is credible: the backend is a LAN uvicorn server and Listen (shelf + read-along + transport) is exactly what a couch/tablet device would open. Effort is honest at medium — collapsing the rail, stacking two fixed two-column screens, and inlining Listen's margin chips is a real multi-screen CSS pass, not a one-liner.

## Developer workflow & tooling

### Add a minimal GitHub Actions CI (ruff + pytest + frontend gates) — 19 PRs have merged with zero automated checks
*Impact: high · Effort: medium*

There is no .github/ directory, so none of the gates CLAUDE.md mandates run automatically on the 19 PRs merged so far. A two-job workflow covers everything the default suite needs: (1) backend on ubuntu-latest — astral-sh/setup-uv with caching, `uv sync`, `uv run ruff check`, `uv run ruff format --check`, `uv run pytest` (the suite is CPU-only by design, GPU tests are deselected via addopts, and ubuntu-latest ships ffmpeg which test_assemble shells out to); (2) frontend — `npm ci`, `npx oxlint`, `npx vitest run`, `npx tsc -b`. The torch cu126 Linux wheel (~2.5GB) is the main cost, but setup-uv's cache amortizes it after the first run. This retroactively catches the class of bug the frontend-test PRs (#18/#19) found by hand.

Evidence:
- `F:\Projects\Seiyuu — `ls .github` returns nothing; repo has no CI config of any kind`
- `git log: merge commits for PRs #12-#19 (13664ae, d19506b, c34b694, 9cbe8f2, 2f5d007, 9f1773e...) all merged without checks`
- `pyproject.toml:58-59 — addopts "-m 'not gpu'" makes the default suite CI-safe (CPU-only); measured 881 passed in 103.32s locally`
- `tests/test_assemble.py:20,214 — real ffmpeg subprocess calls; ubuntu-latest runners include ffmpeg`
- `F:\Projects\Seiyuu\frontend\package.json:6-12 — lint/test/build scripts exist and are CI-ready`

> **Verifier:** Confirmed: no .github/ anywhere, no CI config, and every cited fact held up under inspection. Feasibility checks all pass: uv.lock (line 2343) contains the manylinux_2_28_x86_64 torch-2.6.0+cu126 wheel so `uv sync` resolves on ubuntu-latest; pyproject addopts (-m 'not gpu') deselects the 4 GPU tests (881/885 collected, verified via --collect-only); the default suite is genuinely offline — Kokoro/Chatterbox tests inject fake models (tests/test_kokoro_blend.py:61-63), whisper is faked (test_api_segment_words.py:53), and the IndexTTS-2 real-worker test is gpu-marked with skip guards for the gitignored index-tts/ clone (test_indextts2_gpu.py:30-33); tests/test_assemble.py:20,214 shell to real ffmpeg, which ubuntu runners preinstall. No paid-API or GPU constraint is violated: CI runs CPU-only fakes. Honest caveats keeping effort at medium rather than small: the cu126 Linux wheel bundles CUDA libs (~2.4GB download, large expansion — cache tuning needed), and the suite has only ever run on Windows, so the first Linux run may surface path/case issues that need fixing.

### Automate the torch-CUDA check: a default-suite tripwire test plus a `doctor` script
*Impact: high · Effort: small*

CLAUDE.md mandates manually running `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` after every dependency change, because chatterbox-tts once silently swapped in the CPU wheel — and every engine silently falls back to CPU (`"cuda" if torch.cuda.is_available() else "cpu"`), so a downgrade never errors, it just renders ~10x slower. Two small pieces: (a) a default-suite test asserting `torch.__version__.endswith("+cu126")` — the CPU wheel reports `2.6.0+cpu`, so the exact past incident becomes a red test with zero GPU needed; (b) a `scripts/doctor.py` (or `seiyuu doctor` CLI command) that additionally checks `torch.cuda.is_available()`, ffmpeg on PATH, and Ollama reachability — the trifecta of environment breakage this project documents.

Evidence:
- `CLAUDE.md 'Critical: GPU discipline' — mandates the manual post-install check; 'Gotchas Already Hit' documents the CPU-wheel incident`
- `pyproject.toml:7-9 — comment: 'MUST stay the CUDA build... never bump torch independently'`
- `src/seiyuu/engines/chatterbox_engine.py:41 and kokoro_engine.py:74 — silent CPU fallback means a downgrade passes all tests`
- `Grep of tests/ for '+cu|is_available': only tests/test_chatterbox_gpu.py:24, a gpu-marked skip guard — no tripwire in the default suite`

> **Verifier:** Cited evidence is exact: silent CPU fallback at src/seiyuu/engines/chatterbox_engine.py:41 and kokoro_engine.py:74 (`device or ("cuda" if torch.cuda.is_available() else "cpu")`), the only existing check is the gpu-marked skip guard at tests/test_chatterbox_gpu.py:24, and pyproject.toml:7-9 carries the 'MUST stay the CUDA build' comment. No tripwire exists in the default suite and no doctor script/CLI command exists (verified against cli.py's command list). One partial overlap the proposal missed: src/seiyuu/api/routes/system.py:66 already reports ffmpeg_available and probes Ollama reachability via ?probe=true — so the doctor script's ffmpeg/Ollama halves duplicate an existing API surface (though it requires the server running). The torch-build check — the actual documented incident — is checked nowhere. The tripwire is safe as designed: it asserts the version string, not cuda.is_available(), so it passes on a Linux CI runner (same +cu126 pin installs there) and on non-GPU machines. Impact stays high because the failure mode is a silent 10x degradation the project has actually hit, and the default suite runs constantly.

### Install git hooks for the required gates: ruff on pre-commit, pytest on pre-push
*Impact: medium · Effort: small*

The ruff/pytest gates are enforced only by discipline — .git/hooks contains only .sample files and no pre-commit framework config exists. For a solo dev on Windows, the lowest-ceremony version is a tracked `scripts/hooks/` directory plus `git config core.hooksPath scripts/hooks`: pre-commit runs `uv run ruff check` + `uv run ruff format --check` (sub-second), pre-push runs `uv run pytest` (103s, acceptable at push frequency; frontend vitest when frontend/ files changed). This makes the CLAUDE.md 'must pass before done' rule self-enforcing instead of memory-dependent, and pairs with CI (idea 1) as the fast local layer.

Evidence:
- `F:\Projects\Seiyuu\.git\hooks — only *.sample files, no active hooks`
- `grep for husky/pre-commit/lint-staged in frontend/package.json and pyproject.toml: no matches`
- CLAUDE.md Testing section — 'uv run ruff check and uv run ruff format --check must pass before done'; currently manual only
- `README.md:232-238 — six gate commands documented for humans, none wired to git`

> **Verifier:** Confirmed: .git/hooks contains only *.sample files, `git config core.hooksPath` is unset (exit 1), no husky/pre-commit/lint-staged in frontend/package.json or pyproject.toml, and .claude/settings.local.json contains only permissions — no hook automation exists at any layer. README.md:232-238 documents the gates as human-run commands only. The tracked scripts/hooks/ + core.hooksPath approach works on Windows (hooks run under Git Bash's sh). The 103s pre-push pytest cost is real but measured and bypassable with --no-verify. Impact is honestly medium, not higher: this is a solo-dev repo where the agent workflow already runs the gates, so hooks are enforcement insurance rather than a new capability — but they make the CLAUDE.md 'must pass before done' rule structural instead of memory-dependent.

### One command to run every gate: scripts/check.ps1 + a frontend `typecheck` script
*Impact: medium · Effort: small*

Pre-PR verification currently means six commands across two directories (pytest, ruff check, ruff format --check, npm test, npm run lint, npm run build). Add a single `scripts/check.ps1` (and/or a `check` uv script) that runs all of them and fails fast, plus a `"typecheck": "tsc -b"` script in frontend/package.json so type errors are catchable without paying for a full vite bundle — today `tsc` only runs inside `npm run build`, so a type error in a test-only refactor surfaces late or never. The hook (idea 3) and CI (idea 1) then just call this one script, keeping the three layers in lockstep.

Evidence:
- `README.md:232-238 — the six separate gate commands, split between repo root and frontend/`
- `frontend/package.json:8 — `"build": "tsc -b && vite build"` is the only path that typechecks; no standalone typecheck script`
- `F:\Projects\Seiyuu — no Makefile, justfile, or task runner present at root`
- `frontend/package.json:9,11 — lint (oxlint) and test (vitest) exist but nothing composes them`

> **Verifier:** Mostly confirmed with one evidence correction. Confirmed: scripts/ contains only Smoke.py, demo_section2.py, audition_indextts2_emotions.py — no check script, no Makefile/justfile at root; frontend/package.json:8 has `"build": "tsc -b && vite build"` and no standalone typecheck script; README.md:232-238 spreads the gates across two directories. Correction: the claim that a test-only type error 'surfaces late or never' is half wrong — frontend/tsconfig.app.json includes all of "src" and tests live in src/ (e.g. frontend/src/api/hooks.test.tsx), so `npm run build` DOES typecheck test files; the error surfaces late (only via the full tsc+vite build path), never is inaccurate. The benefit survives the correction: a fast standalone `tsc -b` script and a single composed check entry point that hooks (idea 3) and CI (idea 1) both call is real, small-effort glue. Impact honestly medium as the composition layer; low if the other two ideas don't land.

### Opt-in pytest-xdist to cut the 103s suite to a fraction on the dev machine
*Impact: medium · Effort: small*

The default suite is 881 tests in 103.32s, single-process, and almost entirely CPU-bound Python (the top items are an 8.5s EPUB parse and CLI-convert end-to-end runs). Adding `pytest-xdist` to the dev group and documenting `uv run pytest -n auto` should cut wall clock to roughly a quarter on a modern multi-core machine; tests already use tmp_path-isolated SQLite and in-process fakes, so they are parallel-friendly. Keep `-n auto` out of addopts (deterministic single-process stays the default and the CI/pre-push behavior), making this a zero-risk speedup for the inner loop.

Evidence:
- `Measured: '881 passed, 4 deselected in 103.32s' with slowest items 8.53s (test_ingest_lord_of_mysteries setup) and 6.89s (test_cli_convert full-book)`
- `pyproject.toml:36-40 — dev group is only httpx/pytest/ruff; no xdist`
- `tests/conftest.py:68-75 — tmp_path/session fixtures, no shared mutable state across files; fakes (tests/fake_provider.py, fake_engine.py) are in-process`

> **Verifier:** Confirmed: 881/885 tests collected (verified via --collect-only), dev group is only httpx/pytest/ruff (pyproject.toml:36-40, no xdist), and the parallel-safety claim holds — conftest.py uses tmp_path/session fixtures with no shared mutable state, API tests construct settings with data_dir=tmp_path/"data" (test_api_m6b1.py:21 et al.), engines/providers/whisper/aligner are all in-process fakes, and no test binds ports or touches shared repo dirs. Keeping -n out of addopts preserves the deterministic default, so no constraint is violated. One accuracy discount: the 'roughly a quarter' estimate is optimistic because module-scoped fixtures (the 8.5s pnp_epub parse in test_ingest_pride_and_prejudice.py:19-21) re-execute per worker under the default --dist load; --dist loadfile mitigates but the speedup will be less than linear. Still a real inner-loop win since CLAUDE.md mandates full-suite runs before any task is done.

### Add a snapshot-regeneration script for pnp_summary.json
*Impact: low · Effort: small*

tests/test_ingest_pride_and_prejudice.py compares parse_epub output against tests/fixtures/pnp_summary.json (62 chapters of title/block-count/word-count tuples), but nothing in the repo can regenerate that file — the next ingest-heuristic change means hand-editing a 5.6KB JSON or ad-hoc scripting. A 20-line `scripts/regen_pnp_snapshot.py` that parses the fixture EPUB, writes the summary JSON, and prints a diff against the current snapshot makes intentional snapshot updates a reviewed one-liner. Same pattern would serve any future ingest snapshot (e.g. a LoM summary).

Evidence:
- `tests/test_ingest_pride_and_prejudice.py:24-36 — snapshot read and compared; grep for 'pnp_summary' across the repo finds only this read site, no writer`
- tests/fixtures/pnp_summary.json (5619 bytes, last touched Jun 12) — hand-maintained since M1
- scripts/ contains only Smoke.py, demo_section2.py, audition_indextts2_emotions.py — no fixture tooling

> **Verifier:** Evidence confirmed: the snapshot is read and compared at tests/test_ingest_pride_and_prejudice.py:25-36, a repo-wide grep for pnp_summary finds only that single read site (no writer), and scripts/ contains no fixture tooling. Not already built. However, impact is honestly low, not medium: the fixture has not needed regeneration since Jun 12 — it survived even the M8 shared-ingest-core refactor (PR #17) untouched — so the pain the script relieves occurs rarely. When an ingest-heuristic change does land, the alternative (ad-hoc scripting a 62-chapter JSON) is genuinely annoying, so the 20-line script is worth its trivial cost, just infrequently exercised.

### Repo hygiene: retire scripts/Smoke.py (it violates the repo's own path rule) and sweep root strays
*Impact: low · Effort: small*

scripts/Smoke.py is tracked yet hardcodes bare relative paths ('Test.wav', 'out.wav') — the exact gotcha CLAUDE.md documents ('resolve paths from Path(__file__) or repo root... Test.wav.wav') — and it's fully superseded by the marked GPU smoke test (uv run pytest -m gpu, tests/test_chatterbox_gpu.py). Its droppings are still at the repo root: out.wav (330KB, Jun 12) plus an empty stray node_modules/ directory. Delete Smoke.py (or rewrite to argparse + output/ like the well-behaved audition_indextts2_emotions.py), remove out.wav and the empty node_modules/, and the root matches the standards the project preaches.

Evidence:
- `scripts/Smoke.py:8-10 — audio_prompt_path="Test.wav", torchaudio.save("out.wav", ...) — bare relative paths, cwd-dependent`
- CLAUDE.md Gotchas — '[Errno 2] No such file from relative paths... Watch for Explorer hiding extensions (Test.wav.wav)' — this script is that incident
- `Repo root listing: out.wav (330,320 bytes, Jun 12) and node_modules/ (empty, 0 entries) present; git ls-files confirms scripts/Smoke.py is tracked`
- `tests/test_chatterbox_gpu.py — the sanctioned GPU smoke path already exists; scripts/audition_indextts2_emotions.py:24-26 shows the correct REPO_ROOT pattern`

> **Verifier:** Evidence confirmed almost exactly: scripts/Smoke.py lines 8-10 contain audio_prompt_path="Test.wav" and torchaudio.save("out.wav", ...) — bare relative paths matching the CLAUDE.md gotcha verbatim; git ls-files confirms it is tracked; tests/test_chatterbox_gpu.py is the sanctioned superseding GPU smoke path; out.wav (330,320 bytes, Jun 12) sits at repo root. Two minor discounts: the root node_modules/ is not literally empty — it contains a stray .vite cache dir (still cruft, claim substantively right); and both out.wav and node_modules/ are gitignored (*.wav, node_modules/ in .gitignore), so they are local-machine clutter, not repo pollution — the only change other clones see is deleting the one tracked file. Low impact, trivially small effort, as proposed.

## Missed-by-the-panel finds (completeness critic)

### Chapter-range renders clobber the whole-book manifest (breaks 'continue from ch N', Listen, assemble, and master)
*Impact: high · Effort: small*

render_book_multivoice builds rendered_chapters ONLY from the requested chapter subset and atomically OVERWRITES manifest.json wholesale (src/seiyuu/render/pipeline.py:653-662; single-voice render_book does the same at :410-422). Nothing anywhere reads or merges the previous manifest. The frontend's advertised 'continue - next 10 from ch N' preset (frontend/src/screens/RenderJobs.tsx:148,191; frontend/src/lib/scope.ts:15-25) enqueues exactly such a subset render, so after rendering ch 1-10 then continuing with 11-20, the manifest contains only 11-20: GET /books/{id}/render (src/seiyuu/api/routes/render.py:464-491) reports ch 1-10 as unrendered (continueRange offers ch 1 again), Listen loses their read-along (Listen.tsx:67 uses useRenderSummary), assemble_book and master_book iterate manifest.chapters only (src/seiyuu/assemble/pipeline.py:224, :352) so a subsequent master produces an m4b MISSING chapters 1-10 even though their WAVs sit in cache. Fix: when chapters is a subset, load the existing manifest and merge by chapter index (union voices_used, recompute validation_failures), refusing on mode/assignment mismatch. No frozen-schema change needed.

### No inter-voice loudness matching: multi-voice chapters ship with per-speaker volume jumps
*Impact: high · Effort: medium*

Assembly concatenates raw cached segment WAVs with silence gaps (src/seiyuu/assemble/pipeline.py:98-134) and the only loudness correction is two-pass loudnorm with linear=true — a SINGLE gain applied to the whole chapter/book (:137-147, :232-234, :375). to_canonical only downmixes/resamples/clamps (src/seiyuu/engines/audio.py:39-62), and grep confirms no RMS/LUFS/gain code exists anywhere in render or engines. So a quiet Chatterbox/IndexTTS-2 clone (clones track their reference.wav level) sitting next to a hot Kokoro preset narrator inside one paragraph keeps its full loudness offset in the final m4b — the exact artifact multi-voice audiobooks live or die on. Fix: measure per-segment loudness at assembly (pyloudnorm or simple RMS in numpy, optionally cached as a {key_hash}.loudness sidecar like the existing .words.json pattern in render/cache.py:117-129) and apply a per-segment gain toward the chapter target before concat. In-memory only — cache WAVs and the frozen SegmentKey untouched.

### No backup/export of truth data — casting, edits, lexicons, and consent-attested references are one disk failure from gone
*Impact: high · Effort: medium*

Every data root defaults to the repo checkout (src/seiyuu/settings.py:22-27: books/, output/, voices/, data/ under REPO_ROOT), and a repo-wide grep for backup/export/restore finds nothing. Irreplaceable truth — voices/{id}/reference.wav + meta.json (consent hash, pinned seed), books/{id}/{normalized,attribution,edits,lexicon}.json (hours of LLM attribution plus hand review), output/{id}/assignments.json (casting), data/series.json, and PAID ElevenLabs segment WAVs the API itself refuses to delete without confirm_paid (routes/books.py:262-270) — is interleaved with multi-GB regenerable caches (output/{id}/cache is already 186MB for a partial render; conds .pt; HF weights). Moving machines or restoring after failure requires knowing exactly which files matter, and .gitignore has already silently swallowed source here once. Fix: a `seiyuu export [--book id] / import` pair that archives truth-only files (SPEC already classifies truth vs disposable cache per file), plus a README section naming the truth set.

### No production serving of the frontend — the shipped app is two dev servers, and README's --reload advice contradicts main.py's own warning
*Impact: medium · Effort: small*

The FastAPI app mounts zero static files (src/seiyuu/api/main.py:111-122 registers only error handlers and /api routers; repo grep for StaticFiles matches only uv.lock), so the ONLY way to use the UI — including for the owner, daily — is `npm run dev` with Vite proxying /api (frontend/vite.config.ts:10). Shipping to another user is impossible without a Node toolchain. Worse, README line 193 instructs `uvicorn seiyuu.api.main:app --reload` while main.py:6-8 states --reload is unsupported because a duplicate process 'would reconcile-kill live job rows and break every process-local primitive' — a code reload mid-render kills the job. Fix: `npm run build` output mounted via StaticFiles with an SPA fallback, a `seiyuu serve` CLI command (pyproject.toml:32-33 already defines the entry point) that runs one uvicorn worker without reload, and a README correction.

### The money-spending local API has zero Host validation or auth — DNS rebinding gives any website full control
*Impact: medium · Effort: small*

create_app adds no TrustedHostMiddleware, no CORS policy, and no auth of any kind (src/seiyuu/api/main.py:111-122; repo grep for TrustedHost/Authorization/Bearer/APIKey finds nothing). Every guard on the paid paths is just a request field an attacker can supply: confirm_paid=true on POST /attribute (routes/books.py:351-357), the cost-quote mint + token flow (the attacker calls the same endpoints), and confirm_paid=true on DELETE /books which discards paid artifacts (routes/books.py:248-270). A malicious page using DNS rebinding (browser resolves attacker.com to 127.0.0.1:8000; uvicorn happily serves any Host) gets full same-origin API access — enqueue Anthropic/ElevenLabs spends up to render_max_usd, delete books, exfiltrate the library; cross-origin multipart form POSTs (no preflight) can also hit ingest/cover uploads directly. Fix is a few lines: TrustedHostMiddleware(allowed_hosts=['localhost','127.0.0.1']) kills rebinding (rebound requests carry the attacker's hostname), optionally plus a static bearer token the Vite proxy injects.

### Ingest throws away the EPUB's embedded cover, then asks the user to upload cover art manually
*Impact: medium · Effort: small*

The pipeline's only contact with covers at ingest is SKIPPING them: SKIP_NAME_TOKENS drops cover/titlepage spine items (src/seiyuu/ingest/epub.py:58-62). Yet the product wants a cover everywhere downstream — PUT/GET /books/{id}/cover exists (src/seiyuu/api/routes/books.py:557-600), the M6c shelf renders books by cover, and master embeds it in the m4b (assemble/pipeline.py:276-289) — so the user must manually hunt down art that is almost always already inside the EPUB they uploaded (ebooklib exposes the OPF cover item directly). Fix: during parse_book, extract the declared cover image and write it to output/{id}/cover.jpg|png through the exact atomic write + magic-byte discipline upload_cover already implements; the shelf, BookDetail cover field (books.py:204-211), and m4b pick it up with zero further UI work. Also closes half the panel's 'cover upload UI' item for free.

### Whole-book JSON artifacts are the unit of every read — the in-repo 1,432-chapter book shows where it falls over
*Impact: medium · Effort: medium*

A real book in this repo (books/lord-of-the-mysteries-web-novel-novel-b573b0d1) has 1,432 chapters and a 27.5MB normalized.json. Beyond the panel's GET /api/books item, the per-REQUEST hot paths also re-read and re-validate entire book-scale artifacts: segment_browser parses the FULL render manifest and the full attribution report with edits replay on every chapter view (src/seiyuu/api/routes/books.py:428-441 plus effective_report), attribution_report loads everything and filters after (books.py:396-409), and GET /books/{id}/render re-sums duration_seconds over every segment of every chapter on each Listen/RenderJobs poll (routes/render.py:468-476). Fully attributed and rendered, this book means ~50-100MB of pydantic re-validation per chapter click. Cheap fix, no schema change: an mtime-keyed process cache for parsed normalized/attribution/manifest models in the repository layer (the API is the only writer, single process by design per main.py, so invalidation is trivial). Per-chapter artifact sharding is the deeper fix but touches frozen formats — flag for approval, large effort.

## Killed by verification

### Cache segment duration so fully-cached re-renders skip per-hit sf.info file opens `[render-perf]`

Evidence citations are accurate (sf.info + get_validation per hit at pipeline.py:334-338/557-562; sidecar written at cache.py:100-101) but the benefit is illusory as designed: the hit path does NOT currently read the {key_hash}.json sidecar, so 'add duration to the sidecar and read it on hit' replaces one tiny file open (sf.info parses only the WAV header) with another tiny file open (open+parse the JSON sidecar). Per-hit syscall count is unchanged — cache.get's is_file stat and the validation-sidecar access remain either way — so on the very scenario it targets (slow disks where opens dominate), the opens it claims to eliminate are merely relocated to a different file. Making a cached rebuild 'near-instant' would require consolidating per-segment metadata into one read (the SQLite index cache.py's own docstring already plans, or durations persisted in the manifest), which is a different design than proposed.

### Bound the cache blast radius: content-defined chunk boundaries and id-free content hashes (needs cache-key approval) `[attribution-perf]`

The cited code is real (chunking.py:45 hashes (b.id, b.text); ingest/common.py:184 positional ids; greedy walk from block 0; chapter_index in the PK) but the benefit is illusory because the idea missed how book identity works: book_id embeds the sha256 of the SOURCE FILE — epub.py:248-249 `sha = sha256(epub_path.read_bytes()); book_id = f"{slug(title)}-{sha[:8]}"`, and pdf.py:367 does the same. Editing one paragraph changes the file bytes, which changes book_id, which points at a brand-new books/{book_id}/ directory with an EMPTY attribution.db — the old cache is never consulted at all, so content-defined boundaries and id-free hashes with block-id remap would salvage nothing for the edit-and-re-ingest workflow that motivates the whole design. There is also no in-app path that edits source text under a stable book_id (the edits overlay in services/edits.py rewrites attribution output, never blocks). The only surviving scenario is a parser upgrade re-splitting chapters of the byte-identical file — rare, one-time, and not worth a cache-key migration that itself orphans every existing row and needs CLAUDE.md approval. Fixing this for real would additionally require content-independent book identity, a much larger change than proposed.
