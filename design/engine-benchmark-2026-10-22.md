# Engine benchmark baseline — RTX 3080 Ti 12GB (2026-10-22)

First measured baseline for all three local engines on the target card. Produced by
`uv run python scripts/benchmark_engines.py --use-test-reference --consent --validate` at commit
0e9e03a (torch 2.6.0+cu126, driver 617.42). Raw output: `output/benchmarks/20261022-164639/`
(not committed).

Input: the 10 bundled segments (5 short dialogue lines, 5 ~350–410 char narration paragraphs),
one untimed warm-up synth per engine. Cloning engines used the IndexTTS-2 test clip
(`index-tts/reference.wav`), consent attested for that run only.

| Engine | RTF agg | RTF p50 | RTF p90 | Cold load | Peak VRAM (nvidia-smi Δ) | Torch peak | Same-seed identical | Whisper pass (re-scored) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Kokoro 0.9.4 (`af_heart`) | 0.01 | 0.02 | 0.03 | 21.0 s | 1.9 GB | 1.0 GB | yes | 10/10 |
| Chatterbox 0.1.7 | 0.49 | 0.51 | 0.54 | 10.8 s | 4.9 GB | 4.1 GB | yes | 9/10 |
| IndexTTS-2 (fp16, subprocess) | 0.96 | 1.00 | 1.19 | 132.3 s | 8.9 GB | — | yes | 10/10 |

RTF = wall / audio; below 1 is faster than real time. Cold load for IndexTTS-2 includes the
worker boot. No OOMs or worker restarts.

## What it tells us

- **IndexTTS-2** is ~real time on this card (was ~1.3× on the 8GB 3070) and peaks at ~8.9 GB,
  leaving ~3 GB headroom. Short dialogue lines are slightly slower than real time (p90 1.19);
  per-call overhead dominates short segments.
- **Chatterbox** renders ~2× faster than real time in ~5 GB.
- **Kokoro** is ~100× real time; it is never the bottleneck.
- All three engines are deterministic for a fixed seed, so the pinned-seed rule holds.

## Validator bugs found by this run

The run's original whisper column read 8/10, 8/10 and 7/10. Every one of those "failures" was
word-perfect audio misjudged by the validator, identically across all three engines:

1. `difflib.SequenceMatcher` autojunk is on by default. On folded strings of 200+ chars it treats
   common characters as junk, so one small difference collapsed a near-perfect narration
   paragraph from 0.95 to 0.28.
2. Whisper writes digits ("12,000", "212") where the normalized text has words, so a short line
   failed at 0.81.

A failed check re-renders the segment up to 2 more times and then flags it, so long narration on
Chatterbox/IndexTTS-2 was burning GPU time and raising false review flags.

Two rounds of adversarial review showed that simply turning autojunk off and spelling out digits
was not enough (a whole invented sentence or a wrong number could then pass, and respelled names
or "June 21" failed). The scorer was rebuilt (`src/seiyuu/validate/scoring.py`):

- whisper's text goes through the normalizer's own rules ("Mr." → "Mister", "&", Roman numerals);
- numbers are compared by value ("1,200" = "twelve hundred", "911" = "nine one one", "1984" =
  "nineteen eighty-four", "June 21" = "June twenty-first"), and a number heard as a different
  number fails;
- an optimal character similarity (rapidfuzz) must reach 0.85, and an optimal word alignment must
  show no run of more than 4 wrong words, no lost or added negation, and no more than ~12% of the
  words wrong; a run whose letters closely match ("Her-my-oh-nee" / "Hermione") is a spelling
  difference;
- cached verdicts are re-judged from their stored transcript on the next render;
- every failure carries a plain-words reason.

The re-scored column above uses the rebuilt scorer. The one remaining failure is Chatterbox's
lighthouse paragraph, where whisper heard "its two under 12 steps" for "two hundred and twelve":
plausibly a genuinely slurred number (worth a listen:
`output/benchmarks/20261022-164639/audio/chatterbox/01_narration.wav`). The other two engines
read the same number cleanly.

## Next

- Re-run on real book chapters (`--book <id> --chapter N`) for retry rate on real prose.
- Use this as the comparison baseline for IndexTTS-2.5 and Chatterbox-Turbo (M9 step 2).
