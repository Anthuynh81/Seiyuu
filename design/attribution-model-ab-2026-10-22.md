# Attribution model A/B on the 3080 Ti (2026-10-22)

The question was which local Ollama model should be the default for speaker attribution now that
the target card has 12 GB. The pipeline has defaulted to `qwen2.5:7b` since the 8 GB 3070, when
anything larger spilled to CPU.

## Method

- **Gold labels:** the Project Dialogism Novel Corpus (PDNC) annotations for *Pride and
  Prejudice* (CC-BY 4.0, github.com/Priya22/project-dialogism-novel-corpus). They were imported
  with `seiyuu eval gold-import-pdnc` (PR #37), which labeled 1,679 of the book's 1,759 quote
  spans. An independent check against PDNC's character offsets confirmed all 1,679 by position.
- **Harness:** `seiyuu eval run <book> --gold gold.jsonl --models a,b,... --chapters N-M`. Each
  arm attributes from scratch into its own scratch DB, with no cache hits, using prompt v5 (the
  current default) and the native `/api/chat` transport at `num_ctx` 8192.
- **Scoring:** the share of gold quotes whose predicted character resolves to the gold speaker.
  A predicted character that matches two gold speakers equally (for example "Bingley", which
  matches both Miss Bingley and Mr. Bingley) is scored wrong, so these numbers slightly
  understate every arm.
- **Slices:**
  - chapters 2–12 (423 quotes), the main A/B;
  - chapters 52–62 (355 quotes), a confirmation slice for the four leading arms, so the result
    doesn't depend on one stretch of the book.

## Results

### Chapters 2–12 (423 quotes)

| Model | Size on disk | Accuracy | Unattributed | s/chunk (mean) | Wall |
|---|---:|---:|---:|---:|---:|
| **gemma4:12b-it-qat** | 7.2 GB | **89.4%** | 2 | 23.7 | 808 s |
| gemma4:e4b-it-qat | 6.1 GB | 79.4% | 2 | 16.0 | 548 s |
| qwen3.5:9b | 6.6 GB | 76.1% | 24 | 15.5 | 530 s |
| qwen3.5:4b | 3.3 GB | 70.2% | 38 | 12.8 | 441 s |
| qwen2.5:7b (current default) | 4.7 GB | 68.8% | 56 | 10.2 | 352 s |
| qwen3:4b-instruct | 2.5 GB | 44.4% | 101 | 57.1 | 1,945 s |
| gemma4:26b-a4b-it-qat | 15 GB | — | — | — | timed out |

- **gemma4:26b-a4b-it-qat** does not fit in 12 GB. It spilled to CPU and its first request
  timed out, so it is not viable on the target card.
- **qwen3:4b-instruct** stalled on one chunk for 19 minutes and failed the schema 3 times.

### Chapters 52–62 (355 quotes)

| Model | Accuracy | Unattributed | s/chunk (mean) | Wall |
|---|---:|---:|---:|---:|
| **gemma4:12b-it-qat** | **92.1%** | 1 | 23.7 | 974 s |
| gemma4:e4b-it-qat | 83.4% | 4 | 17.1 | 705 s |
| qwen3.5:9b | 75.8% | 25 | 16.0 | 659 s |
| qwen2.5:7b | 72.7% | 34 | 11.9 | 490 s |

The ranking is the same on both slices, and the lead holds. Combined over 778 quotes:

| Model | Correct | Accuracy | Wrong or missing |
|---|---:|---:|---:|
| gemma4:12b-it-qat | 705/778 | 90.6% | 73 |
| gemma4:e4b-it-qat | 632/778 | 81.2% | 146 |
| qwen3.5:9b | 591/778 | 76.0% | 187 |
| qwen2.5:7b | 549/778 | 70.6% | 229 |

Gemma 12B leaves about a third as many quotes for a person to fix in Review as the current
default.

### By quote type (Gemma 12B vs qwen2.5:7b)

PDNC tags each quote by how its speaker is signalled in the text.

| Type | Meaning | Ch. 2–12 | Ch. 52–62 |
|---|---|---|---|
| Explicit | "said Elizabeth" | 98.2% vs 73.2% | 98.8% vs 76.5% |
| Anaphoric | "she replied" | 91.5% vs 73.4% | 91.1% vs 77.2% |
| Implicit | turn-taking only | 79.4% vs 61.8% | 89.4% vs 66.9% |

On qwen2.5, even quotes with an explicit "said X" tag were wrong a quarter of the time. Gemma
reads those almost perfectly. What it still gets wrong is mostly turn-taking in long two-person
exchanges, such as Elizabeth's lines given to Darcy in ch. 58–60, and Miss Bingley versus
Elizabeth in ch. 8–11. These errors are spread out rather than systematic.

### Unattributed quotes

qwen2.5:7b left 90 of 778 quotes with no speaker (about 12%), which is why many books showed a
large "unattributed" pile in Review. Gemma 12B left 3. That was a model limitation, not a
pipeline bug, so the planned investigation into unattributed quotes is no longer needed.

## Cost of switching

- **Speed:** about 2.3× slower per chunk (23.7 s vs 10–12 s). *Pride and Prejudice* is about 210
  chunks, so a full-book attribution goes from roughly 40 minutes to roughly 85. Attribution
  runs once per book and is cached, while render and review take far longer.
- **VRAM:** 8.0 GB at `num_ctx` 8192, `100% GPU` in `ollama ps`, so there is no CPU spill. The
  GPU manager still unloads it before any TTS engine loads, so the one-heavy-model rule is
  unchanged.
- **Cache:** the model id is part of the attribution cache key. Books already attributed with
  qwen2.5 keep their results; only new or re-run attributions use the new default.
- **Setup:** `ollama pull gemma4:12b-it-qat` (7.2 GB) on fresh installs. The setup checklist
  already prints the pull command for whatever the default is.

## Recommendation

Make `gemma4:12b-it-qat` the default attribution model, and keep `qwen2.5:7b` as the documented
fast or 8 GB fallback (`ATTRIBUTION_MODEL=qwen2.5:7b` in `.env`). `gemma4:e4b-it-qat` is the middle option for
smaller cards: about 81% accuracy at 6 GB.

## Caveats and follow-ups

- **One book.** Regency prose with heavy dialogue. A modern novel with first-person narration
  could rank differently. A second PDNC book would firm this up; the corpus has 22.
- **Prompt v5 was tuned against qwen.** A Gemma-tuned prompt might gain more, especially on
  implicit turn-taking. That would be a v7 prompt, measured with this same harness.
- **Alias drift is model-independent.** "Jane" and "Miss Bennet", and "Charlotte" and "Miss
  Lucas", end up as two characters in several arms, which would give one person two voices. The
  alias post-pass is deliberately conservative, so a family-name rule ("Miss <Surname>" = the
  eldest daughter) is a candidate improvement, with its own precision test.
- **Raw reports were not committed.** They are `eval_report.md` and `eval_report.json` in each
  run's output directory, and are regenerated by rerunning the commands above.
