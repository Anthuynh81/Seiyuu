# Qwen-first casting — design for sign-off (2026-11-08)

Step 5 of the Qwen rollout (steps 1–4 merged: engine #44, gender back-fill #45, Qwen clone
default #46, "describe a voice" #47). Goal from the owner: **Qwen3-TTS is the main engine**,
so a book's cast should come out as designed Qwen voices that fit each character, not Kokoro
blends. Kokoro stays as the quick draft.

## What casting does today

- **Suggest cast** (Review → Speakers & cast) runs `cast_book`: every character gets a
  distinct Kokoro blend `{char_id}_auto`, family chosen by gender, with an optional LLM
  "trait hint" (young/deep) as a tie-breaker. Preview first, then apply; manual picks and
  series inheritance are kept as overrides.
- Rendering already mixes engines in one book. Each chapter's lines are grouped by engine,
  so a Kokoro narrator with Qwen characters costs one model swap per chapter, not per line.
- What it can't do: make a voice *fit* a character. A blend knows only gender, plus the
  young/deep hint.

## Ghost-story book, today (the test case)

50 chapters, 1,360 dialogue lines, 21 speakers. Two problems the new flow has to handle:

- **One person, two characters.** "New Employee" (456 lines) *is* the first-person
  protagonist Kim Soleum (91 lines); attribution split them. Both need the **same voice**.
- **Crowd roles.** "Bystander" has 101 lines from many different people, all in one voice.
  Ghost, Teenager and "???" have 1–11 lines each.

## Proposed flow

A second mode in the Suggest-cast panel: **"Design voices (Qwen)"**, the default, next to
**"Quick draft (Kokoro)"**, which is today's behaviour, unchanged.

### 1. Draft a voice description per character (local LLM, free, ~1 min)

The attribution model (gemma4, through the existing `AttributionLLM` seam with
schema-enforced output, like the LLM caster and alias adjudication) gets per character: name,
aliases, gender, age hint, the registry description, up to 8 of their lines, and up to 6
narration sentences that mention them. It returns:

| field | notes |
|---|---|
| `gender` | male / female / unknown. Fills only an empty gender, like the #45 back-fill |
| `age` | child / teen / young adult / adult / older adult / elderly |
| `voice_description` | ≤ 30 words, **voice only**: pitch, timbre, pace, attitude, accent. This becomes the VoiceDesign prompt |
| `sample_line` | an **index into the lines it was given**, never free text. The line each take speaks is verbatim from the book |
| `same_as` | optional: "this is the same person as character X" (e.g. New Employee → Kim Soleum). A suggestion only; the user confirms it |

Cached per (book, character evidence hash, model, prompt_version), the same rule as the
attribution cache. New prompt file `prompts/describer/v1.md`. Ollama is unloaded
(`keep_alive: 0`) before any Qwen job loads.

### 2. Review the descriptions (Review screen, a table)

One row per character, busiest first. Columns: line count, the description (editable), the
sample line (pick another of their lines), **same voice as…** (pre-filled from `same_as`),
and **keep current voice** (skip designing; manual picks and series voices default to this).
Nothing is spent until the user presses **Design N voices**.

### 3. Design the cast (one GPU job)

A `design_cast` job (subject = the book, deduped per book) runs the #47 design flow for every
selected character in turn, keeps the **recommended** take as `{char_id}_designed`
(CLONED qwen3tts, tags `["designed", book_id]`), and writes the assignment. Characters marked
"same voice as" are assigned the other character's voice; nothing extra is designed for them.
Each character's design id is recorded, so later the user can open it, listen to the other
takes, and swap in one click (the existing keep, plus re-pointing the assignment).

**Time on the 3080 Ti (~30 s per take):** 19 voices × 2 takes ≈ 20 min; × 3 takes ≈ 30 min.
Resumable: a canceled job keeps every finished take (the #47 behaviour).

### 4. Narrator

Options are in decision Q1. In every option the narrator is just another row in the table.

### 5. Crowd roles

A character whose name is a role rather than a person (Bystander, Staff, Ghost, "???",
Researcher) gets **one designed voice like everyone else**. Rotating voices per line inside
one "Bystander" is out of scope here; that's an attribution problem (split the crowd into
people), and this design doesn't make it harder later.

## What changes on disk (for sign-off)

- **No voice-model change.** Designed voices are ordinary CLONED `qwen3tts` voices
  (`reference.wav` + consent + optional transcript), exactly what #47 already writes.
- **No change to `assignments.json`, the attribution segment schema, or any cache key.**
- **New file** `output/{book}/cast_plan.json`: the drafted and edited descriptions, chosen
  sample lines, same-as links, skip flags, LLM provenance, and per-character design ids. It's
  a working file for the Review table, not a cross-stage contract, and it's never read by
  render.
- **New job kind** `design_cast`.
- **New prompt** `prompts/describer/v1.md`.
- **API:** `POST /books/{id}/cast-plan/draft` (job or sync, see the PR plan), `GET` / `PUT
  /books/{id}/cast-plan`, `POST /books/{id}/cast-plan/design` (the job). The route list test
  grows accordingly.

## Render-time consequences to know about

- Qwen renders at ~0.6× real time on this card, including speech checks (A/B: 4.3 min for
  7.3 min of audio). Kokoro is ~0.04×. A 10-hour book takes ~6 hours.
- Qwen speaks ~25% faster than Kokoro (≈184 vs 146 wpm). The design scoring already prefers
  takes in 140–210 wpm. No pace control is added here.
- Mixed books (a Kokoro narrator with Qwen characters) are fine. Loudness matching (#40)
  already evens out voices.

## PR plan

1. **Descriptions**: the describer (prompt, schema, cache), `cast_plan.json`, API + CLI
   (`seiyuu cast describe BOOK`), tested with a scripted fake LLM. Captured gemma4 output on
   the ghost book goes in as a fixture.
2. **Design the cast**: the `design_cast` job, same-as and skip handling, applying the
   assignment, swapping a take later. CLI `seiyuu cast design BOOK`.
3. **Review UI**: the mode switch in Suggest cast, the description table, progress, and
   swapping takes.

## Decisions needed

- **Q1 Narrator.** (a) Designed like the rest, from a "warm, clear audiobook narrator"
  description the user can edit. (b) Keep the user's current narrator voice (today: a Kokoro
  preset). (c) For first-person books, the narrator *is* the protagonist's voice.
  **Recommendation: (a) by default, with (c) offered when the narrator is the protagonist.**
- **Q2 Takes per character.** 2 (≈20 min for this book) or 3 (≈30 min), auto-picking the
  recommended one either way. **Recommendation: 2.** Every character can be re-listened and
  swapped later.
- **Q3 Small parts.** Design everyone, or give characters with fewer than ~5 lines a
  shared designed "extra" voice per gender. **Recommendation: design everyone**, with only 1
  take for characters under 5 lines (about 30 s each). Shared voices are what you complained
  about.
- **Q4 Default mode.** "Design voices (Qwen)" as the panel default, Kokoro as "quick
  draft". **Recommendation: yes** (your call from step 1: Qwen is the main engine).

## Decided (owner, 2026-11-08)

All four recommendations accepted:

- **Q1:** narrator designed by default, with "use the protagonist's voice" offered for
  first-person books.
- **Q2:** 2 takes per character, the recommended one kept.
- **Q3:** design everyone; characters under 5 lines get 1 take.
- **Q4:** "Design voices (Qwen)" is the default in Suggest cast; Kokoro is the quick draft.
