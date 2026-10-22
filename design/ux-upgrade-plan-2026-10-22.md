# Seiyuu UI and Workflow Upgrade Plan (revised)

**Direction:** a hybrid. The base is the **Guided Book Pipeline**: a per-book stepper with smart auto-advance. Its first phase is the **Incremental Clarity** fix list, shipped on the current screens. A few ideas come from **Script-as-Editor Studio**: a chapter-1 pilot step, a sample book, fixing lines from inside Listen, error codes with retry, and cached auditions.

---

## 1. Diagnosis: why Seiyuu feels hard and confusing today

The live browser walkthrough **did not run** because the Claude-in-Chrome extension was not connected. Every finding below comes from reading the code. A short browser check should confirm Phase 1 before Phase 2 starts. The aborted run also left a Vite process listening on port 5173. Stop it by hand: find the PID with `Get-NetTCPConnection -LocalPort 5173`, then run `Stop-Process` on it.

Ranked by how much each one adds to the "confusing" feeling:

1. **The app is organised around its internals, not the user's job.**
   - The nav reads Library, Listen, Character Review, Pronunciation, Voice Studio, Series, Render & Jobs (`app/NavRail.tsx:6-14`). `NavRail.test.tsx:16-24` pins that order.
   - Attribution is the first real step, but it starts on the *last* nav item, inside a box tagged "not attributed" (`screens/RenderJobs.tsx:344-353`).
   - On a fresh book the mode defaults to single (`RenderJobs.tsx:41`) and is already "ready" (`RenderJobs.tsx:65`). The main button is "render single — free" (`render/EstimateCard.tsx:54`). A user who wants a multi-voice book gets pushed toward a single-voice render.
2. **Nothing tells you what to do next.**
   - After upload, the card shows `<slug-hash> · N chapters, M blocks` and offers only a "render & jobs" button (`Library.tsx:195-199`, confirmed by `Library.test.tsx:21-24`).
   - StageRail's "next: cast voices in Character Review" is plain text (`render/StageRail.tsx:17-30`).
   - Empty states are dead ends (`Review.tsx:175,203`, `RenderJobs.tsx:220`, `Lexicon.tsx:141`).
   - The CLI does the whole job in one command (`convert --multivoice --m4b`, `cli/ingest.py:176-195`). The UI needs about six trips across three screens.
3. **Failures and setup problems are invisible.**
   - `useSystem` calls `/api/system` without `?probe=true` (`api/hooks.ts:288-294`), and nothing reads `/api/engines`.
   - If Ollama is down, POST `/attribute` still queues the job (`routes/books.py:327-397` has no pre-flight), and the job fails later out of sight. `useLiveJobs` only polls queued and running jobs (`hooks.ts:56`). So the TransportBar returns to "console idle", the card says "no job active", and the reason is hidden behind "why? ▸" (`render/JobRow.tsx`).
   - The footer is hard-coded to "talkback v4 · console online" (`NavRail.tsx:46-50`), even when the backend is down and every request comes back as a bare 500 (`api/client.ts:20-27`).
4. **Casting is destructive and you can't hear it.**
   - "re-draft", "apply cast" and Series "apply inherited" send no overrides and no narrator. `draft_assignment` then rebuilds the whole map: the narrator resets to `narrator_<preset>` and every character goes back to `{char}_auto` (`services/assignment.py:104-175`; `Review.tsx:307-315,372-381`; `Series.tsx:113-119`). Manual picks are lost with no undo.
   - The primary "auto-cast" runs the collision-blind `hash` caster, because `strategy` defaults to `"hash"` (`api/schemas.py:334`).
   - "suggest cast" shows only counts (`review/SuggestCastPanel.tsx:69-99`).
   - The picker is a bare dropdown (`review/VoicePicker.tsx:19-35`), even though `AuditionRequest.text` accepts up to 500 characters (`schemas.py:625-627`).
5. **Every fix means crossing four screens.**
   - In Listen, speaker chips are plain spans, and the margin shows the raw `voice_id` (`Listen.tsx:377-382`).
   - Review keeps the chapter in `useState(1)` (`Review.tsx:42`), so it can't be linked to.
   - Spoiler-safe starts as `useState(true)` (`Review.tsx:95`).
   - Reassign, emotion and lexicon edits never mark audio as stale, because the casting hash covers only voice ids (`render/gate.py:87-102`).
6. **Customising a voice is a dead end.** After auto-cast the app says "tune them in Voice Studio" (`Review.tsx:394-398`). But `VoiceUpdate` allows only `name` and `tags`, with `extra="forbid"` (`schemas.py:599-608`). Per-engine `settings` exist in `VoiceMeta` (`voices/models.py:58`) and every engine honours them, yet nothing writes them.
7. **Long jobs are opaque and fragile.**
   - Progress is an endless stripe (`TransportBar.tsx:120`, `index.css:163`).
   - Auto-finish is a chain held in browser state (`RenderJobs.tsx:100-147`).
   - `mastered` just means the `.m4b` file exists (`repository/books.py:117`), so it stays green after the audio changes.
   - Partial renders look complete.
8. **Jargon and raw ids are everywhere.** Examples: signal path, attribute/assign/master, mint quote, fingerprint, sig, DRIFT, `conf 0.62`, `ch013_b0042`, `j_xxxx`, raw `voice_id`s, and error codes shown as tags.

---

## 2. What the best tools do

| # | Pattern | Product | URL |
|---|---|---|---|
| 1 | A linear setup wizard settles structure (which sections, the cover), then casting, then pronunciation, all before any audio is generated | ElevenLabs Audiobooks | https://elevenlabs.io/docs/eleven-creative/products/audiobooks |
| 2 | One question at upload: "Single cast or Multi cast?" | ElevenLabs Audiobooks | (same) |
| 3 | Each character's proposed voice is previewed **on that character's own dialogue**. Recasting asks for confirmation and shows how much audio it clears | ElevenLabs Audiobooks | (same) |
| 4 | Numbered core steps (green) kept visually separate from advanced tools (blue) | Alexandria | https://github.com/Finrandojin/alexandria-audiobook |
| 5 | A disabled Generate button states its *one* reason. A pinned footer shows per-chapter progress, ETA, a real Stop and resume | VoiceStudio | https://github.com/debpalash/VoiceStudio |
| 6 | A "Pronunciation check" queue grouped as names, places and numbers, with **play every occurrence** and fix-in-place | Google Play Books | https://support.google.com/books/partner/answer/14182676?hl=en |
| 7 | Ctrl+1..0 reassigns selected text to a character; speakers are colour-coded | Google Play Books | https://support.google.com/books/partner/answer/14183080?hl=en |
| 8 | A pronunciation fix plays before and after and offers "Apply to all instances" | KDP Virtual Voice | https://kdp.amazon.com/en_US/help/topic/GLD3SMMB6B99D5FW |
| 9 | A generated/not-generated bar beside each paragraph, generation history, lock-paragraph | ElevenLabs Studio | https://elevenlabs.io/docs/eleven-creative/products/studio |
| 10 | Regenerating a span gives Play / Retry / Keep / Revert | Descript | https://help.descript.com/script-editing/change-meaning |
| 11 | The machine aligns audio to text and builds a typed pickups queue; the human judges only the flagged spots | Pozotron | https://blog.pozotron.com/hidden-gems-the-pozotron-proofing-tool |
| 12 | Cast cards lead with a persona line; a sequel opens on "returning cast and newcomers" | Castwright | https://www.castwright.ai/features/ |

Only the VoiceStudio URL is corroborated in the repo (`design/voicestudio-assessment-2026-10-22.md`). Check the others before sharing this plan outside the project.

The common thread: one book, one ordered path, one primary action per step. Fix problems where you find them. Review a ranked list of likely problems instead of listening to everything. Show cost, time and what will go stale before the user clicks.

---

## 3. Target experience

### Rules every change keeps

- **The API is the only writer.** Step state, "next step", staleness, issues and flags are derived from server data or stored on the server. localStorage holds only conveniences: the last book opened, Listen position, dismissed hints and the spoiler-safe preference.
- **Paid calls need explicit approval.** They still require a signed quote minted from a fresh estimate. A paid whole-book render becomes one dialog by sending `confirm_full` together with `cost_token`. This needs no contract change:
  - `RenderParams` already has both fields (`schemas.py:445-446`).
  - The route checks `confirm_full` *before* the money dry-run (`routes/render.py:306-367`).
  - The threshold is already exposed as `limits.full_render_confirm_blocks` on `/api/system` (`routes/system.py:80`).
  - Retrying a paid job always needs a fresh quote.
- **Clone consent is unchanged.** It stays bound to the uploaded reference bytes. Replacing a clip means attesting consent again.
- **One heavy GPU model at a time.**
  - Every new synthesis goes through the existing audition gate or job runner: inline auditions, previews of unrendered lines, lexicon "hear it", and warm-up (a queued `WARMUP` job, `routes/engines.py:77-102`).
  - Playing already-rendered segment audio is file serving and needs no GPU.
  - Exclusive auditions are refused with `gpu_busy` while a job holds the GPU. A Kokoro audition may borrow a running Kokoro render's engine (M6c engine sharing).
  - The UI shows `gpu_busy` as "Recording in progress, try after it finishes". It never queues silently and never retries in a loop.
- **`["books"]`/`["book"]` query keys stay stable.**
  - New data gets new keys: `["system","probe"]`, `["engines"]`, `["jobs","recent"]`, `["issues",id]`, `["staleness",id]`.
  - `JobCompletionWatcher` only gains fetches and invalidations of those new keys.
  - New screens read the book id from the route but keep the same `["book", id]` key.

### Information architecture (end state after Phase 2)

```
NAV RAIL
  Books (/)                    shelf: cover, plain status, ONE next-step button; "Try the sample book"
    Recent: last 3 books       deep link to each book's current step (localStorage convenience)
  ── Tools ──
  Voice Library (/voices)      was Voice Studio
  Pronunciation (/lexicon)     every book's dictionary (per-book editing also lives in Proof)
  Series (/series)
  Activity (/activity)         every job (book jobs and engine:<id> warm-ups), failures kept, Retry
  Settings & Health (/settings)
  (●) Ready · Ollama ✓ ffmpeg ✓   live health pill, replaces "console online"
BOTTOM BAR: job pill (title · step · % · ETA · Stop)  |  mini-player (pause from any screen)

BOOK PROJECT  /book/:id/:step
  1 Import → 2 Speakers → 3 Voices → 4 Proof → 5 Produce → 6 Listen
  Each step shows done / now / needs attention / running / locked-with-reason, derived ONLY
  from server data in a pure lib/steps.ts. A "locked" step explains why and stays clickable.
```

Until Phase 3 builds the dedicated Proof workspace, each step body reuses an existing screen:

| Step | Existing screen used |
|---|---|
| Speakers | Review's roster and segments |
| Voices | Review's Cast panel, until the 2.9 board lands |
| Proof | Review's segment pane plus the Lexicon screen |
| Produce | RenderJobs |
| Listen | Listen |

Old routes redirect for one release:
- `/review?book=X` → `/book/X/speakers`
- `/render?book=X` → `/book/X/produce`
- `/lexicon?book=X` → `/book/X/proof#names`

A route with no book opens the most recently opened book, falling back to the shelf, never the first book alphabetically.

UI words map to CLI verbs in `lib/glossary.ts`. Every tooltip keeps the CLI term, and README gains the same table.

| UI word | CLI / internal term |
|---|---|
| Import | ingest |
| Find speakers | attribute |
| Cast voices | assign |
| Record | render |
| Build audiobook | master |
| Chapter MP3s (optional) | assemble |
| line | segment |
| paragraph | block |
| Check this | low-confidence |
| Get price | mint quote |
| Audio check | whisper validation |

### Flow A: first audiobook from an EPUB

1. **Empty shelf.** A "Get ready" checklist built from `/api/system?probe=true` and `/api/engines`:
   - backend ✓
   - Ollama reachable. "Model pulled" is added once 2.2 lands.
   - ffmpeg found
   - Kokoro weights, with a `[Download now]` button. It queues the existing warm-up job, which waits behind any running GPU job.

   Next to the checklist: a drop zone and **Try the sample book** (2.6).
2. **Import.** Choose *Full cast* or *Single narrator*. Review the chapters (word counts, items left out automatically, split level) and the cover. A duplicate upload returns 200 instead of 201 (`routes/books.py:155`) and the page says "Already on your shelf".
3. **Start: find the speakers.**
   - A pre-flight check refuses with one reason if Ollama or the model is missing (2.2). This applies only when the provider is local.
   - While running, the bottom bar shows `Finding speakers · ch 12/40 · 30% · ~18 min` and says the page can be closed.
   - On success, the job drafts a smart, local, free cast, but only if the book has no cast yet (2.7). A notice says "Found 23 characters and drafted a cast → Review voices".
4. **Voices.** Look over the board, play lines, swap any voice, Save.
5. **Proof** (optional on the first pass). Work through "Check this" and the names to check.
6. **Produce.** For a first book the default scope is **Try chapter 1 first**. Listen to the pilot, then choose **Looks good — make the whole book**.
   - Recording and building the `.m4b` run as one server-side chain (2.8).
   - If the cast includes a paid voice, the chain stops at the price-approval sheet. It never auto-confirms.
7. **Listen / Download** `<Title>.m4b`.

Click count to a full draft audiobook, if the user accepts the drafted cast: drop file → Full cast + Start → Make audiobook (pilot) → Looks good, make the whole book → confirm. That is about 5 clicks plus two waits.

```
IMPORT — The Expired Heart                                       step 1 of 6
 1 Import● ─ 2 Speakers ─ 3 Voices ─ 4 Proof ─ 5 Produce ─ 6 Listen
+------------------------------------------------------------------------+
| How should it sound?                                                   |
|  (o) Full cast — every character gets a voice (recommended)            |
|  ( ) Single narrator — one voice, ready fastest   [Heart ▾] [▶]        |
+------------------------------------------------------------------------+
| What's in the book                          [cover] [Replace] [Remove] |
|  [x] Ch 1  The Letter ........ 4,120 words  "It was raining..."        |
|  [x] Ch 2  Arrival ........... 3,880 words                             |
|  Left out: Copyright, Contents, Dedication            [include]        |
|  Chapters look wrong?  Split at: [heading level 2 ▾]                   |
+------------------------------------------------------------------------+
| ✗ Ollama isn't running — run: ollama serve   [copy] [Check again]      |
|                       [ Start: find the speakers ▸ ] (disabled)         |
+------------------------------------------------------------------------+
```

### Flow B: customising a character's voice

Each card on the Voices step shows:
- the name
- gender and age chips (already in the overview: `services/characters.py:18-21`)
- the line count
- the assigned voice as a human label
- badges: Clone / Emotion / Paid / On this PC

Steps:

1. **▶ Hear Mara say: "You came back."** This auditions the assigned voice with `text` set to her sample line, truncated to 500 characters.
2. **Alternatives.** Two or three ranked suggestions, each with a one-line "why", each playing the same line. This needs backend work (2.9): today's suggest response returns one voice id per character and no recipe, so voices that don't exist yet can't be previewed.
3. **Make a voice for Mara.** Opens the existing Add or Clone dialog over the page, prefilled with name, gender and accent. It returns with the new voice selected in the unsaved cast.
4. **Impact before saving.** For example: "312 lines in 18 chapters will re-record (~11 min, free)" (2.9 impact endpoint).
5. **Two separate actions.** **Suggest voices for uncast characters** never touches manual picks. **Reset all to suggestions** confirms first by listing exactly what will be lost.
6. **Tune** (needs owner approval, 3.7): named settings (Steady / Natural / Expressive, Speed) instead of raw numbers. **Make a variant** plays three seeded takes, and the one you keep pins its seed.
7. **Series.** A book in a series opens on **Returning cast (8, pre-filled) / Newcomers (3)**.

```
VOICES — cast the book                                          step 3 of 6
Accent [American ▾]   Narrator: Heart · warm US female [▶] [change]
+--------------------------------+ +--------------------------------+
| MARA  ♀ 60s · 312 lines        | | TOBIAS  ♂ 30s · 140 lines      |
| Bella · bright US F            | | ! needs a voice                |
| [Local][Free][no emotion]      | |  1 George · gravelly UK M [▶]  |
| ▶ "You came back."             | |    why: male, 30s, "gruff"     |
| Alt: Nicole [▶]  Sky [▶]       | |  2 Adam · calm US M      [▶]   |
| [Change] [Make a voice for her]| | [Pick] [Make a voice for him]  |
+--------------------------------+ +--------------------------------+
| ⚑ "Dr. Smith" may be "Smith" — [Merge (31 lines)] [Keep apart]    |
| ⚠ "the stranger": gender unknown → will get a female voice family |
 Mara → Sky re-records 312 lines in 18 chapters (~11 min, free)
 [Reset all to suggestions]              [ Save cast ▸ Proof ]
```

### Flow C: fixing a wrong speaker or a mispronunciation while listening

**Final version (Phase 3).** Proof and Listen share one page with an Edit/Listen toggle.

- **Wrong speaker.** In Listen, click the speaker chip → **Change speaker…**, which records the reassign. If a render is running, the change is saved as a server-side **Flag** (3.1) instead of failing with a 409.
- **Mispronounced word.** Click the word → **Fix pronunciation…**
  - Type a respelling.
  - Play **before and after in this sentence**.
  - **Play all 14 times** the word appears.
  - Choose the scope.
- **Getting fixes into audio.** A footer appears: "Your edits changed 4 lines in 2 chapters → Update audio." It opens Produce with "Only out-of-date lines" selected (staleness, 2.11).

**Interim version (Phase 1b).**
- The Listen chip menu links to `/review?book=&ch=&block=` (deep links from 1.9) or `/lexicon?book=&term=` (1.13).
- "Hear it" works by auditioning the sentence with the respelling substituted. This only approximates the render path's lexicon normalisation, and the UI says so.

```
PROOF · Ch [12 The Storm ▾] (3 to check · 21 out of date)  [Edit|Listen] [Issues 41]
+-----------------------------------------------+---------------------------+
| ✓ The wind tore at the shutters.              | Line · ch 12              |
| ✓ "You came back," Mara said.   [Mara]        | Speaker [Mara ▾]          |
| ! "I had to."   [Tobias?]  check this         |  [Looks right ⏎]          |
| ~ *She can't know,* he thought.  [thought]    |  1 Mara 2 Tobias N narr.  |
|   Kael'thas raised a hand.                    | Say "Kael'thas" as        |
|                                               |  [kale-thahs]             |
| ✓ recorded  ! check  ~ out of date  ✗ failed  |  [▶ before][▶ after]      |
|                                               |  [▶ all 14] [Save]        |
+-----------------------------------------------+---------------------------+
 Your edits changed 37 lines in 6 chapters                [ Update audio ▸ ]
 J/K next issue · 1-9 speaker · N narration · P play · F flag
```

### Flow D: producing the final m4b

The Produce step is one decision card:

1. **Version.** Full cast, or Single narrator with a voice and speed picker. Single narrator sends `single:{engine,voice,speed}` instead of today's `single: {}` (`RenderJobs.tsx:156`).
2. **Record.**
   - Options: Everything / Only out-of-date lines (2.11) / Try chapter 1 first / a chapter range.
   - After a recast that requires a whole-book render, the range option is disabled and the reason is shown (2.10 precheck). Until then, Phase 1 shows a warning.
3. **Finish.** "Build the .m4b when recording ends" is on by default and runs on the server (2.8).
   - Master needs only `rendered` (`routes/render.py:383-399`), so chapter MP3s move to Advanced.
   - The stage rail treats assemble as optional.

The single primary button states cost and time: **Make audiobook — free — about 5 h 40 min on this PC**. The GPU-time figure comes from the per-engine RTF added in 2.3.

A paid render, or a whole-book render above the threshold, opens **one** sheet. It shows hours of audio, the price, a 15:00 countdown, and the whole-book confirmation. Changing any setting throws the price away.

Below the card:
- **Versions.** Cards with a single Active badge. These replace the "listening to" control.
- **Warnings.** Missing chapters, out-of-date `.m4b`, cover changed. Each has a Rebuild button (2.10).
- **Audio checks.** Reported as outcomes. Each failure has ▶ / Open in Proof / Fix pronunciation / Re-make chapter N. Re-make is a forced chapter re-render; its confirm shows the line count and that paid lines bill again.

```
PRODUCE — make the audiobook                                    step 5 of 6
| Version  (o) Full cast   ( ) Single narrator [Heart ▾] speed [normal ▾] |
| Record   ( ) Everything  ( ) Only out-of-date (37)  (o) Try chapter 1    |
| Finish   [x] Build the .m4b when recording ends     ▸ Advanced           |
|  [ Make audiobook — free — about 5 h 40 min ]   safe to close this tab   |
|  Recording · ch 12/40 [######----] 30% · 52 min · ~2 h left   [Stop]     |
 VERSIONS  [Full cast · ACTIVE · 11h02m · 23 voices]  [⬇ The Expired Heart.m4b] [⬇ ZIP]
           ! Built before your last 37 changes                  [Rebuild]
           [Single narrator · 2 days ago]                       [Make active]
 AUDIO CHECKS  412 OK · 9 auto-retried · 2 need you
   ch7 "Kael'thas raised…" heard "Kale has raised"  [▶] [Open] [Fix name] [Re-make ch 7]
```

---

## 4. Phased plan

**Effort scale:** S = under 1 day · M = 2–4 days · L = 1–2 weeks · XL = more than 2 weeks.

**Gate:** the owner must approve first under CLAUDE.md. That applies to anything touching the voice model, the attribution/edit schema, the normalized schema, cache keys, or the money-gate contract.

### Phase 1a: stop the bleeding (about 3–4 weeks, frontend only)

| # | Change | Fixes | Files | Effort | Gate |
|---|---|---|---|---|---|
| 1.1 | **Stop casting data loss in the client.**<br>• Re-suggest, re-draft and Series apply always send the current `narrator_voice_id`, `thought_voice_id` and every manual pick as `overrides`. A manual pick is any `voice_id != \`${char_id}_auto\``.<br>• Overrides are filtered to character ids still in the registry, because the server rejects unknown ids (`assignment.py:172-175`).<br>• Show a diff before applying: "3 change · your 5 picks kept".<br>• Afterwards show "Cast applied · Undo"; Undo PUTs back the snapshot.<br>• Undo is offered only when no `recast` happened. A recast overwrites global `{char}_auto` recipes in the library, which a map snapshot can't restore.<br>• Fix the re-draft tooltip that promises "keeps existing voices". | Dx #4 | `screens/Review.tsx:270-315,372-381`, `Series.tsx:113-119`, `review/SuggestCastPanel.tsx`, new `lib/casting.ts` `keepManualOverrides()` (+test) | M | no |
| 1.2 | **One "Suggest a cast" button.**<br>• It sends `strategy:'smart'`. The hash auto-cast button is removed from the UI; the strategy stays in the CLI.<br>• Remove the draft/final chips. Stage defaults server-side.<br>• Render `SuggestCastResponse.assignment` as a table: character, gender/age, proposed voice, "new" / "keeps your pick" / "will overwrite" (from `would_recast_voice_ids`), and a per-row change dropdown that feeds `overrides`.<br>• Narrator and accent options are sent through: `useSuggestCast` currently sends only `{strategy}` (`hooks.ts:519-524`). | #4 | `Review.tsx:264-297`, `SuggestCastPanel.tsx:69-99`, `api/hooks.ts:507-524` | M | no |
| 1.3 | **Health pill and setup checklist.**<br>• Add a `useSystemProbe()` hook (key `["system","probe"]`, `?probe=true`, refetch on focus, about every 30 s).<br>• Add `useEngines()`.<br>• A System drawer with Download-now, which queues the warm-up job.<br>• `client.ts` turns an empty-body proxy 500/502 or a network error into "Seiyuu backend isn't running: `uv run uvicorn seiyuu.api.main:app`".<br>• The footer shows live status.<br>• Library's empty state shows the same checklist. | Dx #3 | `api/hooks.ts:288-294`, `api/client.ts:20-27`, `app/NavRail.tsx:46-50`, new `app/SystemDrawer.tsx`, `Library.tsx:205-219` | M | no |
| 1.4 | **Job outcome notices.**<br>• The watcher fetches `GET /api/jobs?limit=10` under `["jobs","recent"]` and raises a toast when a job finishes or fails.<br>• Toasts name the book title, or the engine for `engine:<id>` warm-up jobs.<br>• A failed-run banner stays on the Library card until it is dismissed (localStorage) or superseded by a newer job of the same kind.<br>• `lib/jobErrors.ts` maps known error strings (Ollama unreachable, ffmpeg missing, merge identity, render_active) to plain fixes. | #3 | `App.tsx:23-41` (adds a fetch; `["books"]` handling untouched), new `app/JobNotices.tsx`, `lib/jobErrors.ts`, `Library.tsx:28-38`, `render/JobRow.tsx` | M | no |
| 1.5 | **`<NextStep/>` button and upload result.**<br>• A pure `lib/nextStep.ts` computes the next step from stage flags, the active job and the last failure.<br>• Used on cards, StageRail (replacing the plain-text hint), empty states and Listen nudges (`Listen.tsx:270,361`).<br>• Stage labels become plain words; internal flags are unchanged.<br>• The upload result shows the title, `skipped_items`/`dropped_sections` in a details disclosure (already returned, `routes/books.py:157-163`), and "Already in your library" on a 200. This needs the ingest hook to expose the HTTP status.<br>• Cards drop raw job ids. | Dx #2, #8 | new `components/NextStep.tsx`, `StageRail.tsx:17-30`, `Library.tsx:28-38,121-203`, `Review.tsx:175,203`, `RenderJobs.tsx:220`, `Lexicon.tsx:141`, `Listen.tsx:270,361`, `api/types.ts:539-546`, `api/hooks.ts:81-91` | M | no |
| 1.6 | **Voice delete always confirms.** The dialog shows `/references`, including "used by nothing", and says the consent-attested reference clip is removed permanently. | #4 | `voices/VoiceCardView.tsx:285-292`, `hooks.ts:685-698`, `Voices.test.tsx:408` | S | no |

### Phase 1b: clarity on the current screens (about 5–8 weeks, frontend only)

| # | Change | Fixes | Files | Effort | Gate |
|---|---|---|---|---|---|
| 1.7 | **Nav in workflow order.**<br>• Order: Library · 1 Speakers & cast · 2 Proof & names · 3 Produce · 4 Listen · Tools (Voices, Series). Labels already match Phase 2 step names, so tests churn only once.<br>• Every book-scoped link carries `?book=` from a remembered current book (localStorage in try/catch; ignored if the book was deleted; an explicit `?book=` always wins).<br>• Theme labels become dark/light/auto. | #1 | `NavRail.tsx:6-50`, `NavRail.test.tsx:16-24`, new `app/currentBook.ts`, `RenderJobs.tsx:37`, `Review.tsx:38`, `Lexicon.tsx:92` | M | no |
| 1.8 | **Extract `AttributePanel`.**<br>• Mounted as "Full cast: find the speakers" on Library cards and in Review's empty state.<br>• On Render it becomes a normal step card instead of a refusal box.<br>• Disabled with one reason when `provider=local` and the probe says Ollama is unreachable.<br>• Once attributed, an Advanced option "re-run speaker finding for chapters…":<br>&nbsp;&nbsp;– Partial runs merge per chapter (`_merge_partial_attribution`, `services/attribution.py:235`), and edits replay (`apply_edits`).<br>&nbsp;&nbsp;– The attribution cache key means **the same provider/model/prompt returns the same result**, so the panel requires a different model or provider and says so.<br>&nbsp;&nbsp;– Alias adjudication does not run on chapter subsets (`services/attribution.py:212-216`). | #1, #2 | `RenderJobs.tsx:335-403` → `render/AttributePanel.tsx`, `Library.tsx:132-155`, `Review.tsx:175,203` | M | no |
| 1.9 | **Hear a voice where you cast it.**<br>• `useAudition` accepts optional `text`.<br>• ▶ on each roster row plays the assigned voice speaking that character's `sample_lines[0]` (≤500 chars).<br>• AuditionControl's refusal handling is reused: `engine_cold` → Download now, `gpu_busy` → "busy recording", `payment_confirmation_required` → "confirm ~$X & play". It never auto-confirms.<br>• "+ new voice for X" opens AddVoiceDialog over Review, prefilled, so unsaved casting survives. | #4 | `hooks.ts:645-649`, `review/VoicePicker.tsx`, `review/RosterRow.tsx:53-72`, `voices/AuditionControl.tsx` (extract hook), `voices/AddVoiceDialog.tsx` | M | no |
| 1.10 | **Review navigation and power tools** (existing endpoints only).<br>• Spoiler-safe off by default; moved to a View menu, preference remembered per viewer. When it is on, the reassign and merge lists also hide masked characters.<br>• `?ch=&block=` in the URL with scroll-to-line. A chapter dropdown.<br>• "Next to check" walks forward across chapters by lazily fetching the next chapter's `?low_confidence=` segments. Per-chapter counts in the dropdown come with 2.12.<br>• "Looks right ✓" records a same-speaker reassign.<br>• Clicking a character filters to their lines (`?speaker=`).<br>• ▶ on lines with `has_audio`.<br>• Thought lines styled italic with a "thought" tag (display only).<br>• react-aria chips (Enter/Space).<br>• Gender/age chips, plus an "unknown → female voice family" warning.<br>• Merge cards from `registry_notes`, and a "moves N lines" merge preview from `line_count`.<br>• "Go to line" on flagged blocks and edit warnings.<br>• Suppress "unattributed" once confidence is 1.0.<br>• A History drawer from `GET /edits`, with undo showing what it removed. | #5, #8 | `Review.tsx:42,95,97,161-235,508-515,537,559`, `review/SegmentPair.tsx:32-77`, `review/RosterRow.tsx`, `review/CharacterEditor.tsx`, `review/popovers.tsx`, `review/helpers.ts:18`, `hooks.ts:306-314,423-429` | L | no |
| 1.11 | **Listen links back.**<br>• The chip menu offers Fix this line / Fix pronunciation / Open in Review. Depends on the 1.10 deep links.<br>• Voice display names replace raw `voice_id` (`Listen.tsx:381`), with a note when the speaker changed since the line was recorded.<br>• Resume where you left off (localStorage).<br>• Download `.m4b`.<br>• Nudges become real links. | #5 | `Listen.tsx:76-78,180-190,270,361,366-386` | M | no |
| 1.12 | **Render screen clarity.**<br>• Modes renamed Full cast / One narrator.<br>• Narrator voice and speed for single mode.<br>• The quote is cleared on any scope, mode, emotion or force change (today only `force`, `RenderJobs.tsx:152`).<br>• The whole-book confirm moves *before* the request, using `limits.full_render_confirm_blocks`, and `confirm_full` goes out with `cost_token`. One dialog, free or paid; no 409 round trip.<br>• "Version used for Listen" moves next to Downloads.<br>• Finish builds the m4b only by default; MP3s optional; StageRail shows assemble as optional.<br>• After a recast with a chapter range, the existing casting-changed banner steers to the whole book.<br>• Progress is parsed from "chapter N/M", with the stripe as fallback.<br>• Emotion checkbox says Kokoro voices read flat.<br>• Audio-check rows get Open in Review / Fix pronunciation / Re-make chapter (force + scope, with a paid re-bill warning).<br>• The browser auto-finish chain stays until 2.8 replaces it. | #1, #7 | `RenderJobs.tsx:41,145,152-212,280-334,416-469,508-535`, `render/{EstimateCard,QuoteTicket,ActiveModeControl,ValidationFailure,StageRail,ConfirmFullDialog}.tsx`, new `lib/progress.ts`, `TransportBar.tsx:79-130` | L | no |
| 1.13 | **Voice Studio and Series clarity.**<br>• Capability badges from `/api/engines`. Emotion support is a frontend table (Kokoro: none, `voices/emotion.py:245-246`), because `/api/engines` has no emotion field.<br>• "Used by" from `/references`, including an "also used by \<other book\>" warning on `{char}_auto` voices (interim for 3.8).<br>• Human descriptor lines instead of `af_heart · seed 41172`.<br>• Book titles instead of raw book ids in Review tag chips.<br>• ElevenLabs stock-voice browser (`GET /engines/elevenlabs/voices`).<br>• Clone dialog: clip duration and preview, plus a warning under 20 s.<br>• "Replace reference clip" sends `voice_id`+`replace`, with consent attested again.<br>• Series tables show voice names with ▶ instead of raw `voice_id`s. | #4, #6 | `voices/VoiceCardView.tsx:274-331`, `Voices.tsx`, `voices/AddVoiceDialog.tsx:298-354`, `voices/CloneDialog.tsx:452-503`, `hooks.ts:628-698`, `Series.tsx:179-180,316-357`, `Review.tsx:341-366` | M | no |
| 1.14 | **Plain-language copy pass** through one `lib/glossary.ts`.<br>• Raw codes become a sentence plus a recourse. Example: `render_active` → "A render is running; edits pause until it finishes [Cancel render]".<br>• Remove the "tune them in Voice Studio" promise (`Review.tsx:394-398`) until 3.7 ships.<br>• Lexicon: `?book=&term=`, "lines/paragraphs" instead of "segments", "match capitals exactly", IPA under Advanced.<br>• Lexicon "hear it in a sentence" uses a narrator audition with the respelling substituted, labelled approximate.<br>• Lexicon gains a "re-render changed chapters" link. | #8 | all `screens/**`, `Lexicon.tsx:56-92,210-290` | M | no |

### Phase 2: workflow restructure and small backend additions (about 7–11 weeks)

| # | Change | Fixes | Files | Backend | Effort | Gate |
|---|---|---|---|---|---|---|
| 2.1 | **Server-side casting safety.**<br>• Add `keep_existing=true` (default) to `AssignmentDraftRequest`: the server folds in the saved narrator, thought voice and manual picks.<br>• "Reset all" sends `keep_existing=false`.<br>• Propose to the owner first: it changes the endpoint's default behaviour, and CLI/API tests pin the current behaviour. | #4 (CLI too) | `services/assignment.py:104-175`, `schemas.py:327-345` | behaviour only | S–M | no (propose) |
| 2.2 | **Attribution pre-flight.**<br>• For `provider=local`, POST `/attribute` checks Ollama is reachable and the model is installed (`/api/tags`); returns 503 `ollama_unreachable` / `model_missing` with the fix command.<br>• The probe gains `model_installed`.<br>• `GET /api/system/ollama-models` feeds a model dropdown. | #3 | `routes/books.py:327-397`, `routes/system.py:30-80` | S | S | no |
| 2.3 | **Jobs: error codes, retry, real progress, time estimate.**<br>• `JobOut.error_code` is classified at failure.<br>• `POST /jobs/{id}/retry` returns 402 for paid renders, so a fresh quote is needed.<br>• Structured progress (`progress_done/total/unit`, `started_at`) as additive SQLite columns. Today only `progress_text` exists (`repository/jobs.py:112,325`).<br>• Per-engine RTF is recorded from completed renders, and the estimate gains `gpu_seconds`. | #3, #7 | `jobs/runner.py`, `repository/jobs.py`, `attribute/pipeline.py:233`, `render/pipeline.py:532`, `schemas.py:35-50`, `routes/render.py` (estimate) | M | M | no |
| 2.4 | **Book shell and stepper** at `/book/:id/:step`.<br>• BookShell, Stepper and a pure `lib/steps.ts`.<br>• Existing screens are mounted as step bodies (see the IA table) and take a `bookId` prop.<br>• Unsaved drafts move into a `BookWorkspaceProvider` keyed by bookId.<br>• Legacy redirects. | #1, #2 | `App.tsx`, new `app/BookShell.tsx`, `app/Stepper.tsx`, `lib/steps.ts` | none | L | no |
| 2.5 | **Tools group.**<br>• An Activity screen (`GET /api/jobs` across books and engine warm-ups).<br>• A Settings & Health screen (`/api/settings`, probe, engines).<br>• The TransportBar splits into a job pill and a mini-player; `player.clear()` is wired on book switch. | #3, #7 | new `screens/Activity.tsx`, `screens/Settings.tsx`, `TransportBar.tsx`, `app/player.tsx:173-176` | none | M | no |
| 2.6 | **Import step.**<br>• **Keep the original upload** under the book's folder. Today the upload dir is deleted after ingest (`routes/books.py:164-165`), so no later structure review or re-import is possible without it.<br>• `GET /books/{id}/structure` (items, word counts, included/skipped).<br>• `POST /books/{id}/reingest` (refused once attributed, and while jobs are live).<br>• Intent choice and cover.<br>• `POST /books/sample` with a bundled public-domain EPUB (Q5). | #2, #8 | new `screens/book/ImportStep.tsx`, `routes/books.py:99-168`, `repository/books.py` | M | M | no |
| 2.7 | **`then_cast` auto-advance.** After attribution succeeds, the server drafts a smart, local cast with `use_llm=False`, *only if no assignments.json exists*. It never takes a paid path and never touches the GPU, since it only writes Kokoro recipes. | #1, #2 | `routes/books.py:327-397`, `jobs/runner.py`, `services/assignment.py` | M | M | no |
| 2.8 | **Server-side finish chain.**<br>• `RenderParams.finish` (`none` / `m4b` / `mp3+m4b`) and `parent_job_id`.<br>• Cancelling cascades to queued children; a restart doesn't enqueue twice.<br>• A failed master shows as its own notice.<br>• Delete the client chain at `RenderJobs.tsx:100-147`. | #7 | `schemas.py:438-459`, `jobs/runner.py`, `repository/jobs.py` | M | M | no |
| 2.9 | **Speakers step and Voices casting board.**<br>• Roster, merges, alias cards, re-read chapters.<br>• `POST /books/{id}/assignment/impact` (read-only).<br>• The suggest response gains ranked alternatives with reasons **and the Kokoro recipe for each proposed or alternative voice**, so voices not yet created can be previewed. The kokoro preview endpoint also needs a `text` option; today it always speaks the default sentence.<br>• Auditions are cached by (voice_id, text hash) under `data_dir/auditions/`, not inside `voices/{id}/`, so they stop overwriting `audition.wav` without touching voice records. | #4 | new `screens/book/{SpeakersStep,VoicesStep}.tsx`, `routes/review.py`, `routes/voices.py:607-724`, `routes/engines.py:157-250`, `voices/casting.py:118` | M | L | no (`set_traits` → 3.6) |
| 2.10 | **Honest outputs.**<br>• Downloads named `<Title>.m4b`.<br>• `built_at` and a source fingerprint on `FileDownload`, so a stale or partial m4b, or a changed cover, is detected.<br>• A merge-identity precheck at enqueue (409 instead of a late worker failure). | #7 | `routes/books.py:607-616`, `routes/render.py:275-378`, `schemas.py:169-180`, `render/pipeline.py:377-390` | M | M | no |
| 2.11 | **Staleness** (moved here from Phase 3, because Update audio, "Only out-of-date lines" and honest outputs depend on it).<br>• `GET /books/{id}/staleness` recomputes each line's segment key with the renderer's **own** planner and compares it to the active manifest's `audio_key`.<br>• Memoised on the assignment, lexicon, edit-log and voice-settings hashes; falls back to "unknown" rather than blocking.<br>• Read-only over the key format. | #5, #7 | new `services/staleness.py`, `render/cache.py`, `routes/render.py` | M | L | no |
| 2.12 | **Per-chapter review counts** (low-confidence, unattributed, flagged) in the characters overview, for the chapter dropdown and Shelf badges. | #5 | `services/characters.py:31-98` | S | S | no |

### Phase 3: bigger features (about 4–8 weeks ungated, plus gated items)

| # | Change | Fixes | Backend | Effort | Gate |
|---|---|---|---|---|---|
| 3.1 | **Issues inbox.**<br>• `GET /books/{id}/issues` aggregates: lines to check, unattributed quotes, possible duplicate characters, flagged paragraphs, edit warnings, failed audio checks, names to check, listener flags. Per-chapter counts.<br>• `POST/DELETE /books/{id}/flags`, stored on the server beside the edit log and **not** blocked by render_active, since flags never affect audio. | #5 | new `services/issues.py`, `routes/review.py` | L | no |
| 3.2 | **Proof workspace** with an Edit/Listen toggle on one page.<br>• Keyboard: J/K, 1-9, N, Enter, P, F. Scoped to the script view, disabled in inputs, listed in a "?" sheet.<br>• Bulk reassign (loops the existing edit POST, or a batch endpoint).<br>• An inspector panel and chapter badges. | #5 | optional batch edits route | L | no |
| 3.3 | **Pronunciation in place.**<br>• A candidate scan (names, rare capitalised words, numerals) grouped People/Places/Numbers.<br>• `GET /books/{id}/occurrences?term=` for play-every-occurrence.<br>• A lexicon audition that plays the line through the real normalisation path, with and without the rule. GPU-gated; paid voices need confirm. | #5 | `routes/lexicon.py` | L | no |
| 3.4 | **Revert any edit:** `DELETE /books/{id}/edits/{index}` replays the overlay without that edit, which also covers unmerge. render_active-guarded. | #5 | `routes/review.py:97-154` | M | propose first |
| 3.5 | **Versions cards and outputs:** "How it was made" recipe, chapter ZIP, master options (bitrate, target length, cover) under Advanced, audio checks reported as outcomes (`synth_attempts` → "auto-retried"). | #7 | ZIP route | M | no |
| 3.6 | New edit ops: `set_traits` (gender/age, feeds the caster), `set_type` (mark or unmark THOUGHT), `create_character`, `skip_block`, `speech_override` (what is spoken changes; display text and the reconstruction invariant stay intact) | #5, #4 | `services/edits.py`, `schemas.py:318-321`, CLI parity, adversarial tests | L | **yes** |
| 3.7 | Writable per-voice settings (named presets; `settings_hash` already makes affected lines stale) and seeded "Make a variant" auditions (`AuditionRequest.seed`, audition only) | #6 | `schemas.py:599-608`, `routes/voices.py:421-458` | M | **yes** |
| 3.8 | Book-scoped auto voice ids plus a migration, and the `taken`-set fix. The `keepManualOverrides` helper (1.1) must change in the same PR. Until then, 1.13 shows "also used by \<other book\>". | #4 | `services/assignment.py:48-68,133-163` | M | **yes** |
| 3.9 | Per-line re-voice with takes (Play/Retry/Keep/Revert). It bends the pinned-seed rule and adds a per-segment override to the render plan. | #7 | render planner, edits | XL | **yes** |
| 3.10 | Transcode cloned clips to canonical WAV before the consent hash; series/global lexicon layering; voice design (VoiceDesign model → reference.wav) | #6 | engines, voices, normalize | XL | **yes** |

Items that need approval keep their UI slots **hidden** until approved, never shown disabled.

### Effort summary (using the scale above)

| Phase | Estimate | Main drivers |
|---|---|---|
| 1a | ~3–4 weeks | 5 M + 1 S |
| 1b | ~5–8 weeks | 6 M + 2 L |
| 2 | ~7–11 weeks | |
| 3 (ungated) | ~4–8 weeks | |
| Gated items, if approved | ~5–8 weeks | |

Ungated total: about 19–31 developer-weeks.

The shortest path to fix the owner's complaint (the confusion, the dead ends, the data loss) is Phase 1a plus 1.7, 1.8, 1.9 and 1.12. That is about 6–9 weeks.

### Test impact

- **Frontend:**
  - These tests change: `NavRail.test.tsx:16-24` (order pin), `Library.test.tsx:21-24` (card buttons), `RenderJobs.test.tsx:376` (two-step confirm becomes one), `Voices.test.tsx:408` (delete always confirms), plus Review/Lexicon copy assertions. Do the relabelling in one dedicated commit.
  - Migrate the ~185 tests; don't delete them.
  - Each new pure helper gets a co-located test: `lib/nextStep`, `lib/casting`, `lib/progress`, `lib/steps`, `lib/glossary`, `lib/jobErrors`.
- **Backend:**
  - **Add every new route to `tests/test_api_m6b6.py`:** retry, ollama-models, structure, reingest, sample, impact, staleness, issues, flags, occurrences, chapter ZIP, edits/{index}.
  - The jobs-table column migration needs a WAL-safe test.
  - `then_cast` needs a test proving it never overwrites an existing cast and never takes a paid path.
  - The finish chain needs tests for cancel cascading and for not enqueuing twice after a restart.
  - Staleness needs a parity test against the renderer's own key decisions.
- **Before calling any phase done:** `uv run ruff check`, `ruff format --check`, `uv run pytest -n auto`, oxlint, `tsc -b`, vitest.

---

## 5. Judge scores and why this direction won

Per-criterion scores are averages across the three judges, as reported by the judging step. This review did not re-verify them.

| Criterion (avg) | Incremental Clarity | **Guided Pipeline** | Script-as-Editor Studio |
|---|---|---|---|
| First run | 7.3 | **9.0** | 8.2 |
| Confusion reduction | 6.5 | **9.0** | 7.5 |
| Voice customisation | 7.0 | 8.2 | **8.5** |
| Editing loop | 6.5 | 8.0 | **9.2** |
| Constraint safety | **9.0** | 7.3 | 6.0 |
| Effort / risk | **8.3** | 4.2 | 3.7 |
| Judge A total (/10) | 7.3 | **7.8 ✔** | 7.3 |
| Judge B total (/60) | 44 | **45 ✔** | 42 |
| Judge C total (/10) | **7.7 ✔** | 7.2 | 6.8 |

**Why this direction won.** Two of three judges picked the Guided Pipeline. It scored highest on the two criteria that match the owner's complaint: confusion and first run. Its weakness is effort and risk. Judge C's winner, Incremental Clarity, answers that weakness, because its fix list is effectively Guided's Phase 0. So the hybrid ships Incremental's work first, on the current screens:
- Phase 1a: data loss, silent failures, missing next step, delete safety.
- Phase 1b: navigation, auditions, Review/Listen/Render clarity.

The stepper restructure lands only after those are proven.

From the Studio direction, the plan takes:
- the chapter-1 pilot
- the sample book
- Proof and Listen on one surface
- error codes with retry
- audition caching
- server-stored flags
- revert-any-edit

It deliberately does **not**:
- change the quote contract (the one-dialog paid confirm uses fields `RenderParams` already accepts)
- narrow the render_active guard
- flip the API's default casting strategy (the UI just stops using `hash`)

Fixes to Guided's own flaws:
- Re-import becomes server-side, by keeping the original upload plus `/structure` and `/reingest`, instead of relying on a File held in browser memory.
- The Speakers step ships ungated, with `set_traits` split out.
- Book-scoped auto voice ids are separated from the Voice Library work.
- Staleness moves earlier, so "Update audio" isn't months away.

---

## 6. Open questions for the owner

1. **Which gated items do you approve, and in what order?** Per-voice tuning (writes `VoiceMeta.settings`), book-scoped auto voice ids plus migration, new edit ops (traits / thought / skip / speech override / new character), and per-line re-voice with takes (bends the pinned-seed rule).
2. **Is server-side automation OK by default?** Specifically: "Full cast" auto-drafts a local cast after finding speakers (`then_cast`), and "Build the .m4b when done" is on by default.
3. **May `keep_existing=true` become the draft endpoint's default (2.1)?** This also changes what the CLI and API do when no overrides are passed.
4. **Can spoiler-safe default to off in Review and become mainly a Listen "reader mode"?**
5. **Should this be a SPEC.md milestone (e.g. M9 "Guided UX")?** And does it come before or after the October 2026 tech-refresh and engine work?
6. **May Seiyuu keep each book's original upload on disk (needed for structure review and re-import, 2.6)?** And may it bundle a short public-domain sample EPUB in package data? If yes, which one: a few chapters, 3–5 speakers.

---

## Critic notes

- **Checked against the repo.** These references hold:
  - `NavRail.tsx:6-14` order and the hard-coded footer at 46-50
  - `NavRail.test.tsx:16-24`
  - `hooks.ts:56` (live jobs = queued/running), `288-294` (no probe), `507-524`, `645-649` (audition sends only `confirm_paid`)
  - `Review.tsx:38,42,95`
  - `RenderJobs.tsx:37,41,65,152,156`
  - `schemas.py:327-345` (`strategy` default `"hash"`, accent/narrator/overrides exist), `599-608` (VoiceUpdate name/tags only), `625-627` (AuditionRequest.text)
  - `assignment.py:104-175`
  - `repository/books.py:117` (mastered = m4b exists)
  - `Listen.tsx:377-382`
  - `tests/test_api_m6b6.py` exists
- **The one-dialog paid confirm is sound.** `RenderParams` has `cost_token` and `confirm_full`, the route checks `confirm_full` before the money dry-run, and the threshold is in `/api/system` limits. I added this evidence and applied the one-dialog flow to free whole-book renders too.
- **Re-run speaker finding (1.8).** Chapter subsets do merge (`_merge_partial_attribution`) and edits replay. But the attribution cache key returns identical output for the same provider/model/prompt, and adjudication skips subsets. The panel now requires a different model or provider and says so.
- **Re-import was unsupported.** The server deletes the upload after ingest (`routes/books.py:164-165`), so `/structure` and `/reingest` can't work without keeping the source. Added that to 2.6 and as Q6.
- **Suggest-preview playback and alternatives were unsupported.** `SuggestCastResponse` has no alternatives and no recipes, and voices not yet created can't be auditioned. The Kokoro preview always speaks the default sentence. Moved these to 2.9 as backend work and raised 2.9 from M to L.
- **Undo after apply cast (1.1).** It can't restore recipes that a `recast` overwrote, because auto voices are global. Undo is now offered only when no recast happened. Overrides are also filtered to known character ids, since the server rejects unknown ones.
- **GPU rule wording corrected.**
  - Playing rendered audio isn't GPU work.
  - Kokoro auditions can borrow a running Kokoro render's engine.
  - Warm-up is a queued job with an `engine:<id>` subject, which notices and Activity must render.
- **Phase 1 "model pulled" claim fixed.** The probe reports only reachability, so model-installed checks wait for 2.2. Flow A says so. The pre-flight applies only to `provider=local`.
- **Kokoro emotion badges.** `/api/engines` has no emotion field, so 1.13 uses a frontend capability table.
- **Effort was inconsistent with the plan's own scale.** Phase 1 had 11 M + 2 L, called "about 2 weeks". It is now split into 1a (~3–4 wk) and 1b (~5–8 wk). Phase 2 is re-estimated at ~7–11 wk, and I added an honest total and a "shortest path" figure.
- **Staleness moved from 3.1 to Phase 2 (2.11).** Several Phase 2 and Flow D features depend on it, and the high-severity "edits never mark audio stale" item would otherwise wait months. Phase 3 is renumbered: 3.1 Issues … 3.10.
- **Added missing coverage for unaddressed friction items:**
  - Phase 1 only: skipped/dropped items, "already in your library", and raw job ids (1.5); thought-line styling, a spoiler-leak fix in the reassign/merge lists, flagged-block "Go to line", merge line-count preview, and clearing the unattributed label (1.10); Series voice names and audition, and descriptor lines instead of seed jargon (1.13); removing the false "tune them in Voice Studio" promise (1.14); Kokoro-reads-flat notice and validation "Re-make chapter" with a paid re-bill warning (1.12).
  - Backend/Phase 2: GPU-time estimate via per-engine RTF (2.3), needed for Flow D's "about 5 h 40 min" button; per-chapter review counts (2.12).
- **Phase 1 "frontend only" kept honest.** Cross-chapter "Next to check" now walks lazily over existing filtered segment fetches. Chapter-dropdown counts wait for 2.12.
- **Click count fixed.** The plan said "about 4 clicks" while Flow A's own steps, with the pilot, take about 5. Corrected.
- **Nav labels aligned.** 1.7 now uses Phase 2 step names, so tests and copy churn only once. StageRail and Finish now treat assemble as optional.
- **IA gap filled.** Added the mapping of existing screens to stepper steps before the Proof workspace exists. "Recent books" is marked as a localStorage convenience.
- **Audition caching location.** Moved from voice folders to `data_dir/auditions/`, so voice records aren't touched.
- **Gating.** 2.1 (default behaviour change) is now "propose first" and is open question Q3. The flipped default is no longer called harmless.
- **Unverifiable material flagged.** Judge scores come from the judging step and weren't re-verified. Of the competitor URLs, only VoiceStudio's is corroborated in the repo (`design/voicestudio-assessment-2026-10-22.md`); check the rest before sharing externally. Fixed the friction map's wrong warm-up reference (`engines.py:199-224` is the preview route; warm-up is `engines.py:77-102`).