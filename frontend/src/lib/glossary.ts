import { ApiError, BACKEND_DOWN, BACKEND_START_COMMAND } from "../api/client";
import type { JobKind, JobOut, RenderMode } from "../api/types";

/** The ONE place user-facing words for Seiyuu's internal concepts live.

    - TERMS: every internal/CLI concept → the word the UI uses (attribute → "find the
      speakers", segment → "line", multivoice → "Full cast", master → "build the audiobook").
      Tooltips keep the CLI term (`cliHint`) so a power user can still map a screen to a command.
    - SCREENS: what each screen is called, so copy that points somewhere ("on the Produce
      screen") can't drift from the nav.
    - Job words: what a job is doing / did / failed at (lib/jobCopy builds sentences from them).
    - Failures: raw codes and backend phrases → one plain sentence, a fix, and (where the UI
      can do it) a recourse action such as [Cancel recording]. lib/jobErrors re-exports this.

    Pure: no React, no I/O. A raw id or code is never the main text; it may sit in a tooltip
    or a "Details" disclosure. */

/* ================================================================ terms */

export interface Term {
  /** the word the UI uses, lower case (capitalise at the start of a sentence with `cap`) */
  ui: string;
  /** the CLI verb / internal name, for tooltips */
  cli: string;
}

export const TERMS = {
  ingest: { ui: "import", cli: "ingest" },
  attribute: { ui: "find the speakers", cli: "attribute" },
  attribution: { ui: "speaker finding", cli: "attribution" },
  assign: { ui: "cast the voices", cli: "assign" },
  render: { ui: "record", cli: "render" },
  master: { ui: "build the audiobook", cli: "master" },
  assemble: { ui: "chapter MP3s (optional)", cli: "assemble" },
  segment: { ui: "line", cli: "segment" },
  block: { ui: "paragraph", cli: "block" },
  lowConfidence: { ui: "check this", cli: "low-confidence" },
  unattributed: { ui: "speaker unknown", cli: "unattributed" },
  quote: { ui: "get the price", cli: "mint quote" },
  validation: { ui: "audio check", cli: "whisper validation" },
  multivoice: { ui: "Full cast", cli: "multivoice" },
  single: { ui: "One narrator", cli: "single" },
  lexicon: { ui: "pronunciation", cli: "lexicon" },
  respelling: { ui: "say it like", cli: "respelling" },
  ipa: { ui: "phonetic spelling (IPA)", cli: "ipa" },
  caseSensitive: { ui: "match capitals exactly", cli: "case_sensitive" },
  llm: { ui: "the AI", cli: "LLM" },
  registry: { ui: "character list", cli: "registry" },
} as const satisfies Record<string, Term>;

export type TermKey = keyof typeof TERMS;

/** The UI word for an internal concept. */
export const term = (key: TermKey): string => TERMS[key].ui;

/** A tooltip suffix that keeps the CLI term: "(CLI: attribute)". */
export const cliHint = (key: TermKey): string => `(CLI: ${TERMS[key].cli})`;

/** "line"/"lines" by count: `plural("segment", 3)` → "lines". */
export function plural(key: TermKey, n: number): string {
  const word = TERMS[key].ui;
  return n === 1 ? word : `${word}s`;
}

/** Capitalise the first letter of a sentence fragment. */
export const cap = (s: string): string => (s ? s[0].toUpperCase() + s.slice(1) : s);

/** The recording modes. lib/render's MODE_LABEL is this table. */
export const MODE_NAMES: Record<RenderMode, string> = {
  multivoice: TERMS.multivoice.ui,
  single: TERMS.single.ui,
};

/* ================================================================ screens */

/** What each screen is called in the nav: copy that sends the user somewhere uses these. */
export const SCREENS = {
  library: "Library",
  review: "Speakers & cast",
  lexicon: "Proof & names",
  render: "Produce",
  listen: "Listen",
  voices: "Voices",
  series: "Series",
} as const;

export type ScreenKey = keyof typeof SCREENS;

/* ================================================================ jobs */

/** What a job is DOING, as a step name ("Recording", "Building the .m4b audiobook"). */
export const JOB_DOING: Record<JobKind, string> = {
  ingest: "Adding the book",
  attribute: "Finding the speakers",
  render: "Recording",
  assemble: "Building the chapter MP3s",
  master: "Building the .m4b audiobook",
  warmup: "Getting the voice engine ready",
};

/** The finish headline for a job that worked, naming its subject. */
export const JOB_DONE: Record<JobKind, (name: string) => string> = {
  ingest: (t) => `Added ${t} to your library`,
  attribute: (t) => `Found the speakers in ${t}`,
  render: (t) => `Recorded ${t}`,
  assemble: (t) => `Built the chapter MP3s for ${t}`,
  master: (t) => `Built the .m4b audiobook for ${t}`,
  warmup: (e) => `${e} is downloaded and ready`,
};

export const JOB_FAILED: Record<JobKind, (name: string) => string> = {
  ingest: (t) => `Couldn't add ${t}`,
  attribute: (t) => `Couldn't find the speakers in ${t}`,
  render: (t) => `Recording ${t} stopped with a problem`,
  assemble: (t) => `Couldn't build the chapter MP3s for ${t}`,
  master: (t) => `Couldn't build the .m4b audiobook for ${t}`,
  warmup: (e) => `Couldn't get ${e} ready`,
};

export const JOB_STOPPED: Record<JobKind, (name: string) => string> = {
  ingest: (t) => `Stopped adding ${t}`,
  attribute: (t) => `Stopped finding the speakers in ${t}`,
  render: (t) => `Stopped recording ${t}`,
  assemble: (t) => `Stopped building the chapter MP3s for ${t}`,
  master: (t) => `Stopped building the .m4b audiobook for ${t}`,
  warmup: (e) => `Stopped getting ${e} ready`,
};

/** A card's failure headline (the card already names the book). */
export const JOB_FAILED_STEP: Record<JobKind, string> = {
  ingest: "Adding this book failed",
  attribute: "Finding the speakers failed",
  render: "The last recording failed",
  assemble: "Building the chapter MP3s failed",
  master: "Building the .m4b audiobook failed",
  warmup: "Getting the voice engine ready failed",
};

/** The button that stops a running job of this kind ("Cancel recording"). */
export const JOB_CANCEL: Record<JobKind, string> = {
  ingest: "Stop adding the book",
  attribute: "Stop finding the speakers",
  render: "Cancel recording",
  assemble: "Stop building the MP3s",
  master: "Stop building the audiobook",
  warmup: "Stop the download",
};

/* ================================================================ failures */

export type JobErrorId =
  | "ollama_unreachable"
  | "model_missing"
  | "ffmpeg_missing"
  | "render_active"
  | "conflicting_job"
  | "duplicate_job"
  | "busy_saving"
  | "job_already_ended"
  | "internal"
  | "gpu_busy"
  | "gpu_other_process"
  | "engine_cold"
  | "audition_in_flight"
  | "cloud_busy"
  | "quote_expired"
  | "quote_stale"
  | "quote_needed"
  | "ceiling_exceeded"
  | "paid_confirm"
  | "merge_identity"
  | "interrupted"
  | "not_imported"
  | "no_speakers_yet"
  | "not_recorded"
  | "not_cast"
  | "not_built"
  | "nothing_to_undo"
  | "prerequisite"
  | "edit_conflict"
  | "voice_in_use"
  | "voice_exists"
  | "reclone_blocked"
  | "series_exists"
  | "consent_invalid"
  | "upstream"
  | "damaged_file"
  | "backend_down"
  | "starting"
  | "key_missing"
  | "nothing_to_quote"
  | "nothing_to_warm"
  | "alignment_failed"
  | "partial_delete"
  | "whole_book_confirm"
  | "too_large"
  | "wrong_file_type"
  | "not_found"
  | "invalid"
  | "unknown";

/** Something the UI can do about a failure, beyond reading the fix. */
export type Recourse = { kind: "cancel_job"; label: string; jobId: string };

export interface JobErrorExplanation {
  id: JobErrorId;
  /** One plain sentence: what went wrong. */
  summary: string;
  /** What to do about it; null when we don't know. */
  fix: string | null;
  /** An exact command to run, shown copyable. */
  command?: string;
  /** A button the UI can offer (e.g. [Cancel recording] for the job in the way). */
  recourse?: Recourse;
  /** The raw message, for the Details disclosure (null when there was none). */
  details: string | null;
  known: boolean;
}

export interface FailureInput {
  code?: string | null;
  message?: string | null;
  /** an ApiError's detail (a blocking JobOut for render_active / conflicting_job) */
  detail?: unknown;
}

type Explained = Omit<JobErrorExplanation, "id" | "details" | "known">;

interface Rule {
  id: Exclude<JobErrorId, "unknown">;
  codes: string[];
  /** Stable backend phrases (case-insensitive). */
  pattern?: RegExp;
  /** a pattern that applies only together with one of these codes (generic codes like
      not_found / stage_prerequisite are split by their message) */
  onlyWithCodes?: boolean;
  explain: (raw: string, detail: unknown) => Explained;
}

/** The model name from "model qwen2.5:7b isn't pulled" or "`ollama pull qwen2.5:7b`". */
function modelName(raw: string): string | null {
  const m = /model '?([^\s'`]+?)'? isn't pulled/i.exec(raw) ?? /ollama pull ([^\s'`)?]+)/i.exec(raw);
  return m ? m[1] : null;
}

/** The blocking job an API refusal carries in `detail` (render_active, conflicting_job). */
function blockingJob(detail: unknown): Pick<JobOut, "job_id" | "kind" | "state"> | null {
  if (typeof detail !== "object" || detail === null) return null;
  const d = detail as Record<string, unknown>;
  if (typeof d.job_id !== "string" || typeof d.kind !== "string") return null;
  return { job_id: d.job_id, kind: d.kind as JobKind, state: (d.state as JobOut["state"]) ?? "running" };
}

function cancelRecourse(detail: unknown): Recourse | undefined {
  const job = blockingJob(detail);
  if (!job) return undefined;
  return { kind: "cancel_job", jobId: job.job_id, label: JOB_CANCEL[job.kind] ?? "Stop it" };
}

/** The book ids a reclone_blocked refusal names in `detail.stale_books`. */
export function staleBooks(detail: unknown): string[] {
  if (typeof detail !== "object" || detail === null) return [];
  const list = (detail as Record<string, unknown>).stale_books;
  return Array.isArray(list) ? list.filter((b): b is string => typeof b === "string") : [];
}

/** Where cancelling lives, for a failure shown without its own button. */
const IN_THE_BAR = " in the bar at the bottom";

const FRESH_PRICE = "Get a fresh price and approve it to start again.";

// Order matters: codes are matched first (exact); then patterns, in this order —
// model_missing before ollama_unreachable ("cannot reach Ollama" also mentions `ollama pull`,
// but a missing model never says it can't reach Ollama).
const RULES: Rule[] = [
  {
    id: "model_missing",
    codes: ["model_missing"],
    pattern: /OllamaModelMissingError|isn't pulled in Ollama/i,
    explain: (raw) => {
      const model = modelName(raw);
      return {
        summary: "The model that finds the speakers isn't downloaded yet.",
        fix: "Download it in Ollama with the command below, then try again. The first download is a few GB.",
        command: `ollama pull ${model ?? "<model>"}`,
      };
    },
  },
  {
    id: "ollama_unreachable",
    codes: ["ollama_unreachable"],
    pattern: /OllamaUnreachableError|cannot reach Ollama|Ollama isn't reachable/i,
    explain: (raw) => {
      // The provider raises "cannot reach Ollama" for ANY request failure, HTTP errors
      // included: Ollama may be running and still refuse (model not pulled, out of memory).
      // So that text gets both checks; the pre-flight text and code mean it is truly down.
      if (/cannot reach Ollama/i.test(raw)) {
        const model = modelName(raw);
        return {
          summary: "Seiyuu couldn't get an answer from Ollama, the program that finds who is speaking.",
          fix: "Make sure Ollama is running (open the Ollama app, or run `ollama serve`) and that the model is downloaded with the command below, then try again.",
          command: `ollama pull ${model ?? "<model>"}`,
        };
      }
      return {
        summary: "Seiyuu couldn't reach Ollama, the program that finds who is speaking.",
        fix: "Start Ollama (open the Ollama app, or run the command below), then try again.",
        command: "ollama serve",
      };
    },
  },
  {
    id: "ffmpeg_missing",
    codes: ["ffmpeg_missing"],
    pattern: /ffmpeg not found/i,
    explain: () => ({
      summary: "ffmpeg isn't installed. Seiyuu needs it to build the audiobook file.",
      fix: "Install ffmpeg, make sure it's on your PATH, restart the backend, then try again.",
    }),
  },
  {
    id: "render_active",
    codes: ["render_active"],
    pattern: /a render job for .* is (queued|running)/i,
    explain: (raw, detail) => {
      // review.py's guard is per book ("a render job for 'b1' is running"); the Voices
      // guard (voices.py _guard_live_render) is global and may name an assemble/master job
      // for ANY book ("a master job is running; re-cloning this voice would …")
      const job = blockingJob(detail);
      const forThisBook = /a render job for /i.test(raw);
      if (forThisBook || !job || job.kind === "render") {
        return {
          summary: forThisBook
            ? "A recording is running for this book; changes are paused until it finishes."
            : "A recording is running; this has to wait until it finishes.",
          fix: `Wait for the recording to finish, or cancel it${IN_THE_BAR}, then try again.`,
          recourse: cancelRecourse(detail),
        };
      }
      return {
        summary: `${JOB_DOING[job.kind] ?? "Another job"} is under way; this has to wait until it finishes.`,
        fix: `Wait for it to finish, or stop it${IN_THE_BAR}, then try again.`,
        recourse: cancelRecourse(detail),
      };
    },
  },
  {
    id: "conflicting_job",
    codes: ["conflicting_job"],
    explain: (_raw, detail) => {
      const job = blockingJob(detail);
      const doing = job ? (JOB_DOING[job.kind] ?? "Another job") : "Another job";
      return {
        summary: `${doing} is still running for this book, so this has to wait.`,
        fix: `Wait for it to finish, or stop it${IN_THE_BAR}, then try again.`,
        recourse: cancelRecourse(detail),
      };
    },
  },
  {
    id: "duplicate_job",
    // enqueue.py: the standard refusal for starting a job that is already queued/running
    codes: ["duplicate_job"],
    explain: (_raw, detail) => {
      const job = blockingJob(detail);
      const doing = job ? (JOB_DOING[job.kind] ?? "That job") : "That job";
      const when = job?.state === "queued" ? "already waiting to start" : "already under way";
      return {
        summary: `${doing} is ${when} for this book.`,
        fix: `No need to start it again: follow it${IN_THE_BAR}, or stop it to start over.`,
        recourse: cancelRecourse(detail),
      };
    },
  },
  {
    id: "busy_saving",
    // errors.py: SQLite contention; the server sends Retry-After: 1
    codes: ["lock_timeout"],
    explain: () => ({
      summary: "Seiyuu was busy saving something else.",
      fix: "Try again in a moment.",
    }),
  },
  {
    id: "job_already_ended",
    codes: ["illegal_transition"],
    explain: () => ({
      summary: "That job has already finished or stopped, so it can't be changed now.",
      fix: "Reload to see where it stands.",
    }),
  },
  {
    id: "gpu_busy",
    // the API refusal codes: an audition/preview turned away while this server's own job
    // holds the card
    codes: ["gpu_busy", "gpu_busy_retry"],
    explain: () => ({
      summary: "Recording in progress, try after it finishes.",
      fix: "Only one voice model fits on the graphics card at a time. Try again once the current job is done.",
    }),
  },
  {
    id: "gpu_other_process",
    codes: [],
    // A JOB failing with GpuBusyError: the server hands the card between its own jobs
    // in-process, so this only fires when ANOTHER Seiyuu process holds gpu.lock
    // (gpu/manager.py) — never "a recording here is running".
    pattern: /GpuBusyError|holds the GPU/i,
    explain: () => ({
      summary: "Another Seiyuu program (a command window or a second server) is using the graphics card.",
      fix: "Close it or wait for it to finish, then start this again.",
    }),
  },
  {
    id: "engine_cold",
    codes: ["engine_cold"],
    explain: () => ({
      summary: "That voice engine isn't loaded yet.",
      fix: "Its first use loads it. Start a recording with it, or get it ready from the System panel, then try again.",
    }),
  },
  {
    id: "audition_in_flight",
    codes: ["audition_in_flight"],
    explain: () => ({
      summary: "Another preview is still being made.",
      fix: "Wait for it to finish, then try again.",
    }),
  },
  {
    id: "cloud_busy",
    codes: ["cloud_busy"],
    explain: () => ({
      summary: "A recording is using the cloud voices right now.",
      fix: "Try again after it finishes.",
    }),
  },
  {
    id: "quote_expired",
    codes: ["quote_expired"],
    pattern: /cost token expired/i,
    explain: () => ({
      summary: "The price quote expired before the recording started.",
      fix: "Get a fresh estimate and approve the new price to start again.",
    }),
  },
  {
    id: "quote_stale",
    codes: ["quote_mismatch", "quote_used", "cost_drift", "quote_signature_invalid", "quote_refused"],
    pattern: /token already used|drifted upward|quote .*(changed|mismatch)/i,
    explain: () => ({
      summary: "The approved price no longer matches what would be recorded (something changed, or it was already used).",
      fix: FRESH_PRICE,
    }),
  },
  {
    id: "quote_needed",
    codes: ["token_required"],
    explain: () => ({
      summary: "This recording uses paid voices, so it needs an approved price first.",
      fix: "Get the price and approve it; nothing is billed until you do.",
    }),
  },
  {
    id: "ceiling_exceeded",
    codes: ["ceiling_exceeded"],
    pattern: /render_max_usd ceiling/i,
    explain: () => ({
      summary: "This would cost more than the spending limit set for Seiyuu.",
      fix: "Record fewer chapters, or raise the limit (RENDER_MAX_USD) in the backend's settings and restart it.",
    }),
  },
  {
    id: "paid_confirm",
    codes: ["payment_confirmation_required"],
    explain: () => ({
      summary: "This uses a paid online service, so it needs your approval first.",
      fix: "Approve the cost to go ahead. Nothing has been spent yet.",
    }),
  },
  {
    id: "merge_identity",
    codes: [],
    pattern: /refusing (single-voice|multivoice) chapter-subset render/i,
    explain: () => ({
      summary:
        "The other chapters were recorded with different voices, so recording only these chapters was refused.",
      fix: "Record the whole book to use the new voices everywhere, or switch the voices back to match.",
    }),
  },
  {
    id: "interrupted",
    codes: [],
    // written by restart reconciliation for rows still running at startup (repository/jobs.py)
    pattern: /interrupted: server stopped mid-job/i,
    explain: () => ({
      summary: "Seiyuu was closed while this was running.",
      fix: "Start it again; finished parts are reused from the cache.",
    }),
  },
  // -- the generic "an earlier step isn't done" codes, split by their stable messages
  {
    id: "not_imported",
    codes: ["stage_prerequisite", "not_found"],
    onlyWithCodes: true,
    pattern: /not ingested|run ingest first/i,
    explain: () => ({
      summary: "This book hasn't been imported yet.",
      fix: `Add its file in the ${SCREENS.library}, then try again.`,
    }),
  },
  {
    id: "no_speakers_yet",
    codes: ["stage_prerequisite", "not_found"],
    onlyWithCodes: true,
    pattern: /no attribution|run attribute first|not attributed/i,
    explain: () => ({
      summary: "The speakers haven't been found in this book yet.",
      fix: `Find them on the ${SCREENS.render} screen first.`,
    }),
  },
  {
    id: "not_recorded",
    codes: ["stage_prerequisite", "not_found"],
    onlyWithCodes: true,
    pattern: /no render|not rendered|run render first/i,
    explain: () => ({
      summary: "Nothing has been recorded for this book yet.",
      fix: `Record it on the ${SCREENS.render} screen first.`,
    }),
  },
  {
    id: "not_cast",
    codes: ["stage_prerequisite", "not_found"],
    onlyWithCodes: true,
    pattern: /no assignment|seiyuu assign|run assign first|not assigned|no cast/i,
    explain: () => ({
      summary: "The voices haven't been cast for this book yet.",
      fix: `Cast them on ${SCREENS.review} first.`,
    }),
  },
  {
    id: "not_built",
    codes: ["stage_prerequisite", "not_found"],
    onlyWithCodes: true,
    pattern: /run (master|assemble) first/i,
    explain: (raw) =>
      /run master first/i.test(raw)
        ? {
            summary: "The audiobook hasn't been built yet.",
            fix: `Build it on the ${SCREENS.render} screen first.`,
          }
        : {
            summary: "The chapter MP3s haven't been built yet.",
            fix: `Build them on the ${SCREENS.render} screen first.`,
          },
  },
  {
    id: "nothing_to_undo",
    codes: ["not_found"],
    onlyWithCodes: true,
    pattern: /no manual edits to undo/i,
    explain: () => ({
      summary: "There are no changes to undo in this book.",
      fix: "Reload to see the latest list of changes.",
    }),
  },
  {
    id: "prerequisite",
    codes: ["stage_prerequisite"],
    explain: () => ({
      summary: "An earlier step isn't finished for this book yet.",
      fix: "Finish the earlier steps first, then try again.",
    }),
  },
  {
    id: "edit_conflict",
    codes: ["edit_conflict"],
    explain: () => ({
      summary: "That line changed since this page loaded, so the change wasn't saved.",
      fix: "Reload the chapter and make the change again.",
    }),
  },
  {
    id: "voice_in_use",
    codes: ["voice_referenced"],
    explain: () => ({
      summary: "This voice is still part of a book's cast.",
      fix: "Give those characters another voice first, then try again.",
    }),
  },
  {
    id: "voice_exists",
    codes: ["voice_exists"],
    explain: () => ({
      summary: "A voice with that name already exists.",
      fix: "Choose a different name.",
    }),
  },
  {
    id: "reclone_blocked",
    // voices.py: the dry-run consent step; detail.stale_books names the recorded books
    // that would need re-recording (ids: a screen with the library at hand shows titles)
    codes: ["reclone_blocked"],
    explain: (_raw, detail) => {
      const stale = staleBooks(detail);
      const books =
        stale.length === 0
          ? "No recorded book uses it yet."
          : `${stale.length} recorded ${stale.length === 1 ? "book" : "books"} would need re-recording.`;
      return {
        summary: "A voice with this name already exists. Cloning again replaces its recording.",
        fix: `${books} Its saved lines are cleared, and any paid lines are billed again when re-recorded.`,
      };
    },
  },
  {
    id: "series_exists",
    codes: ["conflict"],
    explain: () => ({
      summary: "A series with that name already exists.",
      fix: "Choose a different name, or add the book to the existing series.",
    }),
  },
  {
    id: "consent_invalid",
    codes: ["consent_invalid"],
    explain: () => ({
      summary: "This cloned voice's recording doesn't match the consent given for it.",
      fix: "Re-create the voice with fresh consent.",
    }),
  },
  {
    id: "upstream",
    codes: ["upstream", "upstream_error"],
    explain: () => ({
      summary: "The voice or AI service failed this time.",
      fix: "Try again in a moment.",
    }),
  },
  {
    id: "damaged_file",
    codes: ["corrupt_artifact"],
    explain: () => ({
      summary: "One of this book's saved files is damaged and couldn't be read.",
      fix: "Run the step that made it again. If it keeps happening, check the backend's console.",
    }),
  },
  {
    id: "backend_down",
    codes: [BACKEND_DOWN],
    explain: () => ({
      summary: "The Seiyuu backend isn't running.",
      fix: "Start it from the project folder with the command below, then try again.",
      command: BACKEND_START_COMMAND,
    }),
  },
  {
    id: "key_missing",
    codes: ["not_ready"],
    onlyWithCodes: true,
    // a paid service's key isn't set (books.py, lexicon.py, review.py, render.py, voices.py,
    // engines.py). Naming the setting is fine; a key itself is never shown.
    pattern: /_API_KEY not set/i,
    explain: (raw) =>
      /ANTHROPIC_API_KEY/i.test(raw)
        ? {
            summary: "Using Claude online needs an Anthropic key, and none is set.",
            fix: "Add ANTHROPIC_API_KEY to the backend's .env settings and restart it, or use the local model instead.",
          }
        : {
            summary: "Cloud voices need an ElevenLabs key, and none is set.",
            fix: "Add ELEVENLABS_API_KEY to the backend's .env settings and restart it, or use a local voice instead.",
          },
  },
  {
    id: "starting",
    // enqueue.py: the job runner isn't accepting work (still starting, or shutting down)
    codes: ["not_ready"],
    explain: () => ({
      summary: "The backend isn't ready to take new work yet.",
      fix: "Try again in a moment. If it keeps happening, restart the backend.",
    }),
  },
  {
    id: "nothing_to_quote",
    codes: ["nothing_to_quote"],
    explain: () => ({
      summary: "Nothing in this selection costs money: it's all free or already recorded.",
      fix: "Start the recording without a price.",
    }),
  },
  {
    id: "nothing_to_warm",
    codes: ["nothing_to_warm"],
    explain: () => ({
      summary: "That's an online voice service, so there's nothing to download.",
      fix: "It's ready whenever its key is set.",
    }),
  },
  {
    id: "alignment_failed",
    codes: ["alignment_failed"],
    explain: () => ({
      summary: "The word timings for this line couldn't be worked out.",
      fix: "The audio still plays; only the word-by-word highlight is missing for this line.",
    }),
  },
  {
    id: "partial_delete",
    codes: ["partial_delete"],
    explain: () => ({
      summary: "The book was only partly deleted: some of its files are open in another program.",
      fix: "Close any program using them (a media player, for example), then delete it again.",
    }),
  },
  {
    id: "whole_book_confirm",
    codes: ["full_render_confirmation_required"],
    explain: () => ({
      summary: "This records the whole book, a long job on the graphics card.",
      fix: "Confirm the whole-book recording to start it.",
    }),
  },
  {
    id: "too_large",
    codes: ["payload_too_large"],
    explain: () => ({ summary: "That file is too big.", fix: "Use a smaller file." }),
  },
  {
    id: "wrong_file_type",
    codes: ["unsupported_media_type"],
    explain: () => ({ summary: "That kind of file isn't supported here.", fix: "Use a supported file type." }),
  },
  {
    id: "not_found",
    codes: ["not_found"],
    explain: () => ({
      summary: "That couldn't be found. It may have been deleted.",
      fix: "Reload the page and try again.",
    }),
  },
  {
    id: "internal",
    codes: ["internal"],
    explain: () => ({
      summary: "Seiyuu hit an unexpected problem.",
      fix: "Try again. If it keeps happening, check the backend's console.",
    }),
  },
  {
    id: "invalid",
    codes: ["invalid"],
    // the server's own reason is the useful part ("chapter 99 is out of range")
    explain: (raw) => ({ summary: "Seiyuu couldn't accept that.", fix: raw || null }),
  },
];

function normalize(input: string | null | undefined | FailureInput): FailureInput {
  if (input === null || input === undefined) return {};
  if (typeof input === "string") return { message: input };
  return input;
}

function findRule(code: string | null | undefined, raw: string): Rule | undefined {
  if (code) {
    // a pattern-split rule for this code first (not_found "has no attribution" → speakers)
    const split = RULES.find((r) => r.onlyWithCodes && r.codes.includes(code) && r.pattern?.test(raw));
    if (split) return split;
    const exact = RULES.find((r) => !r.onlyWithCodes && r.codes.includes(code));
    if (exact) return exact;
  }
  return raw ? RULES.find((r) => !r.onlyWithCodes && r.pattern?.test(raw)) : undefined;
}

/** A failure → plain sentence + fix (+ command, + recourse). The raw message is always kept as
    `details`: the mapping explains, it never hides the evidence. */
export function explainFailure(input: string | null | undefined | FailureInput): JobErrorExplanation {
  const { code, message, detail } = normalize(input);
  const raw = (message ?? "").trim();
  const details = raw === "" ? null : raw;
  const rule = findRule(code, raw);
  if (rule) {
    const explained = rule.explain(raw, detail);
    // a rule that already shows the raw words as its fix needs no Details copy of them
    return { id: rule.id, ...explained, details: explained.fix === raw ? null : details, known: true };
  }
  return {
    id: "unknown",
    summary: "Something went wrong.",
    fix: details ? null : "Try again. If it keeps failing, check the backend's console.",
    details,
    known: false,
  };
}

/** Any thrown value (an ApiError, an Error, a string) → the failure input the table reads. */
export function failureOf(err: unknown): FailureInput {
  if (err instanceof ApiError) return { code: err.code, message: err.message, detail: err.detail };
  if (err instanceof Error) return { message: err.message };
  if (typeof err === "string") return { message: err };
  return {};
}

/** One line for places with room for a sentence only: the summary and fix together. An
    unknown failure keeps the server's own words after `fallback` ("Couldn't save: …"). */
export function failureLine(err: unknown, fallback = "Something went wrong"): string {
  const input = failureOf(err);
  const out = explainFailure(input);
  if (out.known) return out.fix ? `${out.summary} ${out.fix}` : out.summary;
  return out.details ? `${fallback}: ${out.details}` : `${fallback}.`;
}
