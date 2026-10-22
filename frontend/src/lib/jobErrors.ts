import type { JobOut } from "../api/types";

/** Pure mapping from a failure (a job's `error` string, or an ApiError's code + message) to a
    plain sentence and a fix a non-developer can act on. The raw message is always kept as
    `details` for a "Details" disclosure: the mapping explains, it never hides the evidence.

    Job errors arrive as `"<ExceptionType>: <message>"` (jobs/runner.py); API refusals carry a
    stable machine `code`. Both are matched: codes first (exact), then the stable message
    phrases the backend raises (assemble/pipeline.py, attribute/providers/local.py,
    render/pipeline.py, gpu/manager.py, api/money.py). */

export type JobErrorId =
  | "ollama_unreachable"
  | "model_missing"
  | "ffmpeg_missing"
  | "render_active"
  | "gpu_busy"
  | "gpu_other_process"
  | "engine_cold"
  | "audition_in_flight"
  | "quote_expired"
  | "merge_identity"
  | "interrupted"
  | "unknown";

export interface JobErrorExplanation {
  id: JobErrorId;
  /** One plain sentence: what went wrong. */
  summary: string;
  /** What to do about it; null when we don't know. */
  fix: string | null;
  /** An exact command to run, shown copyable. */
  command?: string;
  /** The raw message, for the Details disclosure (null when there was none). */
  details: string | null;
  known: boolean;
}

export interface FailureInput {
  code?: string | null;
  message?: string | null;
}

interface Rule {
  id: Exclude<JobErrorId, "unknown">;
  codes: string[];
  /** Stable backend phrases (case-insensitive). */
  pattern?: RegExp;
  explain: (raw: string) => Omit<JobErrorExplanation, "id" | "details" | "known">;
}

/** The model name from "model qwen2.5:7b isn't pulled" or "`ollama pull qwen2.5:7b`". */
function modelName(raw: string): string | null {
  const m = /model '?([^\s'`]+?)'? isn't pulled/i.exec(raw) ?? /ollama pull ([^\s'`)?]+)/i.exec(raw);
  return m ? m[1] : null;
}

// Order matters: model_missing before ollama_unreachable ("cannot reach Ollama" also
// mentions `ollama pull`, but a missing model never says it can't reach Ollama).
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
    explain: () => ({
      summary: "A recording is running for this book, so the change was refused.",
      fix: "Wait for the recording to finish, or cancel it in the bar at the bottom, then try again.",
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
    id: "quote_expired",
    codes: ["quote_expired"],
    pattern: /cost token expired/i,
    explain: () => ({
      summary: "The price quote expired before the recording started.",
      fix: "Get a fresh estimate and approve the new price to start again.",
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
];

function normalize(input: string | null | undefined | FailureInput): FailureInput {
  if (input === null || input === undefined) return {};
  if (typeof input === "string") return { message: input };
  return input;
}

export function explainJobError(input: string | null | undefined | FailureInput): JobErrorExplanation {
  const { code, message } = normalize(input);
  const raw = (message ?? "").trim();
  const details = raw === "" ? null : raw;
  const rule =
    (code ? RULES.find((r) => r.codes.includes(code)) : undefined) ??
    (raw ? RULES.find((r) => r.pattern?.test(raw)) : undefined);
  if (rule) return { id: rule.id, ...rule.explain(raw), details, known: true };
  return {
    id: "unknown",
    summary: "Something went wrong.",
    fix: details ? null : "Try again. If it keeps failing, check the backend's console.",
    details,
    known: false,
  };
}

// -- which failure still matters on a book card ------------------------------------------------

/** The book's most recent failed job, unless a newer job of the same kind exists (the user
    already retried: queued, running, or done, it supersedes the failure). Order by
    created_at so the API's sort order is not load-bearing. */
export function unresolvedFailure(jobs: JobOut[] | undefined, bookId: string): JobOut | null {
  if (!jobs) return null;
  const mine = jobs
    .filter((j) => j.book_id === bookId)
    .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  const failed = mine.find((j) => j.state === "failed");
  if (!failed) return null;
  const failedAt = Date.parse(failed.created_at);
  const superseded = mine.some(
    (j) => j.job_id !== failed.job_id && j.kind === failed.kind && Date.parse(j.created_at) > failedAt,
  );
  return superseded ? null : failed;
}
