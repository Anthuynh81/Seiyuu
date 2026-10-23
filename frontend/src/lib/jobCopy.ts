import type { BookCard, JobKind, JobOut } from "../api/types";
import { JOB_DOING, JOB_DONE, JOB_FAILED, JOB_FAILED_STEP, JOB_STOPPED } from "./glossary";
import { engineName } from "./health";

/** Plain-words sentences for jobs: what a job is DOING (cards, while live) and what it DID (the
    finish notice). Pure; the UI never shows a raw job id as the main text. The words come from
    lib/glossary.ts (assemble builds the "chapter MP3s", master builds the ".m4b audiobook"). */

/** Warm-up jobs use the documented `engine:<id>` subject in the book column. */
export function engineSubject(bookId: string): string | null {
  return bookId.startsWith("engine:") ? bookId.slice("engine:".length) : null;
}

/** The book's title for a job's subject, or the engine's name for a warm-up job. A book that
    is gone (or has no title) reads as "your book" rather than its id. */
export function subjectName(job: Pick<JobOut, "book_id">, books: BookCard[] | undefined): string {
  const engine = engineSubject(job.book_id);
  if (engine !== null) return engineName(engine);
  return books?.find((b) => b.book_id === job.book_id)?.title ?? "your book";
}

/** The plain step name for a job kind ("Recording", "Building the .m4b audiobook"). */
export function stepName(kind: JobKind): string {
  return JOB_DOING[kind] ?? "Working";
}

/** "Recording…" / "Recording — waiting its turn" for the live line on a card. */
export function liveJobLine(job: Pick<JobOut, "kind" | "state">): string {
  const doing = stepName(job.kind);
  return job.state === "queued" ? `${doing} — waiting its turn` : `${doing}…`;
}

/** The one-line headline for a finished job, naming its subject. */
export function finishedHeadline(job: Pick<JobOut, "kind" | "state">, name: string): string {
  const table = job.state === "failed" ? JOB_FAILED : job.state === "canceled" ? JOB_STOPPED : JOB_DONE;
  return (table[job.kind] ?? ((n: string) => `Finished a job for ${n}`))(name);
}

/** Card-level failure headline: the card already names the book. */
export function failedStepLine(kind: JobKind): string {
  return JOB_FAILED_STEP[kind] ?? "The last job failed";
}
