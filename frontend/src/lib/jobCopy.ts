import type { BookCard, JobKind, JobOut } from "../api/types";
import { engineName } from "./health";

/** Plain-words copy for jobs: what a job is DOING (cards, while live) and what it DID (the
    finish notice). Pure; the UI never shows a raw job id as the main text. */

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

const DOING: Record<JobKind, string> = {
  ingest: "Adding the book",
  attribute: "Finding the speakers",
  render: "Recording",
  assemble: "Building the audiobook",
  master: "Making the final audiobook file",
  warmup: "Getting the voice engine ready",
};

/** "Recording…" / "Recording — waiting its turn" for the live line on a card. */
export function liveJobLine(job: Pick<JobOut, "kind" | "state">): string {
  const doing = DOING[job.kind] ?? "Working";
  return job.state === "queued" ? `${doing} — waiting its turn` : `${doing}…`;
}

const DONE: Record<JobKind, (name: string) => string> = {
  ingest: (t) => `Added ${t} to your library`,
  attribute: (t) => `Found the speakers in ${t}`,
  render: (t) => `Recorded ${t}`,
  assemble: (t) => `Built the audiobook for ${t}`,
  master: (t) => `Made the final audiobook file for ${t}`,
  warmup: (e) => `${e} is downloaded and ready`,
};

const FAILED: Record<JobKind, (name: string) => string> = {
  ingest: (t) => `Couldn't add ${t}`,
  attribute: (t) => `Couldn't find the speakers in ${t}`,
  render: (t) => `Recording ${t} stopped with a problem`,
  assemble: (t) => `Couldn't build the audiobook for ${t}`,
  master: (t) => `Couldn't make the final audiobook file for ${t}`,
  warmup: (e) => `Couldn't get ${e} ready`,
};

const STOPPED: Record<JobKind, (name: string) => string> = {
  ingest: (t) => `Stopped adding ${t}`,
  attribute: (t) => `Stopped finding the speakers in ${t}`,
  render: (t) => `Stopped recording ${t}`,
  assemble: (t) => `Stopped building the audiobook for ${t}`,
  master: (t) => `Stopped making the final audiobook file for ${t}`,
  warmup: (e) => `Stopped getting ${e} ready`,
};

/** The one-line headline for a finished job, naming its subject. */
export function finishedHeadline(job: Pick<JobOut, "kind" | "state">, name: string): string {
  const table = job.state === "failed" ? FAILED : job.state === "canceled" ? STOPPED : DONE;
  return (table[job.kind] ?? ((n: string) => `Finished a job for ${n}`))(name);
}

/** Card-level failure headline: the card already names the book. */
export function failedStepLine(kind: JobKind): string {
  const doing: Record<JobKind, string> = {
    ingest: "Adding this book failed",
    attribute: "Finding the speakers failed",
    render: "The last recording failed",
    assemble: "Building the audiobook failed",
    master: "Making the final audiobook file failed",
    warmup: "Getting the voice engine ready failed",
  };
  return doing[kind] ?? "The last job failed";
}
