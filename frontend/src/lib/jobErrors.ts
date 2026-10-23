import type { JobOut } from "../api/types";
import { explainFailure, type FailureInput, type JobErrorExplanation } from "./glossary";

/** Failures in plain words. The table itself (codes and backend phrases → sentence, fix,
    recourse) lives in lib/glossary.ts, the one vocabulary; this module keeps its historical
    exports and the "which failure still matters" helper. */

export type { FailureInput, JobErrorExplanation, JobErrorId, Recourse } from "./glossary";
export { failureLine, failureOf } from "./glossary";

/** A job's `error` string, or an ApiError's code + message (+ detail) → plain words. */
export function explainJobError(input: string | null | undefined | FailureInput): JobErrorExplanation {
  return explainFailure(input);
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
