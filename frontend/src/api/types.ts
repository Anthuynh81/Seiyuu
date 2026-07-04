/* TypeScript mirrors of the M6b API schemas (src/seiyuu/api/schemas.py). Only what the
   built screens consume — grown per section, never speculatively. */

export interface ErrorEnvelope {
  error: { code: string; message: string; detail: unknown };
}

export type JobState = "queued" | "running" | "succeeded" | "failed" | "canceled";
export type JobKind = "ingest" | "attribute" | "render" | "assemble" | "master" | "warmup";

export interface JobOut {
  job_id: string;
  book_id: string;
  kind: JobKind;
  state: JobState;
  progress_text: string;
  error: string | null;
  cancel_requested: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  is_terminal: boolean;
  params: Record<string, unknown> | null;
}

export interface JobsOut {
  jobs: JobOut[];
}

export interface ActiveJobSummary {
  job_id: string;
  kind: JobKind;
  state: "queued" | "running";
}

export interface BookCard {
  book_id: string;
  title: string | null;
  authors: string[];
  ingested: boolean;
  attributed: boolean;
  assigned: boolean;
  rendered: boolean;
  assembled: boolean;
  mastered: boolean;
  active_job: ActiveJobSummary | null;
}

export interface BooksOut {
  books: BookCard[];
}

export interface IngestResponse {
  book: Omit<BookCard, "active_job">;
  chapters: number;
  blocks: number;
  skipped_items: string[];
  dropped_sections: string[];
}

/** The six pipeline stages in signal-path order, with the card flag for each. */
export const STAGES = [
  ["ingested", "ingest"],
  ["attributed", "attribute"],
  ["assigned", "assign"],
  ["rendered", "render"],
  ["assembled", "assemble"],
  ["mastered", "master"],
] as const;

/** Which stage a running job kind lights up on the signal path. */
export const KIND_STAGE: Partial<Record<JobKind, (typeof STAGES)[number][0]>> = {
  attribute: "attributed",
  render: "rendered",
  assemble: "assembled",
  master: "mastered",
};
