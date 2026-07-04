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

export interface ChapterSummary {
  index: number;
  title: string;
  blocks: number;
  speakable_blocks: number;
}

export interface FileDownload {
  url: string;
  bytes: number;
}

export interface ChapterDownload extends FileDownload {
  index: number;
}

export interface BookDetail {
  status: Omit<BookCard, "active_job">;
  chapters: ChapterSummary[] | null;
  runtime_estimate_seconds: number | null;
  active_job: ActiveJobSummary | null;
  recent_jobs: JobOut[];
  downloads: { m4b: FileDownload | null; chapter_mp3s: ChapterDownload[] };
  cover: { content_type: string; bytes: number } | null;
}

export type RenderMode = "multivoice" | "single";

export interface CostEstimateOut {
  total_usd: number;
  paid_segments: number;
  cached_segments: number;
  free_segments: number;
  fingerprint: string;
  assignment_hash: string | null;
  mode: RenderMode;
  chapters: number[];
  edit_warnings: string[];
}

export interface QuoteResponse {
  token: string;
  book_id: string;
  chapters: number[];
  total_usd: number;
  paid_segments: number;
  fingerprint: string;
  assignment_hash: string | null;
  issued_at: number;
  expires_at: number;
  ttl_seconds: number;
  max_usd_ceiling: number;
}

export interface RenderRequest {
  mode: RenderMode;
  chapters?: number[];
  cost_token?: string;
  confirm_full?: boolean;
  single?: { engine?: string; voice?: string; speed?: number; seed?: number };
}

export interface ValidationRow {
  chapter_index: number;
  block_id: string;
  segment_index: number;
  voice_id: string | null;
  ok: boolean;
  score: number;
  expected: string;
  transcript: string;
  synth_attempts: number;
}

export interface ValidationReportOut {
  validated_segments: number;
  validation_failures: number;
  results: ValidationRow[];
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
