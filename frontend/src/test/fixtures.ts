import type { BookCard, JobOut } from "../api/types";

/** Typed factories for the shapes most tests share. Only the widely-reused ones live here —
    screen-specific payloads belong next to the screen's test, built from api/types. */

export function makeBook(overrides: Partial<BookCard> = {}): BookCard {
  return {
    book_id: "demo",
    title: "Demo Book",
    authors: ["A. Author"],
    ingested: true,
    attributed: false,
    assigned: false,
    rendered: false,
    assembled: false,
    mastered: false,
    active_job: null,
    ...overrides,
  };
}

export function makeJob(overrides: Partial<JobOut> = {}): JobOut {
  return {
    job_id: "job-1",
    book_id: "demo",
    kind: "render",
    state: "running",
    progress_text: "chapter 1/3",
    error: null,
    cancel_requested: false,
    created_at: "2026-07-11T00:00:00Z",
    started_at: "2026-07-11T00:00:01Z",
    finished_at: null,
    is_terminal: false,
    params: null,
    ...overrides,
  };
}
