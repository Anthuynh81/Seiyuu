import { useRef } from "react";

import type { JobOut } from "../api/types";
import { jobChapters, parseChapterProgress, type ChapterProgress } from "../lib/progress";

/** A job's chapter progress, held across the in-between lines. The render loop interleaves
    "chapter N/M" with other notices (a flagged line, a voice loading); without this the bar
    would blink back to the stripe every time one of those lands. Holds only for the same
    job id — a different job starts over. Null until the first chapter line. */
export function useJobProgress(job: Pick<JobOut, "job_id" | "progress_text" | "params"> | null | undefined): ChapterProgress | null {
  const last = useRef<{ id: string; p: ChapterProgress } | null>(null);
  if (!job) return null;
  const parsed = parseChapterProgress(job.progress_text, jobChapters(job.params));
  if (parsed) {
    last.current = { id: job.job_id, p: parsed };
    return parsed;
  }
  return last.current?.id === job.job_id ? last.current.p : null;
}
