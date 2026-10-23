/** Reading a job's progress line. The backend's render and speaker-finding loops announce
    each chapter as "chapter N/M: Title" (render/pipeline.py, attribute/pipeline.py), where N
    is the chapter's place in the WHOLE book. Pure: no I/O. When the line is anything else
    (loading weights, "verifying cost approval…", a flagged-line notice) this returns null
    and the UI falls back to the moving stripe. */

export interface ChapterProgress {
  /** the chapter being worked on (1-based, its index in the book) */
  chapter: number;
  /** chapters in the book */
  total: number;
  /** the chapter's title when the line carries one */
  title: string | null;
  /** chapters already finished within this job's scope */
  done: number;
  /** chapters in this job's scope (the whole book, or the picked range) */
  of: number;
  /** done / of, 0..1 */
  fraction: number;
  /** true when the job covers only some chapters */
  partial: boolean;
}

const LINE = /^\s*chapter\s+(\d+)\s*\/\s*(\d+)(?:\s*:\s*(.*\S))?/i;

/** The chapter list a job was started with (`params.chapters`), or [] for the whole book. */
export function jobChapters(params: Record<string, unknown> | null | undefined): number[] {
  const raw = params?.chapters;
  if (!Array.isArray(raw)) return [];
  return raw.filter((c): c is number => typeof c === "number" && Number.isInteger(c) && c > 0);
}

/** Parse "chapter N/M[: title]". `scope` is the job's chapter list ([] = whole book); with a
    range, progress counts chapters within the range, not N out of the whole book (a render
    of chapters 30–32 on chapter 31 is a third done, not 75%). */
export function parseChapterProgress(
  text: string | null | undefined,
  scope: readonly number[] = [],
): ChapterProgress | null {
  const m = text?.match(LINE);
  if (!m) return null;
  const chapter = Number(m[1]);
  const total = Number(m[2]);
  if (!(total > 0) || chapter < 1 || chapter > total) return null;
  const title = m[3]?.trim() || null;
  const inScope = [...new Set(scope)].filter((c) => c <= total).sort((a, b) => a - b);
  const pos = inScope.indexOf(chapter);
  if (inScope.length > 0 && pos >= 0) {
    return { chapter, total, title, done: pos, of: inScope.length, fraction: pos / inScope.length, partial: true };
  }
  return { chapter, total, title, done: chapter - 1, of: total, fraction: (chapter - 1) / total, partial: false };
}

/** "chapter 12 of 40 · 27%" — or "chapter 31 (2 of 3) · 33%" for a chapter range. */
export function progressLabel(p: ChapterProgress): string {
  const pct = `${Math.round(p.fraction * 100)}%`;
  const where = p.partial ? `chapter ${p.chapter} (${p.done + 1} of ${p.of})` : `chapter ${p.chapter} of ${p.total}`;
  return `${where} · ${pct}`;
}
