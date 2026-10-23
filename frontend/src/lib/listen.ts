import type { SegmentRow, VoiceAssignment } from "../api/types";

/** Listen's links back to the fixes, the voice credited in each line's margin, and the
    per-viewer "resume where you left off" record. Pure: no I/O, no storage access (the
    screen reads/writes storage through lib/storage and hands the raw string here). */

/* -------------------------------------------------- links back to the fixes */

/** Character Review, landed on this exact line (the 1.10 deep link: ?ch=&block=&seg=). */
export function reviewLineHref(bookId: string, chapter: number, row: Pick<SegmentRow, "block_id" | "segment_index">): string {
  const q = new URLSearchParams({
    book: bookId,
    ch: String(chapter),
    block: row.block_id,
    seg: String(row.segment_index),
  });
  return `/review?${q}`;
}

/** Character Review, opened on this chapter. */
export function reviewChapterHref(bookId: string, chapter: number): string {
  return `/review?${new URLSearchParams({ book: bookId, ch: String(chapter) })}`;
}

/** The Pronunciation screen for this book, with `term` waiting in a new row when given. */
export function lexiconHref(bookId: string, term: string | null): string {
  const q = new URLSearchParams({ book: bookId });
  if (term) q.set("term", term);
  return `/lexicon?${q}`;
}

/** A word worth adding to the pronunciation dictionary, from a text selection or a clicked
    word: outer punctuation and quotes trimmed (inner apostrophes and hyphens kept, so
    "Kael'thas" and "Jean-Luc" survive). Null when nothing word-like is left, or when the
    selection is a whole phrase rather than a name (more than 4 words or 60 characters). */
export function pronunciationTerm(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const text = raw.replace(/\s+/g, " ").trim().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
  if (!text || text.length > 60 || text.split(" ").length > 4) return null;
  return text;
}

/* -------------------------------------------------- voices in the margin */

/** The voice the CURRENT cast gives a line — the same rule the renderer applies
    (voices/assignment.py resolve_voice): narration and speakerless lines take the narrator,
    a thought takes the thought voice when one is set, a character takes their pick, else
    the narrator. */
export function castVoiceFor(
  row: Pick<SegmentRow, "type" | "speaker">,
  assignment: Pick<VoiceAssignment, "narrator_voice_id" | "assignments" | "thought_voice_id">,
): string {
  if (row.type === "narration" || row.speaker === null) return assignment.narrator_voice_id;
  if (row.type === "thought" && assignment.thought_voice_id) return assignment.thought_voice_id;
  return assignment.assignments[row.speaker] ?? assignment.narrator_voice_id;
}

/** The voice a full-cast line would get today, when that differs from the voice it was
    recorded with — its speaker was changed, or that speaker was recast, after recording.
    Null when they match, the line has no recording, nothing is cast, or the audio is a
    one-narrator recording (every line is that narrator by design). */
export function voiceChangedSince(
  row: Pick<SegmentRow, "type" | "speaker" | "voice_id">,
  assignment: Pick<VoiceAssignment, "narrator_voice_id" | "assignments" | "thought_voice_id"> | null | undefined,
  fullCast: boolean,
): string | null {
  if (!fullCast || !assignment || !row.voice_id) return null;
  const now = castVoiceFor(row, assignment);
  return now === row.voice_id ? null : now;
}

/* -------------------------------------------------- resume where you left off */

/** Where this viewer stopped in a book: the chapter, the rendered clip (`block:segment`) and
    the offset inside it, and the seconds into the chapter (for the offer's mm:ss, and the
    fallback when a re-render replaced that clip). */
export interface ResumePoint {
  chapter: number;
  clip: string;
  offset: number;
  seconds: number;
}

export const resumeKey = (bookId: string) => `seiyuu.listen.${bookId}`;

/** A stored resume point, or null for anything missing or garbled (never throws). */
export function parseResume(raw: string | null): ResumePoint | null {
  if (!raw) return null;
  try {
    const v: unknown = JSON.parse(raw);
    if (typeof v !== "object" || v === null) return null;
    const { chapter, clip, offset, seconds } = v as Record<string, unknown>;
    if (!Number.isInteger(chapter) || (chapter as number) < 1) return null;
    if (typeof clip !== "string") return null;
    if (typeof offset !== "number" || !Number.isFinite(offset) || offset < 0) return null;
    if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return null;
    return { chapter: chapter as number, clip, offset, seconds };
  } catch {
    return null;
  }
}

/** Seconds into a chapter given the clip durations before the current one. */
export function chapterSeconds(clipDurations: readonly number[], index: number, offset: number): number {
  let sum = 0;
  for (let i = 0; i < index && i < clipDurations.length; i++) sum += clipDurations[i];
  return sum + offset;
}

/** Where to seek for a resume point in a chapter's clips: its own clip when still there,
    else the same seconds-into-the-chapter (clamped to the last clip). */
export function resumeTarget(
  clips: readonly { key: string; duration: number }[],
  point: Pick<ResumePoint, "clip" | "offset" | "seconds">,
): { index: number; offset: number } | null {
  if (clips.length === 0) return null;
  const own = clips.findIndex((c) => c.key === point.clip);
  if (own >= 0) return { index: own, offset: point.offset };
  let left = point.seconds;
  for (let i = 0; i < clips.length; i++) {
    if (left < clips[i].duration || i === clips.length - 1) return { index: i, offset: Math.max(0, left) };
    left -= clips[i].duration;
  }
  return null;
}

/** m:ss, or h:mm:ss from an hour. */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

/** Human file size for a download button. */
export function fileSize(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}
