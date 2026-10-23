import type { SegmentRow } from "../api/types";
import { chapterOfBlock } from "../api/types";

/** Character Review navigation: where the page is (from the URL) and "Next to check", which
    walks forward through lines that need a look, chapter by chapter. Pure — the screen
    supplies the rows; no I/O here. */

/** A line on the page: a paragraph (block) and, when known, which line within it. */
export interface LineRef {
  block: string;
  /** the segment index within the block; null = the paragraph's first line */
  seg: number | null;
}

/** What the URL asks the page to show: `?ch=` (else the linked paragraph's own chapter, else
    1), `?block=`/`?seg=` for a line to scroll to, `?speaker=` for a one-character filter. */
export interface ReviewLocation {
  chapter: number;
  /** true when the chapter came from `?ch=` itself (an explicit choice) */
  explicitChapter: boolean;
  target: LineRef | null;
  speaker: string | null;
}

const positiveInt = (raw: string | null): number | null => {
  if (raw === null || raw.trim() === "") return null;
  const n = Number(raw);
  return Number.isInteger(n) && n >= 1 ? n : null;
};

export function readLocation(params: URLSearchParams): ReviewLocation {
  const block = params.get("block") || null;
  const segRaw = params.get("seg");
  const seg = segRaw !== null && /^\d+$/.test(segRaw) ? Number(segRaw) : null;
  const ch = positiveInt(params.get("ch"));
  return {
    chapter: ch ?? chapterOfBlock(block) ?? 1,
    explicitChapter: ch !== null,
    target: block ? { block, seg } : null,
    speaker: params.get("speaker") || null,
  };
}

/** Does a row belong to the one-character view? ("narration" = lines with no speaker) */
export function matchesSpeaker(row: SegmentRow, speaker: string | null): boolean {
  if (speaker === null) return true;
  if (speaker === "narration") return row.speaker === null;
  return row.speaker === speaker;
}

/** The row a LineRef points at (its first line when no index is given). */
export function findLine(rows: readonly SegmentRow[], ref: LineRef): SegmentRow | undefined {
  return rows.find((r) => r.block_id === ref.block && (ref.seg === null || r.segment_index === ref.seg));
}

/** The first row AFTER `after` (or from the top, when null or not on this page) that `keep`
    accepts. Rows are in reading order. */
export function nextInChapter(
  rows: readonly SegmentRow[],
  after: LineRef | null,
  keep: (row: SegmentRow) => boolean,
): SegmentRow | null {
  let start = 0;
  if (after) {
    const at = rows.findIndex(
      (r) => r.block_id === after.block && (after.seg === null || r.segment_index === after.seg),
    );
    if (at >= 0) start = at + 1;
  }
  for (let i = start; i < rows.length; i++) if (keep(rows[i])) return rows[i];
  return null;
}

export type WalkResult =
  | { kind: "found"; chapter: number; row: SegmentRow }
  | { kind: "end" }
  | { kind: "error"; chapter: number; message: string }
  | { kind: "cancelled" };

/** "Next to check": the first line after `from` that `keep` accepts, looking first at the rest
    of `from.chapter` (its full rows, `startRows`) and then lazily at each later chapter through
    `laterRows` — the server's `?low_confidence=` view, so only the lines worth checking cross
    the wire. `laterRows` resolves null for a chapter with nothing to read (front matter); a
    throw stops the walk with a plain message. `cancelled()` is checked after every await so a
    book switch abandons an in-flight walk. `onChapter` reports progress. */
export async function walkToNextCheck({
  chapters,
  from,
  startRows,
  keep,
  laterRows,
  cancelled = () => false,
  onChapter,
}: {
  chapters: readonly number[];
  from: { chapter: number; after: LineRef | null };
  startRows: readonly SegmentRow[] | null;
  keep: (row: SegmentRow) => boolean;
  laterRows: (chapter: number) => Promise<readonly SegmentRow[] | null>;
  cancelled?: () => boolean;
  onChapter?: (chapter: number) => void;
}): Promise<WalkResult> {
  if (startRows) {
    const here = nextInChapter(startRows, from.after, keep);
    if (here) return { kind: "found", chapter: from.chapter, row: here };
  }
  for (const ch of chapters) {
    if (ch < from.chapter || (ch === from.chapter && startRows)) continue;
    onChapter?.(ch);
    let rows: readonly SegmentRow[] | null;
    try {
      rows = await laterRows(ch);
    } catch (e) {
      if (cancelled()) return { kind: "cancelled" };
      return { kind: "error", chapter: ch, message: e instanceof Error ? e.message : String(e) };
    }
    if (cancelled()) return { kind: "cancelled" };
    const hit = rows ? nextInChapter(rows, null, keep) : null;
    if (hit) return { kind: "found", chapter: ch, row: hit };
  }
  return { kind: "end" };
}
