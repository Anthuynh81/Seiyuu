import type { AttributedChapterView, EmotionVerdict } from "../api/types";

/** The closed emotion taxonomy, ordered as in the backend enum. */
export const EMOTION_LABELS = [
  "neutral",
  "happy",
  "sad",
  "angry",
  "fearful",
  "tender",
  "tense",
] as const;

/** The key a chip is looked up by: a segment's block id + its ordinal WITHIN that block —
    exactly the segment browser's `segment_index`. */
export function emotionKey(blockId: string, ordinal: number): string {
  return `${blockId}:${ordinal}`;
}

/** Small dots for the 3-level intensity (clamped 1–3). */
export function intensityDots(intensity: number): string {
  return "•".repeat(Math.min(Math.max(Math.round(intensity), 1), 3));
}

/** Build `(block_id:ordinal) -> emotion verdict` for one chapter's attribution report.

    The ordinal is the segment's position within its block, enumerated in report order — the
    SAME derivation the segment browser uses for `segment_index`, so each chip lines up with its
    row. NEUTRAL and null are skipped: they degrade to no render override, so surfacing them
    would only clutter the margin. A `segment_emotions` list whose length doesn't match
    `segments` (a pre-emotion v3/v4 report has an empty one) is treated as "no tags" and yields
    an empty map — never a misaligned chip. */
export function buildEmotionMap(
  chapter: AttributedChapterView | undefined,
): Map<string, EmotionVerdict> {
  const map = new Map<string, EmotionVerdict>();
  if (!chapter) return map;
  const emotions = chapter.segment_emotions;
  if (emotions.length !== chapter.segments.length) return map; // desync guard: treat as none
  const ordinalOf: Record<string, number> = {};
  chapter.segments.forEach((seg, i) => {
    const ordinal = ordinalOf[seg.block_id] ?? 0;
    ordinalOf[seg.block_id] = ordinal + 1;
    const emo = emotions[i];
    if (emo && emo.label !== "neutral") map.set(emotionKey(seg.block_id, ordinal), emo);
  });
  return map;
}
