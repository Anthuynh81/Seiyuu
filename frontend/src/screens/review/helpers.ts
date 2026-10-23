import { useEffect, useState } from "react";

import type { SegmentRow } from "../../api/types";
import { readStored, writeStored } from "../../lib/storage";

/** TalkSelect keys are strings; this sentinel stands in for "no voice / narration". */
export const NONE = "__none__";

/** The review-queue predicate, shared by the chapter queue count and each row's low flag.
 * Unattributed quotes (speaker null but the text is a quoted span) are queue material too —
 * they render in the narrator's voice, which is exactly what needs eyes. */
export const isReviewable = (s: SegmentRow, threshold: number) =>
  (s.speaker !== null || s.unattributed_quote) && s.confidence < threshold;

/* -------------------------------------------------- frontier (localStorage, per book) */

export function useFrontier(bookId: string | null): [number, (n: number) => void] {
  const key = `seiyuu.frontier.${bookId}`;
  const [value, setValue] = useState(() => Number(readStored(key)) || 1);
  // The initializer runs once per mount; on a book switch re-read the NEW book's frontier
  // (otherwise book A's frontier unmasks book B's late-debut characters — spoiler leak).
  useEffect(() => {
    setValue(Number(readStored(key)) || 1);
  }, [key]);
  return [
    value,
    (n: number) => {
      setValue(n);
      writeStored(key, String(n));
    },
  ];
}

/* -------------------------------------------------- spoiler-safe (localStorage, per viewer) */

const SPOILER_KEY = "seiyuu.review.spoilerSafe";

/** Spoiler-safe hides characters who first appear beyond the reading frontier. OFF unless this
    viewer turned it on (owner decision); remembered per viewer, not per book. Storage that
    refuses just means the choice lasts this page only. */
export function useSpoilerSafe(): [boolean, (on: boolean) => void] {
  const [on, setOn] = useState(() => readStored(SPOILER_KEY) === "1");
  return [
    on,
    (next: boolean) => {
      setOn(next);
      writeStored(SPOILER_KEY, next ? "1" : "0");
    },
  ];
}
