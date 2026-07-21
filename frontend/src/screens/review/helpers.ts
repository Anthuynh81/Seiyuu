import { useEffect, useState } from "react";

import type { SegmentRow } from "../../api/types";

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
  const [value, setValue] = useState(() => Number(localStorage.getItem(key)) || 1);
  // The initializer runs once per mount; on a book switch re-read the NEW book's frontier
  // (otherwise book A's frontier unmasks book B's late-debut characters — spoiler leak).
  useEffect(() => {
    setValue(Number(localStorage.getItem(key)) || 1);
  }, [key]);
  return [
    value,
    (n: number) => {
      setValue(n);
      localStorage.setItem(key, String(n));
    },
  ];
}
