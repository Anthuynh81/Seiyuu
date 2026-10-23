import { useEffect, useRef, useState } from "react";

import { ApiError } from "../../api/client";
import { useRecordEdit } from "../../api/hooks";
import { JobErrorText } from "../../components/JobErrorText";
import { explainJobError } from "../../lib/jobErrors";
import { mergePreview, type MergeSuggestion } from "../../lib/reviewTools";
import { readStored, writeStored } from "../../lib/storage";

/* -------------------------------------------------- "same person?" cards */

const SHOWN = 3;
const dismissKey = (bookId: string) => `seiyuu.review.notSame.${bookId}`;

/** "Not the same person" is a per-viewer hint dismissal (a convenience, so localStorage):
    the server keeps no record of it, and a browser that refuses storage just shows the card
    again next visit. */
function readDismissed(bookId: string): Set<string> {
  try {
    const raw = readStored(dismissKey(bookId));
    const list: unknown = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(list) ? list.filter((x): x is string => typeof x === "string") : []);
  } catch {
    return new Set();
  }
}

/** Possible duplicates from the attribution's registry notes, one card per pair: what Seiyuu
    noticed, what a merge would move, and a Merge key that records the existing merge edit.
    The caller leaves out pairs involving a character spoiler-safe hides. */
export function MergeSuggestions({ bookId, suggestions }: { bookId: string; suggestions: MergeSuggestion[] }) {
  const [dismissed, setDismissed] = useState(() => readDismissed(bookId));
  useEffect(() => setDismissed(readDismissed(bookId)), [bookId]);
  const [showAll, setShowAll] = useState(false);
  // what the last Merge / Not the same person did. A card that leaves takes keyboard focus with
  // it, so focus moves to this line, which keeps the section up even when no card is left.
  const [notice, setNotice] = useState<string | null>(null);
  const [focusNotice, setFocusNotice] = useState(0);
  const noticeRef = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    if (focusNotice > 0) noticeRef.current?.focus();
  }, [focusNotice]);
  const open = suggestions.filter((s) => !dismissed.has(s.key));
  if (open.length === 0 && notice === null) return null;
  const say = (text: string) => {
    setNotice(text);
    setFocusNotice((n) => n + 1);
  };
  const dismiss = (s: MergeSuggestion) => {
    const next = new Set(dismissed).add(s.key);
    setDismissed(next);
    writeStored(dismissKey(bookId), JSON.stringify([...next]));
    say(`${s.loser.name} and ${s.winner.name} stay two people.`);
  };
  const shown = showAll ? open : open.slice(0, SHOWN);
  return (
    <section className="mergecards" aria-label="possible duplicates">
      <span className="tag">same person?</span>
      <span className="mono ml-2 text-[11px] text-ink-3">{open.length}</span>
      <div aria-live="polite">
        {notice && (
          <p className="navnote mt-2" ref={noticeRef} tabIndex={-1}>
            {notice}
            {open.length === 0 ? " No more suggestions." : ""}
          </p>
        )}
      </div>
      {shown.map((s) => (
        <MergeCard
          key={s.key}
          bookId={bookId}
          suggestion={s}
          onDismiss={() => dismiss(s)}
          onMerged={(text) => say(text)}
        />
      ))}
      {open.length > SHOWN && (
        <button className="link mt-2 text-[12px]" onClick={() => setShowAll(!showAll)}>
          {showAll ? "show fewer" : `show all ${open.length}`}
        </button>
      )}
    </section>
  );
}

function MergeCard({
  bookId,
  suggestion,
  onDismiss,
  onMerged,
}: {
  bookId: string;
  suggestion: MergeSuggestion;
  onDismiss: () => void;
  onMerged: (text: string) => void;
}) {
  const record = useRecordEdit(bookId);
  // which way the lines move: the less busy into the busier by default; one click flips it
  const [flipped, setFlipped] = useState(false);
  const loser = flipped ? suggestion.winner : suggestion.loser;
  const winner = flipped ? suggestion.loser : suggestion.winner;
  const refused = record.error
    ? explainJobError(
        record.error instanceof ApiError
          ? { code: record.error.code, message: record.error.message }
          : record.error.message,
      )
    : null;
  return (
    <div className="mergecard">
      <b>
        {suggestion.loser.name} and {suggestion.winner.name}
      </b>
      <p>{suggestion.why}</p>
      <p>{mergePreview(loser, winner.name)}</p>
      {refused && (
        <div className="refusal mt-2">
          <span className="tag">couldn't merge</span>
          <JobErrorText explained={refused} />
        </div>
      )}
      <div className="acts">
        <button
          className="key px-3 py-[3px]"
          disabled={record.isPending}
          onClick={() =>
            // mutateAsync, not mutate's callbacks: the refetch removes this card before they
            // would run, and an unmounted observer drops them. A refusal shows via record.error.
            record
              .mutateAsync({ op: "merge", loser_id: loser.id, winner_id: winner.id })
              .then(() => onMerged(`Merged ${loser.name} into ${winner.name}.`))
              .catch(() => undefined)
          }
        >
          {record.isPending ? "merging…" : `Merge into ${winner.name}`}
        </button>
        <button className="key quiet px-2 py-[3px]" disabled={record.isPending} onClick={() => setFlipped(!flipped)}>
          keep {loser.name} instead
        </button>
        <button className="key quiet px-2 py-[3px]" disabled={record.isPending} onClick={onDismiss}>
          Not the same person
        </button>
      </div>
    </div>
  );
}
