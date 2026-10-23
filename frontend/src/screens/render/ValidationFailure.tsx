import { useState } from "react";
import { Link } from "react-router-dom";

import type { ValidationRow } from "../../api/types";
import { lexiconHref, reviewLineHref } from "../../lib/listen";
import { misheardWord } from "../../lib/render";

/** One line the audio check flagged: what the book says, what the check heard, the plain
    reason, and the three ways out — look at the line in Review, teach the word on the
    Pronunciation screen, or set up a fresh recording of its chapter (the screen confirms
    that and still shows the price; nothing records from here). */
export function ValidationFailure({
  bookId,
  row,
  voiceName,
  onRemake,
  respelled,
}: {
  bookId: string;
  row: ValidationRow;
  /** the reading voice's display name, when the library knows it */
  voiceName?: string;
  /** set up a forced re-recording of this chapter (opens the screen's confirm) */
  onRemake?: (chapter: number) => void;
  /** words of the book's lexicon respellings: already taught, never suggested again */
  respelled?: ReadonlySet<string>;
}) {
  const [playing, setPlaying] = useState(false);
  const src = `/api/books/${bookId}/segments/${row.block_id}/audio?segment=${row.segment_index}`;
  const word = misheardWord(row.expected, row.transcript, row.reason, respelled);
  return (
    <div className="valrow">
      <div className="vhead">
        <i className="led warn" />
        <span title={`${row.block_id}[${row.segment_index}]`}>Chapter {row.chapter_index}</span>
        <span>· score {row.score.toFixed(2)}</span>
        {(voiceName ?? row.voice_id) && <span className="text-ink-3">· {voiceName ?? row.voice_id}</span>}
        <button className="key quiet ml-auto px-2 py-[2px]" onClick={() => setPlaying(!playing)}>
          {playing ? "hide player" : "▶ play line"}
        </button>
      </div>
      {/* the scorer's plain-words why (wrong number, dropped words, ...) — the score alone
          can't say it: a high-similarity read still fails when it gets a number wrong */}
      {row.reason && <p className="mb-2 text-caution">{row.reason}</p>}
      <div className="vdiff">
        <div className="paper exp"><span className="cap text-paper-ink-2">the book says</span>{row.expected}</div>
        <div className="got"><span className="cap text-ink-3">the check heard</span>{row.transcript}</div>
      </div>
      {playing && <audio controls autoPlay src={src} className="mt-2 h-8 w-full" />}
      <div className="mt-2 flex flex-wrap items-center gap-2" role="group" aria-label="fix this line">
        <Link className="key quiet px-2 py-[2px]" to={reviewLineHref(bookId, row.chapter_index, row)}>
          Open in Review
        </Link>
        <Link className="key quiet px-2 py-[2px]" to={lexiconHref(bookId, word)}>
          {word ? `Fix pronunciation of “${word}”` : "Fix pronunciation"}
        </Link>
        {onRemake && (
          <button className="key quiet px-2 py-[2px]" onClick={() => onRemake(row.chapter_index)}>
            Re-make chapter {row.chapter_index}
          </button>
        )}
      </div>
    </div>
  );
}
