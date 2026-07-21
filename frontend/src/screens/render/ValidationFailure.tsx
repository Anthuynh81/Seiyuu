import { useState } from "react";

import type { ValidationRow } from "../../api/types";

export function ValidationFailure({ bookId, row }: { bookId: string; row: ValidationRow }) {
  const [playing, setPlaying] = useState(false);
  const src = `/api/books/${bookId}/segments/${row.block_id}/audio?segment=${row.segment_index}`;
  return (
    <div className="valrow">
      <div className="vhead">
        <i className="led warn" />
        ch{row.chapter_index} · {row.block_id}[{row.segment_index}] · score {row.score.toFixed(2)}
        {row.voice_id && <span className="text-ink-3">· {row.voice_id}</span>}
        <button className="key quiet ml-auto px-2 py-[2px]" onClick={() => setPlaying(!playing)}>
          {playing ? "hide player" : "▶ play segment"}
        </button>
      </div>
      <div className="vdiff">
        <div className="paper exp"><span className="cap text-paper-ink-2">expected (book)</span>{row.expected}</div>
        <div className="got"><span className="cap text-ink-3">whisper heard</span>{row.transcript}</div>
      </div>
      {playing && <audio controls autoPlay src={src} className="mt-2 h-8 w-full" />}
    </div>
  );
}
