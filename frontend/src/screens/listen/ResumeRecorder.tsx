import { useEffect, useRef } from "react";

import { usePlayer, usePlayerTime } from "../../app/usePlayer";
import { chapterSeconds, resumeKey, type ResumePoint } from "../../lib/listen";
import { writeStored } from "../../lib/storage";
import { listenHeldChapter } from "./held";

/** Remembers where this viewer is in the book (a per-viewer convenience: localStorage via
    lib/storage, so a blocked store just means "not remembered"). Renders nothing. It is its own
    component so only IT re-renders on the player's ~4/s clock tick, never the whole page.

    Records only while this book's Listen chapter is actually PLAYING: opening a book loads its
    first chapter paused at 0:00, and that must never overwrite the place the listener stopped —
    nor may a one-line clip played from Review. */
export function ResumeRecorder({ bookId }: { bookId: string }) {
  const player = usePlayer();
  const elapsed = usePlayerTime();
  const lastWrite = useRef<string>("");

  useEffect(() => {
    if (!player || !player.playing) return;
    const ch = listenHeldChapter(player, bookId);
    const clip = player.clips[player.index];
    if (ch === null || !clip) return;
    const point: ResumePoint = {
      chapter: ch,
      clip: clip.key,
      offset: Math.round(elapsed * 10) / 10,
      seconds: Math.round(chapterSeconds(player.clips.map((c) => c.duration), player.index, elapsed) * 10) / 10,
    };
    // one write per whole second of listening (or a clip/chapter change) is plenty
    const sig = `${point.chapter}|${point.clip}|${Math.floor(point.offset)}`;
    if (sig === lastWrite.current) return;
    lastWrite.current = sig;
    writeStored(resumeKey(bookId), JSON.stringify(point));
  }, [player, elapsed, bookId]);

  return null;
}
