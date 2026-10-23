import type { PlayClip, PlayerApi } from "../../app/usePlayer";

/** Which chapter Listen last handed the player. The player lives above the routes, so it
    keeps playing while the listener follows a fix link to Review or Pronunciation; Listen
    itself unmounts and forgets its chapter. On the way back it asks this instead of
    reloading chapter 1 over the chapter still playing.

    In-memory only, never truth: it counts only while the player still holds the exact clip
    list Listen gave it (compared by identity), so any other load — a Review one-line clip,
    another book — makes it answer null. */
interface ListenHold {
  bookId: string;
  chapter: number;
  clips: readonly PlayClip[];
}

let hold: ListenHold | null = null;

export function holdListenChapter(bookId: string, chapter: number, clips: readonly PlayClip[]): void {
  hold = { bookId, chapter, clips };
}

/** The chapter of `bookId` the player holds from Listen right now, or null. */
export function listenHeldChapter(
  player: Pick<PlayerApi, "bookId" | "clips"> | null | undefined,
  bookId: string | null,
): number | null {
  if (!hold || !player || !bookId) return null;
  if (hold.bookId !== bookId || player.bookId !== bookId) return null;
  if (player.clips.length === 0 || player.clips !== hold.clips) return null;
  return hold.chapter;
}
