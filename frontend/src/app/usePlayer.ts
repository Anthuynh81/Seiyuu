import { createContext, useContext } from "react";

/** One playable unit = one rendered wav (a manifest segment). Words carry interpolated
    start offsets WITHIN the clip so the read-along can highlight and seek by word. */
export interface PlayWord {
  el: HTMLElement;
  offset: number; // seconds from the clip start (when this word begins)
  end: number; // seconds from the clip start (when this word ends) — makes the active-word test exact
}
export interface PlayClip {
  src: string;
  duration: number;
  key: string; // `${block_id}:${audio_segment}` — one clip per rendered wav
  speaker: string;
  words: PlayWord[]; // may be empty (no per-word highlight, still plays)
}

export interface LoadOptions {
  autoplay?: boolean;
  onEnded?: () => void; // the chapter finished end-to-end
}

export interface PlayerApi {
  bookId: string | null;
  chapterTitle: string;
  index: number;
  clips: PlayClip[];
  playing: boolean;
  clipElapsed: number;
  load: (bookId: string, chapterTitle: string, clips: PlayClip[], opts?: LoadOptions) => void;
  toggle: () => void;
  seekClip: (index: number, offset?: number) => void;
  seekFraction: (frac: number) => void;
  setVolume: (v: number) => void;
  volume: number;
  /** playback speed (0.75–2). Persisted; applies to the element immediately. */
  rate: number;
  setRate: (r: number) => void;
  audio: HTMLAudioElement | null;
  totalDuration: number;
  clear: () => void;
}

export const PlayerContext = createContext<PlayerApi | null>(null);
export const usePlayer = () => useContext(PlayerContext);
