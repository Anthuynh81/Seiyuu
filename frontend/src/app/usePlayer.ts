import { createContext, useContext } from "react";

/** One playable unit = one rendered wav (a manifest segment). Words carry interpolated
    start offsets WITHIN the clip so the read-along can highlight and seek by word. */
export interface PlayWord {
  el: HTMLElement;
  offset: number; // seconds from the clip start
}
export interface PlayClip {
  src: string;
  duration: number;
  blockId: string;
  speaker: string;
  words: PlayWord[]; // may be empty (no per-word highlight, still plays)
}

export interface PlayerApi {
  bookId: string | null;
  chapterTitle: string;
  index: number;
  clips: PlayClip[];
  playing: boolean;
  clipElapsed: number;
  load: (bookId: string, chapterTitle: string, clips: PlayClip[], startIndex?: number) => void;
  toggle: () => void;
  seekClip: (index: number, offset?: number) => void;
  seekFraction: (frac: number) => void;
  setVolume: (v: number) => void;
  volume: number;
  audio: HTMLAudioElement | null;
  totalDuration: number;
  clear: () => void;
}

export const PlayerContext = createContext<PlayerApi | null>(null);
export const usePlayer = () => useContext(PlayerContext);
