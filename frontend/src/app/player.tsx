import { useMemo, useRef, useState } from "react";

import { PlayerContext, type PlayClip, type PlayerApi } from "./usePlayer";

interface PlayerState {
  bookId: string | null;
  chapterTitle: string;
  index: number; // current clip
  clips: PlayClip[];
  playing: boolean;
  clipElapsed: number; // seconds into the current clip
}

export function PlayerProvider({ children }: { children: React.ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [state, setState] = useState<PlayerState>({
    bookId: null,
    chapterTitle: "",
    index: 0,
    clips: [],
    playing: false,
    clipElapsed: 0,
  });
  const [volume, setVolumeState] = useState(() => Number(localStorage.getItem("seiyuu.volume") ?? "0.8"));

  const ensureAudio = () => {
    if (!audioRef.current) {
      const el = new Audio();
      el.volume = volume;
      el.addEventListener("timeupdate", () => setState((s) => ({ ...s, clipElapsed: el.currentTime })));
      el.addEventListener("ended", () =>
        setState((s) => {
          const next = s.index + 1;
          if (next >= s.clips.length) return { ...s, playing: false, clipElapsed: 0 };
          el.src = s.clips[next].src;
          el.play().catch(() => {});
          return { ...s, index: next, clipElapsed: 0 };
        }),
      );
      audioRef.current = el;
    }
    return audioRef.current;
  };

  const api = useMemo<PlayerApi>(() => {
    const cumulative = () => {
      const offsets: number[] = [];
      let sum = 0;
      for (const c of state.clips) {
        offsets.push(sum);
        sum += c.duration;
      }
      return { offsets, total: sum };
    };
    const playFrom = (index: number, offset: number, play: boolean) => {
      const el = ensureAudio();
      const clip = state.clips[index];
      if (!clip) return;
      if (!el.src.endsWith(clip.src)) el.src = clip.src;
      el.currentTime = Math.min(offset, Math.max(0, clip.duration - 0.05));
      if (play) el.play().catch(() => {});
      setState((s) => ({ ...s, index, clipElapsed: offset, playing: play }));
    };
    return {
      ...state,
      volume,
      audio: audioRef.current,
      totalDuration: cumulative().total,
      load: (bookId, chapterTitle, clips, startIndex = 0) => {
        const el = ensureAudio();
        if (clips.length) {
          el.src = clips[startIndex]?.src ?? clips[0].src;
          el.currentTime = 0;
        }
        setState({ bookId, chapterTitle, index: startIndex, clips, playing: false, clipElapsed: 0 });
      },
      toggle: () => {
        const el = ensureAudio();
        if (!state.clips.length) return;
        if (el.paused) {
          if (!el.src) el.src = state.clips[state.index].src;
          el.play().catch(() => {});
          setState((s) => ({ ...s, playing: true }));
        } else {
          el.pause();
          setState((s) => ({ ...s, playing: false }));
        }
      },
      seekClip: (index, offset = 0) => playFrom(index, offset, true),
      seekFraction: (frac) => {
        const { offsets, total } = cumulative();
        const target = frac * total;
        let idx = offsets.findIndex((_, i) => target < (offsets[i + 1] ?? total));
        if (idx < 0) idx = state.clips.length - 1;
        playFrom(idx, target - offsets[idx], true);
      },
      setVolume: (v) => {
        setVolumeState(v);
        localStorage.setItem("seiyuu.volume", String(v));
        if (audioRef.current) audioRef.current.volume = v;
      },
      clear: () => {
        audioRef.current?.pause();
        setState((s) => ({ ...s, playing: false }));
      },
    };
    // ensureAudio is a stable closure over the ref; excluded intentionally.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, volume]);

  return <PlayerContext.Provider value={api}>{children}</PlayerContext.Provider>;
}
