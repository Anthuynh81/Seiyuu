import { useMemo, useRef, useState } from "react";

import { PlayerContext, type LoadOptions, type PlayClip, type PlayerApi } from "./usePlayer";

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
  const onEndedRef = useRef<(() => void) | undefined>(undefined);
  const [state, setState] = useState<PlayerState>({
    bookId: null,
    chapterTitle: "",
    index: 0,
    clips: [],
    playing: false,
    clipElapsed: 0,
  });
  // The audio element's listeners are attached once but must read the CURRENT clips; the
  // ref keeps them honest without putting side effects inside setState updaters (StrictMode
  // double-invokes updaters in dev, which would double-fire onEnded and double-play).
  const stateRef = useRef(state);
  stateRef.current = state;
  const [volume, setVolumeState] = useState(() => Number(localStorage.getItem("seiyuu.volume") ?? "0.8"));
  const [rate, setRateState] = useState(() => Number(localStorage.getItem("seiyuu.rate")) || 1);
  const rateRef = useRef(rate);
  rateRef.current = rate;

  // a play() the browser refuses (autoplay policy, unloadable src) must not leave the
  // transport claiming "playing" — the key would then look stuck to the user
  const playEl = (el: HTMLAudioElement) =>
    el.play().catch(() => setState((s) => ({ ...s, playing: false })));

  const ensureAudio = () => {
    if (!audioRef.current) {
      const el = new Audio();
      el.volume = volume;
      // defaultPlaybackRate survives src swaps; playbackRate covers the current clip
      el.defaultPlaybackRate = rateRef.current;
      el.playbackRate = rateRef.current;
      el.addEventListener("timeupdate", () => setState((s) => ({ ...s, clipElapsed: el.currentTime })));
      el.addEventListener("ended", () => {
        const s = stateRef.current;
        const next = s.index + 1;
        if (next >= s.clips.length) {
          onEndedRef.current?.();
          setState((cur) => ({ ...cur, playing: false, clipElapsed: 0 }));
          return;
        }
        el.src = s.clips[next].src;
        playEl(el);
        setState((cur) => ({ ...cur, index: next, clipElapsed: 0 }));
      });
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
      const apply = () => {
        // clamp inside the clip; seeking an unloaded source is silently ignored by
        // browsers, hence the loadedmetadata gate below
        try {
          el.currentTime = Math.min(Math.max(offset, 0), Math.max(0, clip.duration - 0.05));
        } catch {
          /* not seekable yet */
        }
        if (play) playEl(el);
      };
      const target = new URL(clip.src, window.location.href).href;
      if (el.src !== target) {
        el.src = clip.src;
        el.addEventListener("loadedmetadata", apply, { once: true });
        el.load();
      } else {
        apply();
      }
      setState((s) => ({ ...s, index, clipElapsed: offset, playing: play }));
    };
    return {
      ...state,
      volume,
      audio: audioRef.current,
      totalDuration: cumulative().total,
      load: (bookId, chapterTitle, clips, opts?: LoadOptions) => {
        const el = ensureAudio();
        onEndedRef.current = opts?.onEnded;
        // Same book, same clip list = a background refetch rebuilt the page, not a real
        // navigation. Swap in the fresh clips (their word spans point at the new DOM) but
        // leave the element — and the listener's place in the chapter — untouched, so a
        // window refocus or query invalidation never stops or rewinds playback.
        const cur = stateRef.current;
        const sameAudio =
          cur.bookId === bookId &&
          cur.clips.length === clips.length &&
          clips.length > 0 &&
          cur.clips.every((c, i) => c.src === clips[i].src);
        if (sameAudio) {
          setState((s) => ({ ...s, chapterTitle, clips }));
          return;
        }
        setState({ bookId, chapterTitle, index: 0, clips, playing: !!opts?.autoplay && clips.length > 0, clipElapsed: 0 });
        if (clips.length) {
          el.src = clips[0].src;
          el.currentTime = 0;
          if (opts?.autoplay) playEl(el);
          else el.pause();
        } else {
          el.pause();
        }
      },
      toggle: () => {
        const el = ensureAudio();
        if (!state.clips.length) return;
        if (el.paused) {
          if (!el.src) el.src = state.clips[state.index].src;
          playEl(el);
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
      rate,
      setRate: (r) => {
        setRateState(r);
        localStorage.setItem("seiyuu.rate", String(r));
        if (audioRef.current) {
          audioRef.current.defaultPlaybackRate = r;
          audioRef.current.playbackRate = r;
        }
      },
      clear: () => {
        audioRef.current?.pause();
        setState((s) => ({ ...s, playing: false }));
      },
    };
    // ensureAudio is a stable closure over the ref; excluded intentionally.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, volume, rate]);

  return <PlayerContext.Provider value={api}>{children}</PlayerContext.Provider>;
}
