import { act, render } from "@testing-library/react";
import { StrictMode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../test/utils";
import { PlayerProvider } from "./player";
import { usePlayer, type PlayClip, type PlayerApi } from "./usePlayer";

/** jsdom's HTMLMediaElement never flips `paused` (the setup shims make play/pause safe
    no-ops), but toggle() branches on it — so mirror play/pause into a controllable
    `paused` here. File-scoped: vitest isolates jsdom per test file. */
let mediaPaused = true;
const playSpy = vi.fn(function play() {
  mediaPaused = false;
  return Promise.resolve();
});
const pauseSpy = vi.fn(function pause() {
  mediaPaused = true;
});
Object.defineProperty(HTMLMediaElement.prototype, "play", { configurable: true, writable: true, value: playSpy });
Object.defineProperty(HTMLMediaElement.prototype, "pause", { configurable: true, writable: true, value: pauseSpy });
Object.defineProperty(HTMLMediaElement.prototype, "paused", { configurable: true, get: () => mediaPaused });

beforeEach(() => {
  mediaPaused = true;
  playSpy.mockClear();
  pauseSpy.mockClear();
});

const clip = (src: string, duration: number): PlayClip => ({
  src,
  duration,
  key: src,
  speaker: "Narrator",
  words: [],
});

/** Mount the real PlayerProvider (via the app's provider stack) and expose the live api
    through a probe consumer; `holder.api` always reflects the latest render. */
function renderPlayer(): { api: PlayerApi } {
  const holder: { api?: PlayerApi } = {};
  function Probe() {
    const api = usePlayer();
    if (!api) throw new Error("usePlayer() returned null — PlayerProvider missing");
    holder.api = api;
    return null;
  }
  renderWithProviders(<Probe />);
  if (!holder.api) throw new Error("probe never rendered");
  return holder as { api: PlayerApi };
}

const absolute = (src: string) => new URL(src, window.location.href).href;

describe("PlayerProvider", () => {
  it("load with autoplay points the audio element at the first clip and starts playback", () => {
    const player = renderPlayer();
    const clips = [clip("/audio/a.wav", 10), clip("/audio/b.wav", 20)];

    act(() => player.api.load("b1", "Chapter 1", clips, { autoplay: true }));

    const audio = player.api.audio;
    expect(audio).not.toBeNull();
    expect(audio!.src).toBe(absolute("/audio/a.wav"));
    expect(playSpy).toHaveBeenCalled();
    expect(player.api.playing).toBe(true);
    expect(player.api.index).toBe(0);
    expect(player.api.bookId).toBe("b1");
    expect(player.api.totalDuration).toBe(30);
  });

  it("an 'ended' event advances to the next clip and keeps playing", () => {
    const player = renderPlayer();
    act(() => player.api.load("b1", "ch", [clip("/audio/a.wav", 10), clip("/audio/b.wav", 20)], { autoplay: true }));
    const audio = player.api.audio!;
    playSpy.mockClear();

    act(() => {
      audio.dispatchEvent(new Event("ended"));
    });

    expect(player.api.index).toBe(1);
    expect(player.api.playing).toBe(true);
    expect(audio.src).toBe(absolute("/audio/b.wav"));
    expect(playSpy).toHaveBeenCalled();
  });

  it("'ended' on the last clip fires the load-time onEnded callback and stops playback", () => {
    const player = renderPlayer();
    const onEnded = vi.fn();
    act(() =>
      player.api.load("b1", "ch", [clip("/audio/a.wav", 10), clip("/audio/b.wav", 20)], {
        autoplay: true,
        onEnded,
      }),
    );
    const audio = player.api.audio!;

    act(() => {
      audio.dispatchEvent(new Event("ended")); // a -> b
    });
    expect(onEnded).not.toHaveBeenCalled();

    act(() => {
      audio.dispatchEvent(new Event("ended")); // past the end
    });
    expect(onEnded).toHaveBeenCalledTimes(1);
    expect(player.api.playing).toBe(false);
    expect(player.api.index).toBe(1); // stays on the last clip
  });

  it("StrictMode double-invocation cannot double-advance or double-fire onEnded (side effects live outside updaters)", () => {
    // main.tsx wraps the app in StrictMode, whose dev build double-invokes setState
    // updaters — side effects inside one (play(), onEnded) would fire twice. The shared
    // harness deliberately omits StrictMode, so mount the provider under it directly.
    const holder: { api?: PlayerApi } = {};
    function Probe() {
      holder.api = usePlayer() ?? undefined;
      return null;
    }
    render(
      <StrictMode>
        <PlayerProvider>
          <Probe />
        </PlayerProvider>
      </StrictMode>,
    );
    const onEnded = vi.fn();
    act(() =>
      holder.api!.load("b1", "ch", [clip("/audio/a.wav", 10), clip("/audio/b.wav", 20)], { onEnded }),
    );
    const audio = holder.api!.audio!;
    playSpy.mockClear();

    // mid-chapter advance: exactly one play() per ended event
    act(() => {
      audio.dispatchEvent(new Event("ended"));
    });
    expect(holder.api!.index).toBe(1);
    expect(playSpy).toHaveBeenCalledTimes(1);

    // chapter end: the completion callback (auto-advance to the next chapter) fires once
    act(() => {
      audio.dispatchEvent(new Event("ended"));
    });
    expect(onEnded).toHaveBeenCalledTimes(1);
    expect(holder.api!.playing).toBe(false);
  });

  it("toggle pauses playback and toggling again resumes it", () => {
    const player = renderPlayer();
    act(() => player.api.load("b1", "ch", [clip("/audio/a.wav", 10)], { autoplay: true }));
    expect(player.api.playing).toBe(true);
    pauseSpy.mockClear();

    act(() => player.api.toggle());
    expect(pauseSpy).toHaveBeenCalled();
    expect(player.api.playing).toBe(false);

    playSpy.mockClear();
    act(() => player.api.toggle());
    expect(playSpy).toHaveBeenCalled();
    expect(player.api.playing).toBe(true);
  });

  it("seekFraction lands in the right clip at the right offset once metadata loads", () => {
    const player = renderPlayer();
    // durations 10 + 20 + 30 = 60s; fraction 0.4 => 24s => clip 1 at 14s in
    const clips = [clip("/audio/a.wav", 10), clip("/audio/b.wav", 20), clip("/audio/c.wav", 30)];
    act(() => player.api.load("b1", "ch", clips, { autoplay: false }));
    const audio = player.api.audio!;

    act(() => player.api.seekFraction(0.4));
    expect(player.api.index).toBe(1);
    expect(player.api.clipElapsed).toBe(14);
    expect(audio.src).toBe(absolute("/audio/b.wav"));

    // the seek is gated on loadedmetadata (seeking an unloaded src is ignored)
    playSpy.mockClear();
    act(() => {
      audio.dispatchEvent(new Event("loadedmetadata"));
    });
    expect(audio.currentTime).toBe(14);
    expect(playSpy).toHaveBeenCalled();
    expect(player.api.playing).toBe(true);
  });

  it("volume is read from localStorage on mount and setVolume persists + applies it", () => {
    localStorage.setItem("seiyuu.volume", "0.5");
    const player = renderPlayer();
    expect(player.api.volume).toBe(0.5);

    act(() => player.api.load("b1", "ch", [clip("/audio/a.wav", 10)]));
    expect(player.api.audio!.volume).toBe(0.5);

    act(() => player.api.setVolume(0.3));
    expect(localStorage.getItem("seiyuu.volume")).toBe("0.3");
    expect(player.api.volume).toBe(0.3);
    expect(player.api.audio!.volume).toBe(0.3);
  });

  it("rate is read from localStorage on mount and setRate persists + applies it", () => {
    localStorage.setItem("seiyuu.rate", "1.5");
    const player = renderPlayer();
    expect(player.api.rate).toBe(1.5);

    act(() => player.api.load("b1", "ch", [clip("/audio/a.wav", 10)]));
    expect(player.api.audio!.playbackRate).toBe(1.5);
    // defaultPlaybackRate is what survives the src swap on clip advance
    expect(player.api.audio!.defaultPlaybackRate).toBe(1.5);

    act(() => player.api.setRate(2));
    expect(localStorage.getItem("seiyuu.rate")).toBe("2");
    expect(player.api.rate).toBe(2);
    expect(player.api.audio!.playbackRate).toBe(2);
    expect(player.api.audio!.defaultPlaybackRate).toBe(2);
  });

  it("re-loading the SAME clip list (a background refetch) neither pauses nor rewinds playback", () => {
    const player = renderPlayer();
    const clips = () => [clip("/audio/a.wav", 10), clip("/audio/b.wav", 20)];
    act(() => player.api.load("b1", "ch", clips(), { autoplay: true }));
    act(() => {
      player.api.audio!.dispatchEvent(new Event("ended")); // now mid-chapter on clip b
    });
    expect(player.api.index).toBe(1);
    pauseSpy.mockClear();
    playSpy.mockClear();

    // a window-refocus refetch rebuilds the page and calls load again with identical srcs
    const onEnded = vi.fn();
    act(() => player.api.load("b1", "ch", clips(), { autoplay: false, onEnded }));

    expect(pauseSpy).not.toHaveBeenCalled();
    expect(player.api.playing).toBe(true);
    expect(player.api.index).toBe(1); // place in the chapter kept
    expect(player.api.audio!.src).toBe(absolute("/audio/b.wav"));

    // the fresh onEnded took over: finishing the chapter fires it
    act(() => {
      player.api.audio!.dispatchEvent(new Event("ended"));
    });
    expect(onEnded).toHaveBeenCalledTimes(1);
  });

  it("re-loading with DIFFERENT clips (a real navigation) restarts from the top", () => {
    const player = renderPlayer();
    act(() => player.api.load("b1", "ch1", [clip("/audio/a.wav", 10)], { autoplay: true }));

    act(() => player.api.load("b1", "ch2", [clip("/audio/c.wav", 30)], { autoplay: false }));
    expect(player.api.index).toBe(0);
    expect(player.api.playing).toBe(false);
    expect(player.api.audio!.src).toBe(absolute("/audio/c.wav"));
  });

  it("clear pauses the audio and flips playing off", () => {
    const player = renderPlayer();
    act(() => player.api.load("b1", "ch", [clip("/audio/a.wav", 10)], { autoplay: true }));
    expect(player.api.playing).toBe(true);
    pauseSpy.mockClear();

    act(() => player.api.clear());
    expect(pauseSpy).toHaveBeenCalled();
    expect(player.api.playing).toBe(false);
  });
});
