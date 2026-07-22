import { useEffect, useRef, useState } from "react";

import { BORROW_RETRY_MAX } from "./helpers";

/* -------------------------------------------------- preview demos (the mixer's ear) */

/** Play a kokoro preview. Fetch first — refusals (gpu_busy, engine_cold…) come back as
    JSON envelopes, which an <audio src> would swallow silently. */
export function useDemoPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null); // the object URL currently backing audioRef
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false); // waiting out a render's GPU borrow

  // pause the prior element and revoke its blob URL — object URLs live until revoked, so tuning
  // a blend (a preview per click) would otherwise leak a blob every click.
  const release = () => {
    audioRef.current?.pause();
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
  };
  useEffect(() => release, []); // on unmount: stop playback and revoke any outstanding url

  const play = async (url: string, attempt = 0) => {
    setError(null);
    audioRef.current?.pause();
    setBusy(url);
    try {
      const res = await fetch(url);
      if (!res.ok) {
        const body = (await res.json().catch(() => null)) as { error?: { code?: string; message?: string } } | null;
        // gpu_busy_retry is soft: a render is lending the GPU between segments — wait & retry
        if (body?.error?.code === "gpu_busy_retry" && attempt < BORROW_RETRY_MAX) {
          setRetrying(true);
          await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
          return play(url, attempt + 1);
        }
        throw new Error(body?.error?.message ?? `preview failed (${res.status})`);
      }
      setRetrying(false);
      const objectUrl = URL.createObjectURL(await res.blob());
      release(); // free the previous preview's element + url now that this one is ready
      urlRef.current = objectUrl;
      const el = new Audio(objectUrl);
      audioRef.current = el;
      const done = () => {
        setBusy(null);
        // revoke only if we're still the current preview (a newer play() may have taken over)
        if (urlRef.current === objectUrl) {
          URL.revokeObjectURL(objectUrl);
          urlRef.current = null;
        }
      };
      el.onended = done;
      el.onerror = done;
      await el.play();
    } catch (e) {
      setBusy(null);
      setRetrying(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  return { play, busy, error, retrying };
}

export const presetPreviewUrl = (id: string) => `/api/engines/kokoro/preview?preset=${id}`;
export const mixPreviewUrl = (layers: { preset_id: string; weight: number }[]) =>
  `/api/engines/kokoro/preview?components=${layers.map((l) => `${l.preset_id}:${l.weight}`).join(",")}`;
