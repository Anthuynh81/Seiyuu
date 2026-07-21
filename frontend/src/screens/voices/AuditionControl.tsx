import { useEffect, useRef, useState } from "react";

import { ApiError } from "../../api/client";
import { useAudition, useWarmup } from "../../api/hooks";
import type { VoiceOut } from "../../api/types";
import { BORROW_RETRY_MAX } from "./helpers";

/* -------------------------------------------------- audition control */

export function AuditionControl({ voice }: { voice: VoiceOut }) {
  const audition = useAudition(voice.voice_id);
  const warmup = useWarmup();
  const [playerOpen, setPlayerOpen] = useState(false);
  // Cache-buster latched once per successful audition. Date.now() inline in the src
  // would mint a new URL every parent re-render (the 2s job poll), restarting playback.
  const [take, setTake] = useState(0);
  useEffect(() => {
    if (audition.isSuccess) setTake(Date.now());
  }, [audition.isSuccess]);
  const err = audition.error instanceof ApiError ? audition.error : null;

  // gpu_busy_retry is a SOFT refusal: a render is lending the GPU between segments, so the
  // right move is to wait and retry — not to give up. Auto-retry with a short backoff, up to
  // BORROW_RETRY_MAX, preserving the same confirm_paid the user chose.
  const borrowRetries = useRef(0);
  useEffect(() => {
    if (audition.isSuccess) borrowRetries.current = 0;
  }, [audition.isSuccess]);
  useEffect(() => {
    if (err?.code !== "gpu_busy_retry" || borrowRetries.current >= BORROW_RETRY_MAX) return;
    borrowRetries.current += 1;
    const confirmPaid = audition.variables ?? false;
    const timer = setTimeout(() => audition.mutate(confirmPaid), 500 * borrowRetries.current);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [err]);
  const startAudition = (confirmPaid: boolean) => {
    borrowRetries.current = 0; // a fresh user-initiated attempt resets the borrow budget
    audition.mutate(confirmPaid);
  };

  if (audition.isPending) {
    return (
      <div className="audit">
        <span className="stop" />
        <span className="lbl">auditioning…</span>
        <span className="live" />
      </div>
    );
  }

  // soft borrow-retry: while attempts remain, show a "queued behind a render" spinner rather
  // than a refusal — the render keeps running and the audition slips in between its segments
  if (err?.code === "gpu_busy_retry" && borrowRetries.current < BORROW_RETRY_MAX) {
    return (
      <div className="audit" title="a render is lending the GPU between segments — it keeps running">
        <span className="stop" />
        <span className="lbl">queued behind a render segment — retrying…</span>
        <span className="live" />
      </div>
    );
  }

  if (err) {
    const detail = (err.detail ?? {}) as Record<string, unknown>;
    const recourse = (() => {
      switch (err.code) {
        case "engine_cold":
          return (
            <button
              className="link"
              disabled={warmup.isPending}
              onClick={() => warmup.mutate(voice.engine, { onSuccess: () => audition.reset() })}
            >
              {warmup.isPending ? "starting warmup…" : "warm up first"}
            </button>
          );
        case "payment_confirmation_required":
          return (
            <button className="link" onClick={() => startAudition(true)}>
              confirm ~${Number(detail.estimated_usd ?? 0).toFixed(4)} &amp; play
            </button>
          );
        case "gpu_busy_retry":
          // auto-retries exhausted — the render is still holding the GPU; let the user retry
          return (
            <span>
              the render keeps running — it hasn't yielded the GPU yet;{" "}
              <button className="link" onClick={() => startAudition(audition.variables ?? false)}>
                retry
              </button>
            </span>
          );
        case "gpu_busy":
        case "cloud_busy":
          return <span>wait for the job in the transport bar, or cancel it — <button className="link" onClick={() => audition.reset()}>retry</button></span>;
        default:
          return (
            <button className="link" onClick={() => audition.reset()}>
              dismiss
            </button>
          );
      }
    })();
    return (
      <div className="refusal">
        <span className="tag">{err.code}</span>
        <p>
          {err.message} — {recourse}
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="audit cursor-pointer" role="button" tabIndex={0} onClick={() => startAudition(false)}>
        <span className="play" />
        <span className="lbl">audition</span>
        {voice.has_audition && (
          <button
            className="key quiet ml-auto px-2 py-px text-[10.5px]"
            onClick={(e) => {
              e.stopPropagation();
              setPlayerOpen(!playerOpen);
            }}
          >
            {playerOpen ? "hide last" : "▶ last take"}
          </button>
        )}
      </div>
      {(playerOpen || audition.isSuccess) && voice.has_audition && (
        <audio
          controls
          autoPlay={audition.isSuccess}
          src={`/api/voices/${voice.voice_id}/audition.wav?t=${take}`}
          className="mt-2 h-[30px] w-full"
        />
      )}
    </div>
  );
}
