import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { ApiError } from "../../api/client";
import { useAudition, useJob, useWarmup } from "../../api/hooks";
import type { JobState } from "../../api/types";
import { explainAuditionRefusal, type AuditionRefusal } from "../../lib/audition";
import { engineName, warmupErrorText } from "../../lib/health";
import { explainJobError } from "../../lib/jobErrors";
import { BORROW_RETRY_MAX } from "./helpers";

/* -------------------------------------------------- one audition flow, every player */

/** Only one sample plays at a time across the app: starting another pauses this one. */
let nowPlaying: HTMLAudioElement | null = null;

export type AuditionPhase =
  | "idle"
  | "making" // the server is synthesizing
  | "waiting" // a render is lending the GPU between segments — bounded auto-retry in progress
  | "refused"
  | "ready"; // a fresh take exists (and, with autoplay, is playing or has played)

export interface AuditionFlow {
  /** the voice's engine id (for wording: "Kokoro isn't downloaded") */
  engine: string;
  phase: AuditionPhase;
  refusal: AuditionRefusal | null;
  /** A user-initiated attempt. `confirmPaid` must come from an explicit click, never a default:
      every play and every "try again" passes false, so a paid voice quotes its price each time. */
  start: (confirmPaid: boolean) => void;
  /** After an exhausted borrow (gpu_busy_retry) ONLY: repeat that attempt as it was sent,
      keeping the confirmation the user already gave for it — it is the same request the soft
      auto-retries were repeating. Every other "try again" is start(false). */
  retryBorrow: () => void;
  dismiss: () => void;
  /** the latest take's URL (cache-busted per success), or null before the first success */
  takeUrl: string | null;
  /** autoplay players: is the take playing right now, and stop it */
  playing: boolean;
  stop: () => void;
  /** engine_cold recourse: queue the engine's download/warm-up job (never retried) */
  warmup: {
    start: () => void;
    pending: boolean;
    /** the warm-up job this player queued, followed until it finishes; null when none (or
        dismissed). "starting" = queued, its first poll not landed yet. */
    job: { state: JobState | "starting"; failure: string | null } | null;
    dismiss: () => void;
    error: string | null;
  };
}

/** The audition state machine shared by the Voice Library card and the casting ▶ keys.

    Refusals are shown once, in plain words, and only a click acts on them:
    - engine_cold → Download now (a WARMUP job, followed until it finishes); gpu_busy /
      cloud_busy / audition_in_flight → "try after it finishes" with a manual try-again that is
      always unconfirmed (a paid voice re-quotes its price), never a retry loop;
    - payment_confirmation_required → an explicit "confirm ~$X & play" (never auto-confirmed);
    - gpu_busy_retry is the one SOFT refusal (a render lends the GPU between its segments), so
      it auto-retries with a short backoff, at most BORROW_RETRY_MAX times, preserving the
      confirm_paid the user chose.

    `text` is the line to speak (omitted: the server's default sentence). `autoplay` plays each
    new take through one shared element (the compact ▶ keys); without it the caller renders
    its own <audio> from `takeUrl`. */
export function useAuditionFlow({
  voiceId,
  engine,
  text,
  autoplay = false,
}: {
  voiceId: string;
  engine: string;
  text?: string;
  autoplay?: boolean;
}): AuditionFlow {
  const audition = useAudition(voiceId, text);
  const warmup = useWarmup();
  const qc = useQueryClient();
  const [warmupJobId, setWarmupJobId] = useState<string | null>(null);
  const warmupJob = useJob(warmupJobId);

  // One take per successful audition, latched in the mutate call's own onSuccess and paired with
  // the voice it was made for: the autoplay effect keys on this alone, so it can never run with a
  // stale take (a second ▶ replaying the first take) or pair a new voice with an old take.
  // Date.now() inline in the src would mint a new URL every parent re-render (the 2s job poll).
  const [take, setTake] = useState<{ voiceId: string; t: number } | null>(null);

  // a different voice (a picker change) or a different line starts from a clean slate: the old
  // refusal, take or download belongs to the old pair
  const resetAudition = audition.reset;
  useEffect(() => {
    resetAudition();
    setTake(null);
    setWarmupJobId(null);
  }, [voiceId, text, resetAudition]);

  const err: Error | null = audition.error ?? null;
  const apiErr = err instanceof ApiError ? err : null;

  const borrowRetries = useRef(0);
  const attempt = (confirmPaid: boolean) =>
    audition.mutate(confirmPaid, {
      onSuccess: () => {
        borrowRetries.current = 0;
        setTake({ voiceId, t: Date.now() });
      },
    });
  useEffect(() => {
    if (apiErr?.code !== "gpu_busy_retry" || borrowRetries.current >= BORROW_RETRY_MAX) return;
    borrowRetries.current += 1;
    // the soft borrow refusal is the one auto-retry; it repeats the attempt it follows with the
    // same confirm_paid (a borrow only ever serves a local GPU engine, so in practice false)
    const confirmPaid = audition.variables ?? false;
    const timer = setTimeout(() => attempt(confirmPaid), 500 * borrowRetries.current);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiErr]);

  const start = (confirmPaid: boolean) => {
    borrowRetries.current = 0; // a fresh user-initiated attempt resets the borrow budget
    setWarmupJobId(null);
    attempt(confirmPaid);
  };

  // autoplay: each new take plays through one element; unmount / a newer take stops it
  const [playing, setPlaying] = useState(false);
  const elRef = useRef<HTMLAudioElement | null>(null);
  useEffect(() => {
    if (!autoplay || !take) return;
    const el = new Audio(`/api/voices/${take.voiceId}/audition.wav?t=${take.t}`);
    nowPlaying?.pause();
    nowPlaying = el;
    elRef.current = el;
    // only the CURRENT element may clear `playing`: a superseded one's late pause event or
    // rejected play() promise (AbortError) must not flip the ■ of the take that replaced it
    const ended = () => {
      if (elRef.current === el) setPlaying(false);
    };
    el.onended = ended;
    el.onpause = ended;
    el.onerror = ended;
    setPlaying(true);
    el.play().catch(ended);
    return () => {
      el.onended = null;
      el.onpause = null;
      el.onerror = null;
      if (elRef.current === el) elRef.current = null;
      el.pause();
      if (nowPlaying === el) nowPlaying = null;
      setPlaying(false);
    };
  }, [autoplay, take]);

  const waiting = apiErr?.code === "gpu_busy_retry" && borrowRetries.current < BORROW_RETRY_MAX;
  const refusal: AuditionRefusal | null =
    err && !waiting
      ? apiErr
        ? explainAuditionRefusal(apiErr, engine)
        : { kind: "other", title: "couldn't play", text: "The sample couldn't be made.", code: "error", detail: err.message }
      : null;
  const phase: AuditionPhase = audition.isPending
    ? "making"
    : waiting
      ? "waiting"
      : refusal
        ? "refused"
        : audition.isSuccess
          ? "ready"
          : "idle";

  const job = warmupJob.data?.job_id === warmupJobId ? warmupJob.data : undefined;
  return {
    engine,
    phase,
    refusal,
    start,
    retryBorrow: () => start(audition.variables ?? false),
    dismiss: () => audition.reset(),
    takeUrl: take ? `/api/voices/${take.voiceId}/audition.wav?t=${take.t}` : null,
    playing,
    stop: () => {
      elRef.current?.pause();
      setPlaying(false);
    },
    warmup: {
      start: () =>
        warmup.mutate(engine, {
          onSuccess: (queued) => {
            // seed the followed job with the enqueue reply, so its state shows at once
            qc.setQueryData(["jobs", "one", queued.job_id], queued);
            setWarmupJobId(queued.job_id);
            audition.reset();
          },
        }),
      pending: warmup.isPending,
      job:
        warmupJobId === null
          ? null
          : {
              state: job?.state ?? "starting",
              failure: job?.state === "failed" ? failureText(job.error) : null,
            },
      dismiss: () => setWarmupJobId(null),
      error: warmup.error ? warmupErrorText(warmup.error) : null,
    },
  };
}

function failureText(error: string | null): string {
  const e = explainJobError(error);
  return e.fix ? `${e.summary} ${e.fix}` : e.summary;
}

/** Does AuditionNotice have something to say (so a table can give it a row)? */
export const hasAuditionNotice = (flow: AuditionFlow) =>
  flow.phase === "waiting" || flow.phase === "refused" || flow.warmup.job !== null;

/** What AuditionNotice says right now, as plain words (no buttons): also what a row's
    always-mounted live region announces. null when there is nothing to say. */
export function auditionNoticeText(flow: AuditionFlow): string | null {
  if (flow.phase === "waiting") return "Waiting for the recording to pause between lines — trying again…";
  if (flow.refusal) return flow.refusal.text;
  const job = flow.warmup.job;
  if (!job) return null;
  const name = engineName(flow.engine);
  switch (job.state) {
    case "queued":
      return `The ${name} download is queued — it starts on its own when the job ahead of it finishes.`;
    case "succeeded":
      return `${name} is downloaded and ready — press ▶ to hear the voice.`;
    case "failed":
      return `Couldn't get ${name} ready. ${job.failure ?? ""}`.trim();
    case "canceled":
      return `The ${name} download was stopped.`;
    default: // starting / running
      return `Downloading ${name} in the background — press ▶ again once it's ready.`;
  }
}
