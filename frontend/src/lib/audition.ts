import { ApiError, BACKEND_DOWN } from "../api/client";
import type { JobKind } from "../api/types";
import { JOB_DOING } from "./glossary";
import { engineName, manualInstallFor } from "./health";

/** Plain words for a refused voice sample (POST /voices/{id}/audition). Pure, so every player
    (Voice Library card, Review roster ▶, Suggest-a-cast ▶) says the same thing. The raw code
    stays available for a tooltip; it is never the main text. */

/** The audition endpoint takes at most 500 characters of custom text (AuditionRequest). */
export const AUDITION_MAX_CHARS = 500;

/** The line a character's ▶ speaks: their first sample line, trimmed to the endpoint's limit.
    Cut on code points (never half a surrogate pair) and at a word boundary when possible.
    Returns undefined when there is nothing usable, so the server's default sentence plays. */
export function auditionLine(raw: string | null | undefined): string | undefined {
  const text = (raw ?? "").replace(/\s+/g, " ").trim();
  if (!text) return undefined;
  const chars = Array.from(text);
  if (chars.length <= AUDITION_MAX_CHARS) return text;
  const head = chars.slice(0, AUDITION_MAX_CHARS - 1).join("");
  const space = head.lastIndexOf(" ");
  return `${space > AUDITION_MAX_CHARS / 2 ? head.slice(0, space) : head}…`;
}

export type RefusalKind =
  | "engine_cold" // engine weights never downloaded → Download now (a warm-up job)
  | "not_installed" // engine the warm-up job can't fetch (IndexTTS-2) → the manual-install steps
  | "gpu_busy" // a job holds the GPU → try after it finishes
  | "cloud_busy" // a render holds the cloud voice slots → try after it finishes
  | "in_flight" // another sample is being made right now → wait, then try again
  | "paid" // paid engine → explicit "confirm ~$X & play"
  | "borrow_exhausted" // a render didn't lend the GPU within the auto-retry budget
  | "upstream" // the engine / cloud service failed while making it → try again
  | "offline" // the backend isn't answering → start it, try again
  | "other"; // a fault a retry won't fix (no key, voice gone, consent, unknown) → dismiss

/** Refusals worth a manual "try again". Such a retry is a FRESH, unconfirmed attempt, so a paid
    voice quotes its price again before anything is billed (an exhausted borrow excepted: it
    repeats the request its auto-retries were repeating). */
export const RETRIABLE: ReadonlySet<RefusalKind> = new Set<RefusalKind>([
  "gpu_busy",
  "cloud_busy",
  "in_flight",
  "borrow_exhausted",
  "upstream",
  "offline",
]);

export interface AuditionRefusal {
  kind: RefusalKind;
  /** a short label for the refusal box */
  title: string;
  /** the sentence the user reads */
  text: string;
  /** the server's code, for a tooltip only */
  code: string;
  /** the server's own message, for a tooltip only (never the main text) */
  detail?: string;
  /** paid only: the estimate the confirm button names */
  estimatedUsd?: number;
}

/** "recording", "finding the speakers": the glossary's step name, mid-sentence. */
const busyDoing = (kind: JobKind | null): string => {
  const doing = kind ? JOB_DOING[kind] : undefined;
  return doing ? doing[0].toLowerCase() + doing.slice(1) : "recording";
};

/** "$0.0123" — enough places that a fraction-of-a-cent sample doesn't read as free. */
export function formatSampleCost(usd: number): string {
  return `$${(Number.isFinite(usd) ? usd : 0).toFixed(4)}`;
}

export function explainAuditionRefusal(err: ApiError, engine: string): AuditionRefusal {
  const detail = (err.detail ?? {}) as Record<string, unknown>;
  const base = { code: err.code, detail: err.message };
  switch (err.code) {
    case "engine_cold": {
      const manual = manualInstallFor(engine);
      if (manual) return { ...base, kind: "not_installed", title: "not installed", text: manual };
      return {
        ...base,
        kind: "engine_cold",
        title: "not downloaded yet",
        // no size: it differs per engine (Kokoro is a few hundred MB, Chatterbox several GB)
        text: `${engineName(engine)} hasn't been downloaded yet — it's a one-time download.`,
      };
    }
    case "gpu_busy": {
      const kind = typeof detail.kind === "string" ? (detail.kind as JobKind) : null;
      const doing = busyDoing(kind);
      return {
        ...base,
        kind: "gpu_busy",
        title: "busy",
        text: `Busy ${doing} — try after it finishes.`,
      };
    }
    case "cloud_busy":
      return {
        ...base,
        kind: "cloud_busy",
        title: "busy",
        text: "A recording is using the cloud voices — try after it finishes.",
      };
    case "audition_in_flight":
      return {
        ...base,
        kind: "in_flight",
        title: "one at a time",
        text: "Another voice sample is being made — wait a moment, then try again.",
      };
    case "payment_confirmation_required": {
      const usd = Number(detail.estimated_usd ?? 0);
      return {
        ...base,
        kind: "paid",
        title: "paid voice",
        text: `This voice is billed per use — this sample costs about ${formatSampleCost(usd)}.`,
        estimatedUsd: usd,
      };
    }
    case "gpu_busy_retry":
      return {
        ...base,
        kind: "borrow_exhausted",
        title: "busy",
        text: "A recording is running and hasn't paused for a sample yet — it keeps going.",
      };
    case "upstream":
      return {
        ...base,
        kind: "upstream",
        title: "couldn't make it",
        text: `${engineName(engine)} couldn't make the sample this time.`,
      };
    case BACKEND_DOWN:
      return {
        ...base,
        kind: "offline",
        title: "backend off",
        text: "The Seiyuu backend isn't running — start it, then try again.",
      };
    case "not_ready":
      return {
        ...base,
        kind: "other",
        title: "needs setup",
        text:
          engine === "elevenlabs"
            ? "Cloud voices need an ElevenLabs key — add it to the backend's settings, then restart it."
            : "The backend is still starting — try again in a moment.",
      };
    case "not_found":
      return {
        ...base,
        kind: "other",
        title: "voice gone",
        text: "This voice no longer exists — pick another one.",
      };
    case "consent_invalid":
      return {
        ...base,
        kind: "other",
        title: "can't use",
        text: "This cloned voice's recording doesn't match the consent given for it — re-create the voice with fresh consent.",
      };
    default:
      return {
        ...base,
        kind: "other",
        title: "couldn't play",
        text: "The sample couldn't be made.",
      };
  }
}
