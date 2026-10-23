/** How long an AI-hinted cast apply takes on this machine, remembered per viewer so the apply
    progress bar can estimate (lib/storage: losing it only means the default estimate). */

import { readStored, writeStored } from "./storage";

/** How long the last AI-hinted apply took, so the bar's estimate fits this machine. */
const LAST_AI_SECONDS_KEY = "seiyuu.castAiSeconds";
const DEFAULT_AI_SECONDS = 60;
/** Anything shorter is not a believable AI apply to estimate from. */
const MIN_REMEMBERED_SECONDS = 5;

export function expectedAiSeconds(): number {
  const n = Number(readStored(LAST_AI_SECONDS_KEY));
  return Number.isFinite(n) && n >= MIN_REMEMBERED_SECONDS ? n : DEFAULT_AI_SECONDS;
}

/** Remember how long a SUCCESSFUL AI-hinted apply took (call it from the apply's onSuccess):
    a canceled or failed one says nothing about how long the next will take. */
export function rememberAiApply(seconds: number): void {
  if (seconds >= MIN_REMEMBERED_SECONDS) writeStored(LAST_AI_SECONDS_KEY, String(Math.round(seconds)));
}
