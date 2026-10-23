import type { ReactNode } from "react";

import { formatSampleCost, RETRIABLE } from "../../lib/audition";
import { auditionNoticeText, type AuditionFlow } from "./useAuditionFlow";

/** What a voice sample is waiting on, or why it was refused, with the one recourse that fits.
    Shared by every player so the words (and the never-auto-confirm rule) can't drift. Renders
    nothing when there is nothing to say. `compact` trims the padding for table rows; `live`
    false drops the status role when the caller keeps its own always-mounted live region (a
    region inserted together with its text is often not announced). */
export function AuditionNotice({
  flow,
  compact = false,
  live = true,
}: {
  flow: AuditionFlow;
  compact?: boolean;
  live?: boolean;
}) {
  const text = auditionNoticeText(flow);
  if (text === null) return null;
  const cls = `refusal ${compact ? "compact" : ""}`;
  const role = live ? "status" : undefined;

  if (flow.phase === "waiting") {
    return (
      <div className={cls} role={role}>
        <p title="a recording is lending the GPU between its lines — it keeps running">{text}</p>
      </div>
    );
  }

  const r = flow.refusal;
  if (!r) {
    // the warm-up job this player queued, followed until it finishes
    const job = flow.warmup.job;
    const again = job !== null && (job.state === "failed" || job.state === "canceled");
    return (
      <div className={cls} role={role}>
        <p>
          {text}{" "}
          {again && (
            <>
              <button className="link" disabled={flow.warmup.pending} onClick={flow.warmup.start}>
                {flow.warmup.pending ? "starting the download…" : "download again"}
              </button>{" "}
              ·{" "}
            </>
          )}
          <button className="link" onClick={flow.warmup.dismiss}>
            {job?.state === "succeeded" || again ? "dismiss" : "hide"}
          </button>
        </p>
        {flow.warmup.error && <p className="text-clip">{flow.warmup.error}</p>}
      </div>
    );
  }

  let recourse: ReactNode;
  if (r.kind === "engine_cold") {
    recourse = (
      <button className="link" disabled={flow.warmup.pending} onClick={flow.warmup.start}>
        {flow.warmup.pending ? "starting the download…" : "Download now"}
      </button>
    );
  } else if (r.kind === "paid") {
    // the ONLY way a paid sample is made: this explicit click
    recourse = (
      <>
        <button className="link" onClick={() => flow.start(true)}>
          confirm ~{formatSampleCost(r.estimatedUsd ?? 0)} &amp; play
        </button>{" "}
        ·{" "}
        <button className="link" onClick={flow.dismiss}>
          cancel
        </button>
      </>
    );
  } else if (RETRIABLE.has(r.kind)) {
    // one manual try per click, never a loop — and UNconfirmed: a paid voice refused as busy
    // quotes its price again before anything is billed. The one exception is an exhausted
    // borrow, which repeats the very request the auto-retries were repeating.
    const again = r.kind === "borrow_exhausted" ? flow.retryBorrow : () => flow.start(false);
    recourse = (
      <>
        <button className="link" onClick={again}>
          try again
        </button>{" "}
        ·{" "}
        <button className="link" onClick={flow.dismiss}>
          dismiss
        </button>
      </>
    );
  } else {
    recourse = (
      <button className="link" onClick={flow.dismiss}>
        dismiss
      </button>
    );
  }

  return (
    <div className={cls} role={role}>
      <span className="tag" title={r.code}>
        {r.title}
      </span>
      <p>
        <span title={r.detail}>{r.text}</span> {recourse}
      </p>
      {r.kind === "engine_cold" && flow.warmup.error && <p className="text-clip">{flow.warmup.error}</p>}
    </div>
  );
}
