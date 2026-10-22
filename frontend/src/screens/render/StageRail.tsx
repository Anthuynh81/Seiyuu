import type { ActiveJobSummary, BookCard, JobOut } from "../../api/types";
import { KIND_STAGE, STAGES } from "../../api/types";
import { NextStep } from "../../components/NextStep";
import { nextStep } from "../../lib/nextStep";

/* -------------------------------------------------- pipeline rail */

/** The whole signal path in one glance — which stages are done, which one is running —
    and the ONE next step as a real button (lib/nextStep.ts), so "what do I click after
    recording?" never needs the order memorized. */
export function StageRail({
  status,
  activeJob,
  failed = null,
}: {
  status: Omit<BookCard, "active_job"> | undefined;
  activeJob: ActiveJobSummary | null;
  /** The book's latest unresolved failure: the step turns into a retry with the fix. */
  failed?: JobOut | null;
}) {
  const runningStage = activeJob ? KIND_STAGE[activeJob.kind] : undefined;
  return (
    <div className="scoperow" role="group" aria-label="pipeline">
      <span className="tag">steps</span>
      <span role="list" aria-label="steps" className="contents">
        {STAGES.map(([flag, label], i) => (
          <span key={flag} role="listitem" className="state mono text-[11px]">
            <i className={`led ${runningStage === flag ? "run" : status?.[flag] ? "ok" : "off"}`} />
            {label}
            {i < STAGES.length - 1 && <span aria-hidden className="text-ink-3">›</span>}
          </span>
        ))}
      </span>
      {status && <NextStep step={nextStep({ status, activeJob, failed })} />}
    </div>
  );
}
