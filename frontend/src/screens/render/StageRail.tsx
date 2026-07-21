import type { ActiveJobSummary, BookCard } from "../../api/types";
import { KIND_STAGE, STAGES } from "../../api/types";

/* -------------------------------------------------- pipeline rail */

/** The whole signal path in one glance — which stages are done, which one is running,
    and what to do next. Replaces the old three-paragraph "steps" banner and answers
    "what do I click after render?" without the user memorizing the order. */
export function StageRail({
  status,
  activeJob,
}: {
  status: Omit<BookCard, "active_job"> | undefined;
  activeJob: ActiveJobSummary | null;
}) {
  const runningStage = activeJob ? KIND_STAGE[activeJob.kind] : undefined;
  const nextFlag = STAGES.find(([flag]) => !status?.[flag])?.[0];
  const hint = runningStage
    ? `${activeJob!.kind} is ${activeJob!.state} — follow it on the right`
    : nextFlag === "ingested"
      ? "no text yet — re-ingest from the Library"
      : nextFlag === "attributed"
        ? "next: attribute (below) — or render single voice right away"
        : nextFlag === "assigned"
          ? "next: cast voices in Character Review — or render single voice"
          : nextFlag === "rendered"
            ? "next: render (below)"
            : nextFlag // assembled or mastered
              ? "next: finish · assemble + master (Outputs panel)"
              : "all stages done — download the .m4b in Outputs";
  return (
    <div className="scoperow" role="list" aria-label="pipeline">
      <span className="tag">pipeline</span>
      {STAGES.map(([flag, label], i) => (
        <span key={flag} role="listitem" className="state mono text-[11px]">
          <i className={`led ${runningStage === flag ? "run" : status?.[flag] ? "ok" : "off"}`} />
          {label}
          {i < STAGES.length - 1 && <span aria-hidden className="text-ink-3">›</span>}
        </span>
      ))}
      <span className="mono scopehint">{hint}</span>
    </div>
  );
}
