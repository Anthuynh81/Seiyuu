import { useId } from "react";

import { useEngines, useLiveJobs, useSystemProbe, useWarmup } from "../api/hooks";
import { buildChecklist, type CheckRow, type CheckState, warmupErrorText } from "../lib/health";

const LED: Record<CheckState, string> = {
  ok: "ok",
  missing: "err",
  busy: "run",
  checking: "q",
  unknown: "off",
};

function Row({
  row,
  onDownload,
  queuing,
  refreshing,
  error,
}: {
  row: CheckRow;
  onDownload: (engineId: string) => void;
  queuing: boolean;
  /** The engine catalog is refetching (e.g. a warm-up just finished): its weights_cached may
      be stale for one round-trip, so don't re-offer the download until it lands. */
  refreshing: boolean;
  error: string | null;
}) {
  const labelId = useId();
  // an optional row that's missing is advice, not a fault: caution, not red
  const led = row.state === "missing" && !row.required ? "warn" : LED[row.state];
  return (
    <li className="chk" aria-labelledby={labelId} data-state={row.state}>
      <i className={`led ${led}`} aria-hidden />
      <div className="chk-body">
        <div className="chk-top">
          <span id={labelId} className="chk-label">
            {row.label}
          </span>
          {!row.required && row.state === "missing" && <span className="chk-opt">optional</span>}
          <span className="chk-status">{row.status}</span>
        </div>
        {row.fix && <p className="chk-fix">{row.fix}</p>}
        {row.command && <code className="chk-cmd">{row.command}</code>}
        {error && <p className="chk-err">{error}</p>}
      </div>
      {row.canDownload && row.engineId && (
        <button className="key quiet chk-act" disabled={queuing || refreshing} onClick={() => onDownload(row.engineId!)}>
          {queuing ? "queuing…" : refreshing ? "checking…" : "Download now"}
        </button>
      )}
    </li>
  );
}

/** Setup checklist — shared by the System drawer and Library's empty state. Every row is
    derived from server data (probe, engine catalog, live jobs); "Download now" queues the
    existing warm-up job, whose queued/running state then shows from the live-jobs poll that
    is already running (no extra polling here). */
export function SetupChecklist() {
  const probe = useSystemProbe();
  const engines = useEngines();
  const live = useLiveJobs();
  const warmup = useWarmup();

  const rows = buildChecklist({
    probe: probe.data,
    probeError: probe.error,
    engines: engines.data?.engines ?? (engines.isError ? null : undefined),
    liveJobs: live.data?.jobs,
  });

  return (
    <ul className="chklist" aria-label="Setup checklist">
      {rows.map((row) => {
        const mine = row.engineId !== undefined && warmup.variables === row.engineId;
        return (
          <Row
            key={row.id}
            row={row}
            queuing={mine && warmup.isPending}
            refreshing={engines.isFetching}
            // once the live-jobs poll shows the warm-up, the row's own Queued/Downloading
            // status is the answer — a stale refusal (e.g. duplicate_job) would contradict it
            error={mine && warmup.isError && row.state !== "busy" ? warmupErrorText(warmup.error) : null}
            onDownload={(id) => warmup.mutate(id)}
          />
        );
      })}
    </ul>
  );
}
