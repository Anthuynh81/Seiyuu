import { useState } from "react";

import type { JobOut } from "../../api/types";
import { useJobProgress } from "../../app/useJobProgress";
import { stepName } from "../../lib/jobCopy";
import { explainJobError } from "../../lib/jobErrors";
import { progressLabel } from "../../lib/progress";
import { MODE_LABEL } from "../../lib/render";

/* -------------------------------------------------- jobs + validation */

const LED: Record<string, string> = { queued: "q", running: "run", succeeded: "ok", failed: "err", canceled: "off" };

export function JobRow({ job }: { job: JobOut }) {
  const [open, setOpen] = useState(false);
  const canceling = job.state === "running" && job.cancel_requested;
  const detail = job.error;
  const progress = useJobProgress(job.state === "running" ? job : null);
  return (
    <div className={`job ${open ? "open" : ""}`} onClick={() => detail && setOpen(!open)}>
      <div className="jobtop">
        <span className="state">
          <i className={`led ${canceling ? "cxl" : LED[job.state]}`} />
          {canceling ? "canceling" : job.state}
        </span>
        <span className="what">
          {stepName(job.kind)}
          {typeof job.params?.mode === "string" && (
            <span className="mono ml-1.5 text-[10.5px] text-ink-3">
              {MODE_LABEL[job.params.mode as keyof typeof MODE_LABEL] ?? job.params.mode}
            </span>
          )}
          {detail && <span className="more">why? ▸</span>}
        </span>
        <span className="id">{job.job_id}</span>
        <span className="when">{new Date(job.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
      </div>
      {job.state === "running" && progress && (
        <div className="jobmeter" role="progressbar" aria-label={`${stepName(job.kind)} progress`}
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress.fraction * 100)}>
          <i style={{ width: `${progress.fraction * 100}%` }} />
        </div>
      )}
      {job.state === "running" && (progress || job.progress_text) && (
        <div className="mono mt-1.5 text-[11px] text-ink-2">
          {progress ? `${progressLabel(progress)}${progress.title ? ` · ${progress.title}` : ""}` : job.progress_text}
        </div>
      )}
      {detail && (
        // clicks inside (selecting the fix command to copy it) must not collapse the row
        <div className="jdetail" onClick={(e) => e.stopPropagation()}>
          <JobFix error={detail} />
          <div className="errblock">{detail}</div>
        </div>
      )}
    </div>
  );
}

/** The plain-words reading of a known failure, above the raw message (which stays visible
    here: this is the history view). Unknown failures add nothing. */
function JobFix({ error }: { error: string }) {
  const explained = explainJobError(error);
  if (!explained.known) return null;
  return (
    <div className="mt-2">
      <p className="joberr-sum">{explained.summary}</p>
      {explained.fix && <p className="joberr-fix">{explained.fix}</p>}
      {explained.command && <code className="chk-cmd">{explained.command}</code>}
    </div>
  );
}
