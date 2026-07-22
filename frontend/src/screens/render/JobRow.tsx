import { useState } from "react";

import type { JobOut } from "../../api/types";

/* -------------------------------------------------- jobs + validation */

const LED: Record<string, string> = { queued: "q", running: "run", succeeded: "ok", failed: "err", canceled: "off" };

export function JobRow({ job }: { job: JobOut }) {
  const [open, setOpen] = useState(false);
  const canceling = job.state === "running" && job.cancel_requested;
  const detail = job.error;
  return (
    <div className={`job ${open ? "open" : ""}`} onClick={() => detail && setOpen(!open)}>
      <div className="jobtop">
        <span className="state">
          <i className={`led ${canceling ? "cxl" : LED[job.state]}`} />
          {canceling ? "canceling" : job.state}
        </span>
        <span className="what">
          {job.kind}
          {typeof job.params?.mode === "string" && (
            <span className="mono ml-1.5 text-[10.5px] text-ink-3">{job.params.mode}</span>
          )}
          {detail && <span className="more">why? ▸</span>}
        </span>
        <span className="id">{job.job_id}</span>
        <span className="when">{new Date(job.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
      </div>
      {job.state === "running" && job.progress_text && (
        <div className="mono mt-1.5 text-[11px] text-ink-2">{job.progress_text}</div>
      )}
      {detail && <div className="jdetail"><div className="errblock">{detail}</div></div>}
    </div>
  );
}
