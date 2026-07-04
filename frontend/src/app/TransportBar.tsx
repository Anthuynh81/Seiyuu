import { useCancelJob, useLiveJobs } from "../api/hooks";
import type { JobOut } from "../api/types";

function ledFor(job: JobOut): string {
  if (job.state === "running") return job.cancel_requested ? "cxl" : "run";
  return "q";
}

/** The persistent bottom transport: the single live job on every screen. (Becomes the
    audio transport on the Listen screen in M6c-5.) */
export function TransportBar() {
  const live = useLiveJobs();
  const cancel = useCancelJob();
  const jobs = live.data?.jobs ?? [];
  const job = jobs.find((j) => j.state === "running") ?? jobs[0];

  if (!job) {
    return (
      <div className="transport">
        <span className="state">
          <i className="led off" />
          idle
        </span>
        <span className="idle">console idle — no jobs queued or running</span>
      </div>
    );
  }
  const canceling = job.cancel_requested && job.state === "running";
  return (
    <div className="transport">
      <span className="state">
        <i className={`led ${ledFor(job)}`} />
        {canceling ? "canceling" : job.state}
      </span>
      <span style={{ color: "var(--ink-2)" }}>
        {job.kind} · {job.book_id}
      </span>
      <div className={`meter ${job.state === "running" ? "run" : ""}`} />
      <span style={{ color: "var(--ink-2)" }}>{job.progress_text || "waiting for the worker…"}</span>
      <button
        className="cancel"
        disabled={cancel.isPending || canceling}
        onClick={() => cancel.mutate(job.job_id)}
      >
        {canceling ? "canceling…" : "cancel"}
      </button>
    </div>
  );
}
