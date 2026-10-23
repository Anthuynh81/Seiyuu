import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useBooks, useDepartedJobs, useRecentJobs } from "../api/hooks";
import type { JobKind, JobOut } from "../api/types";
import { JobErrorText } from "../components/JobErrorText";
import { SCREENS } from "../lib/glossary";
import { engineSubject, finishedHeadline, subjectName } from "../lib/jobCopy";
import { explainJobError } from "../lib/jobErrors";

/** Successes (and user-requested stops) clear themselves; failures wait for the user. */
export const NOTICE_AUTO_DISMISS_MS = 6000;
/** Cap on the stack. Only successes/cancels are trimmed to meet it (oldest first): a failure
    waits for the user, however many notices arrive after it. */
export const MAX_NOTICES = 5;

const LED: Record<string, string> = { succeeded: "ok", failed: "err", canceled: "off" };

/** Where a finished book job's next step lives. */
const NEXT: Partial<Record<JobKind, [screen: string, label: string]>> = {
  attribute: ["review", "Review characters"],
  render: ["listen", "Listen"],
  assemble: ["render", `Open ${SCREENS.render}`],
  master: ["render", `Open ${SCREENS.render}`],
};

function Notice({ job, name, onDismiss: dismissId }: { job: JobOut; name: string; onDismiss: (id: string) => void }) {
  const nav = useNavigate();
  const failed = job.state === "failed";
  // tracked apart: the pointer leaving must not restart the timer while focus is still inside
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const paused = hovered || focused;
  const id = job.job_id;
  const onDismiss = useCallback(() => dismissId(id), [dismissId, id]);
  // auto-dismiss successes; the timer holds while the pointer or focus is on the notice so
  // nobody loses a notice they are reading or about to click
  useEffect(() => {
    if (failed || paused) return;
    const t = window.setTimeout(onDismiss, NOTICE_AUTO_DISMISS_MS);
    return () => window.clearTimeout(t);
  }, [failed, paused, onDismiss]);

  const headline = finishedHeadline(job, name);
  const isBook = engineSubject(job.book_id) === null;
  const next = !isBook ? undefined : failed ? (["render", `Open ${SCREENS.render}`] as const) : NEXT[job.kind];
  const cls = failed ? "fail" : job.state === "canceled" ? "stop" : "";
  return (
    <div
      className={`toast ${cls}`}
      data-notice={id}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onFocus={() => setFocused(true)}
      onBlur={(e) => {
        // moving between the notice's own buttons is not leaving it
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFocused(false);
      }}
    >
      <i className={`led ${LED[job.state] ?? "off"}`} aria-hidden />
      <div className="toast-body">
        <p className="toast-head">{headline}</p>
        {failed && <JobErrorText explained={explainJobError(job.error)} />}
        {next && job.state !== "canceled" && (
          <div className="toast-acts">
            <button
              className="link"
              onClick={() => {
                nav(`/${next[0]}?book=${encodeURIComponent(job.book_id)}`);
                if (!failed) onDismiss();
              }}
            >
              {next[1]}
            </button>
          </div>
        )}
      </div>
      <button className="toast-x" aria-label={`Dismiss: ${headline}`} onClick={onDismiss}>
        ✕
      </button>
    </div>
  );
}

/** Trim to MAX_NOTICES by dropping the oldest successes/cancels; failures are never trimmed. */
function capNotices(list: JobOut[]): JobOut[] {
  let extra = list.length - MAX_NOTICES;
  if (extra <= 0) return list;
  return list.filter((j) => {
    if (extra > 0 && j.state !== "failed") {
      extra -= 1;
      return false;
    }
    return true;
  });
}

/** Focus somewhere sensible when a focused notice goes away: the next notice's dismiss
    button, else the previous one's, else the main region. Never strand focus on <body>. */
function refocusAfter(id: string, root: HTMLElement | null): void {
  const gone = Array.from(root?.children ?? []).find((el) => (el as HTMLElement).dataset.notice === id);
  if (!gone || !gone.contains(document.activeElement)) return;
  const sibling = (gone.nextElementSibling ?? gone.previousElementSibling) as HTMLElement | null;
  const target =
    sibling?.querySelector<HTMLElement>(".toast-x") ?? document.querySelector<HTMLElement>("main") ?? null;
  if (target) {
    if (target.tagName === "MAIN" && !target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
    target.focus();
  }
}

/** One notice per job that finishes while the app is open. Two sources, both written by
    JobCompletionWatcher: the jobs it saw leave the live poll, fetched by id
    (["jobs","departed"]: finished in this session by construction), and the recent-jobs
    window (["jobs","recent"]: catches a job that started and finished inside one poll). The
    FIRST window only seeds the seen set, so jobs that were finished before the app loaded
    never toast, and a refetch never toasts twice. */
export function JobNotices() {
  const recent = useRecentJobs();
  const departed = useDepartedJobs();
  const books = useBooks();
  const seen = useRef<Set<string> | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const [notices, setNotices] = useState<JobOut[]>([]);

  const windowJobs = recent.data?.jobs;
  const departedJobs = departed.data;
  useEffect(() => {
    if (!windowJobs) return; // wait for the seed
    const watched = departedJobs ?? [];
    const watchedIds = new Set(watched.map((j) => j.job_id));
    const terminal = windowJobs.filter((j) => j.is_terminal);
    if (seen.current === null) {
      // a job seen leaving the live poll finished in this session, even if the window
      // happens to load after it
      seen.current = new Set(terminal.filter((j) => !watchedIds.has(j.job_id)).map((j) => j.job_id));
    }
    const known = seen.current;
    const byId = new Map<string, JobOut>();
    for (const j of [...terminal, ...watched.filter((w) => w.is_terminal)]) {
      if (!known.has(j.job_id)) byId.set(j.job_id, j);
    }
    const fresh = [...byId.values()].sort(
      (a, b) => Date.parse(a.finished_at ?? a.created_at) - Date.parse(b.finished_at ?? b.created_at),
    );
    if (fresh.length === 0) return;
    for (const j of fresh) known.add(j.job_id);
    setNotices((prev) => capNotices([...prev, ...fresh]));
  }, [windowJobs, departedJobs]);

  // stable, so a notice's auto-dismiss timer isn't reset by every parent render
  const dismiss = useCallback((id: string) => {
    refocusAfter(id, root.current);
    setNotices((prev) => prev.filter((n) => n.job_id !== id));
  }, []);

  return (
    <div ref={root} className="toasts" role="status" aria-live="polite" aria-label="Job notices">
      {notices.map((job) => (
        <Notice key={job.job_id} job={job} name={subjectName(job, books.data?.books)} onDismiss={dismiss} />
      ))}
    </div>
  );
}
