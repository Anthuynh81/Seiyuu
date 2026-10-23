import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { ApiError, isBackendDown } from "../api/client";
import { useBooks, useDeleteBook, useIngest, useShelfJobs } from "../api/hooks";
import type { BookCard as BookCardT, IngestResult, JobOut, PaidArtifacts } from "../api/types";
import { KIND_STAGE, STAGES } from "../api/types";
import { TalkDialog } from "../components/Dialog";
import { JobErrorText } from "../components/JobErrorText";
import { NextStep } from "../components/NextStep";
import { PlainError } from "../components/PlainError";
import { SetupChecklist } from "../components/SetupChecklist";
import { failureLine, JOB_DOING } from "../lib/glossary";
import { engineName } from "../lib/health";
import { failedStepLine, liveJobLine } from "../lib/jobCopy";
import { explainJobError, unresolvedFailure } from "../lib/jobErrors";
import { ADD_ANCHOR, nextStep } from "../lib/nextStep";
import { AttributePanel } from "./render/AttributePanel";

function SignalPath({ book }: { book: BookCardT }) {
  const activeStage = book.active_job ? KIND_STAGE[book.active_job.kind] : undefined;
  return (
    <div className="sigpath">
      {STAGES.map(([flag, label]) => {
        const done = book[flag];
        const now = !done && flag === activeStage;
        return (
          <div key={flag} className={`stop ${done ? "done" : ""} ${now ? "now" : ""}`}>
            <div className="bar" />
            <span className="lbl">{label}</span>
          </div>
        );
      })}
    </div>
  );
}

function JobLine({ book }: { book: BookCardT }) {
  const job = book.active_job;
  if (!job) return <div className="jobline text-ink-3">Nothing running</div>;
  return (
    <div className="jobline">
      <i className={`led ${job.state === "running" ? "run" : "q"}`} />
      {liveJobLine(job)}
    </div>
  );
}

/** Per-viewer convenience only: which failure banners this browser has dismissed. The
    failure itself is server truth (the job row); storage that throws or is empty just means
    the banner shows again. */
const DISMISS_PREFIX = "seiyuu.dismissedFailure.";

function readDismissed(jobId: string): boolean {
  try {
    return localStorage.getItem(DISMISS_PREFIX + jobId) === "1";
  } catch {
    return false;
  }
}

function writeDismissed(jobId: string): void {
  try {
    localStorage.setItem(DISMISS_PREFIX + jobId, "1");
  } catch {
    // private window / blocked storage: the dismissal lasts for this page only
  }
}

/** The failure the card should show: the book's latest unresolved failed run, unless a retry of
    the same kind is live right now or this browser dismissed it. The card owns this (not the
    banner) so the next step can tell whether the banner is still there to carry the fix. */
function useVisibleFailure(book: BookCardT, failed: JobOut | null) {
  const [dismissedNow, setDismissedNow] = useState<Set<string>>(() => new Set());
  const visible =
    failed !== null &&
    book.active_job?.kind !== failed.kind &&
    !dismissedNow.has(failed.job_id) &&
    !readDismissed(failed.job_id)
      ? failed
      : null;
  const dismiss = (jobId: string) => {
    writeDismissed(jobId);
    setDismissedNow((prev) => new Set(prev).add(jobId));
  };
  return [visible, dismiss] as const;
}

/** The book's latest failed run, in plain words, until it is dismissed or a newer job of the
    same kind replaces it (the user retried). */
function FailBanner({ failed, onDismiss }: { failed: JobOut; onDismiss: (jobId: string) => void }) {
  const headline = failedStepLine(failed.kind);
  return (
    <div className="failbanner" role="group" aria-label={headline}>
      <div className="failbanner-top">
        <i className="led err" aria-hidden />
        <b>{headline}</b>
        <button
          className="key quiet"
          onClick={(e) => {
            // the button is about to vanish: hand focus to the card's first action so a
            // keyboard user keeps their place instead of landing on <body>
            const next = e.currentTarget
              .closest(".bookcard")
              ?.querySelector<HTMLElement>(".nextrow a[href], .nextrow button:not([disabled]), .acts button:not([disabled])");
            onDismiss(failed.job_id);
            next?.focus();
          }}
        >
          Dismiss
        </button>
      </div>
      <JobErrorText explained={explainJobError(failed.error)} />
    </div>
  );
}

/** Two-step book delete. Step 1 is a plain confirm; a 402 escalates to a second, blunt
    confirm that re-sends confirm_paid=true and spells out the paid segments it discards.
    409 (a live job) and 500 (partial delete) render their own dead-ends. */
function DeleteBookDialog({ book, onClose }: { book: BookCardT; onClose: () => void }) {
  const del = useDeleteBook();
  const err = del.error instanceof ApiError ? del.error : null;
  const run = (confirmPaid: boolean) => del.mutate({ bookId: book.book_id, confirmPaid }, { onSuccess: onClose });

  const paid = err?.code === "payment_confirmation_required" ? (err.detail as PaidArtifacts) : null;
  const conflict = err?.code === "conflicting_job" ? (err.detail as JobOut) : null;
  const partial = err?.code === "partial_delete" ? (err.detail as { survivors: string[] }) : null;
  const otherErr = err && !paid && !conflict && !partial ? err : null;
  const title = book.title ?? book.book_id;

  return (
    <TalkDialog
      title={paid ? "Discard paid renders?" : "Delete book"}
      onClose={onClose}
      footer={
        <>
          <button className="key quiet" onClick={onClose}>
            {conflict ? "close" : "cancel"}
          </button>
          {conflict ? null : partial ? (
            <button className="key danger" disabled={del.isPending} onClick={() => run(true)}>
              {del.isPending ? "retrying…" : "retry delete"}
            </button>
          ) : paid ? (
            <button className="key danger" disabled={del.isPending} onClick={() => run(true)}>
              {del.isPending
                ? "deleting…"
                : `discard ${paid.paid_segment_count} paid line${paid.paid_segment_count === 1 ? "" : "s"} & delete`}
            </button>
          ) : (
            <button className="key danger" disabled={del.isPending} onClick={() => run(false)}>
              {del.isPending ? "deleting…" : "delete book"}
            </button>
          )}
        </>
      }
    >
      {conflict ? (
        <div className="refusal">
          <span className="tag">busy</span>
          <p>
            {JOB_DOING[conflict.kind] ?? "A job"} is {conflict.state === "queued" ? "waiting its turn" : "running"} for
            this book — cancel it in the bar at the bottom, then delete.
          </p>
        </div>
      ) : partial ? (
        <div className="refusal">
          <span className="tag">partly deleted</span>
          <p>
            the delete only partly completed — these files survived and may need a retry or removing by hand:
            <br />
            {partial.survivors.map((s) => (
              <span key={s} className="mono block text-caution">{s}</span>
            ))}
          </p>
        </div>
      ) : paid ? (
        <>
          <p className="mb-2.5 mt-0 text-ink-2">
            <b className="text-clip">{paid.paid_segment_count}</b> paid cloud line
            {paid.paid_segment_count === 1 ? " was" : "s were"} recorded for <b>{title}</b> — deleting the book
            discards them, and recording them again costs real money
            {paid.estimated_usd !== null && ` (~$${paid.estimated_usd.toFixed(2)})`}.
          </p>
          <div className="mono text-[11px] text-ink-3">
            voice engines: {paid.engines.map(engineName).join(", ") || "—"}
            {paid.paid_voice_ids.length > 0 && ` · voices: ${paid.paid_voice_ids.join(", ")}`}
          </div>
        </>
      ) : (
        <p className="m-0 text-ink-2">
          Permanently delete <b>{title}</b> — the imported text, the speakers found, the casting, and every
          recording and audiobook file made from it. This cannot be undone.
        </p>
      )}
      {otherErr && <PlainError className="mt-3" error={otherErr} title="couldn't delete" />}
    </TalkDialog>
  );
}

function Card({ book, failed }: { book: BookCardT; failed: JobOut | null }) {
  const nav = useNavigate();
  const go = (screen: string) => nav(`/${screen}?book=${encodeURIComponent(book.book_id)}`);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const blocked = book.active_job !== null;
  const [visibleFailure, dismissFailure] = useVisibleFailure(book, failed);
  const step = nextStep({ status: book, activeJob: book.active_job, failed });
  // the banner already says what to fix for the step it failed: don't say it twice. Once the
  // banner is dismissed (or the step is something else), the step carries its own reason.
  const fixInBanner = visibleFailure !== null && step.retryOf === visibleFailure.kind;
  // Finding the speakers is done right here: the step's key opens the panel inside the card
  // (no trip to Render), and the card stays one line tall until it's asked for.
  const findHere = step.stepId === "attribute" && !step.disabled;
  const [attrOpen, setAttrOpen] = useState(false);
  // once started, the step turns into the disabled running-job key: keep the open panel (now
  // its "finding the speakers" strip, which takes the focus) instead of dropping it mid-use
  const showPanel = attrOpen && (findHere || book.active_job?.kind === "attribute");
  const panelId = useId();
  return (
    <div className="bookcard">
      <h3>{book.title ?? book.book_id}</h3>
      <div className="by">{book.authors.join(", ") || "—"}</div>
      <SignalPath book={book} />
      <JobLine book={book} />
      {visibleFailure && <FailBanner failed={visibleFailure} onDismiss={dismissFailure} />}
      <div className="nextrow">
        <NextStep
          step={step}
          // the open panel explains the step itself: don't say it twice
          showReason={!fixInBanner && !showPanel}
          expand={findHere ? { open: attrOpen, onToggle: () => setAttrOpen(!attrOpen), controls: panelId } : undefined}
        />
      </div>
      {showPanel && (
        <AttributePanel id={panelId} book={{ book_id: book.book_id, attributed: book.attributed, active_job: book.active_job }} />
      )}
      <div className="acts">
        {book.rendered && !step.to.startsWith("/listen") && (
          <button className="key quiet" onClick={() => go("listen")}>
            ▶ listen
          </button>
        )}
        {book.attributed && (
          <button className="key quiet" onClick={() => go("review")}>
            review characters
          </button>
        )}
        <button className="key quiet" onClick={() => go("render")}>
          render &amp; jobs
        </button>
        <span className="flex-1" />
        <button
          className={`key ${blocked ? "quiet" : "danger"}`}
          disabled={blocked}
          title={blocked ? "a job is live — cancel it in the transport bar before deleting" : "delete this book"}
          onClick={() => setConfirmDelete(true)}
        >
          delete
        </button>
      </div>
      {confirmDelete && <DeleteBookDialog book={book} onClose={() => setConfirmDelete(false)} />}
    </div>
  );
}

/** The "What was left out" disclosure: what the importer skipped, each group in plain words. */
function LeftOut({ result }: { result: IngestResult }) {
  const skipped = result.skipped_items;
  const dropped = result.dropped_sections;
  const total = skipped.length + dropped.length;
  if (total === 0) return <p className="nextstep-why m-0">Nothing was left out.</p>;
  return (
    <details className="fine">
      <summary>What was left out ({total})</summary>
      {skipped.length > 0 && (
        <>
          <h4>Pages skipped as front or back matter</h4>
          <ul>
            {skipped.map((item, i) => (
              <li key={`${i}:${item}`}>{item}</li>
            ))}
          </ul>
        </>
      )}
      {dropped.length > 0 && (
        <>
          <h4>Sections dropped as too short or with nothing to read aloud</h4>
          <ul>
            {dropped.map((section, i) => (
              <li key={`${i}:${section}`}>{section}</li>
            ))}
          </ul>
        </>
      )}
    </details>
  );
}

function resultHeadline(result: IngestResult): string {
  const title = result.book.title ?? "your book";
  return result.already_in_library ? `Already in your library: ${title}` : `Added ${title}`;
}

/** The honest upload result: which book, whether it was new, what was left out, and the one
    thing to do next. Lives outside the drop button (a disclosure can't sit inside a button).
    The next step comes from the LIVE shelf (same inputs as the book's card), falling back to
    the upload's snapshot only until the refreshed shelf lists the book; once the shelf has
    listed it and then lost it (deleted from its card), the panel goes away. */
function IngestResultPanel({
  result,
  books,
  jobs,
}: {
  result: IngestResult;
  books: BookCardT[] | undefined;
  jobs: JobOut[] | undefined;
}) {
  const live = books?.find((b) => b.book_id === result.book.book_id);
  const [seen, setSeen] = useState(false);
  if (live && !seen) setSeen(true);
  if (!live && seen) return null;
  const step = live
    ? nextStep({ status: live, activeJob: live.active_job, failed: unresolvedFailure(jobs, live.book_id) })
    : nextStep({ status: result.book });
  const dup = result.already_in_library;
  return (
    <section className={`ingestres ${dup ? "dup" : ""}`} aria-label="Upload result">
      <p className="ingestres-head">{resultHeadline(result)}</p>
      <p className="ingestres-meta">
        {result.chapters} chapter{result.chapters === 1 ? "" : "s"} · {result.blocks} paragraph
        {result.blocks === 1 ? "" : "s"}
        {dup && " · this exact file was added before"}
      </p>
      <NextStep step={step} />
      <LeftOut result={result} />
    </section>
  );
}

function UploadSlot({ books, jobs }: { books: BookCardT[] | undefined; jobs: JobOut[] | undefined }) {
  const ingest = useIngest();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const submit = (file: File | undefined) => {
    if (file) ingest.mutate(file);
  };
  const error = ingest.error ? failureLine(ingest.error, "Couldn't add the book") : undefined;
  // one plain line for screen readers, in a region that is always mounted so every change is
  // announced; the panel with its link and disclosure sits outside it
  const outcome = ingest.isPending
    ? "Adding the book…"
    : ingest.isSuccess
      ? resultHeadline(ingest.data)
      : error
        ? `Couldn't add the book: ${error}`
        : "";

  return (
    <div className="uploadcell">
      <button
        id={ADD_ANCHOR}
        className={`drop ${over ? "over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          submit(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={input}
          type="file"
          accept=".epub,.pdf"
          hidden
          onChange={(e) => {
            submit(e.target.files?.[0] ?? undefined);
            e.target.value = ""; // picking the same file again must upload again
          }}
        />
        <span className="tag">{ingest.isPending ? "adding the book…" : "drop epub or pdf / click to browse"}</span>
        <span>The book appears here in seconds.</span>
        {error && <span className="err">{error}</span>}
      </button>
      <p className="sr-only" role="status">
        {outcome}
      </p>
      {ingest.isSuccess && (
        <IngestResultPanel key={ingest.submittedAt} result={ingest.data} books={books} jobs={jobs} />
      )}
    </div>
  );
}

export function Library() {
  const qc = useQueryClient();
  const books = useBooks();
  const shelfJobs = useShelfJobs();
  const empty = books.data?.books.length === 0;
  // the backend row of the checklist already says "not running" and how to start it
  const backendDown = isBackendDown(books.error);
  // the shelf noticed the outage before the probe's 30 s poll did: ask the probe once now, so
  // the checklist below agrees with this panel instead of showing a stale "Running"
  useEffect(() => {
    if (backendDown) void qc.invalidateQueries({ queryKey: ["system", "probe"] });
  }, [backendDown, qc]);
  const shelfCached = (books.data?.books.length ?? 0) > 0;
  // "Add a book" / "Add the book again" land here: bring the upload slot into view and focus
  // it, also when the link points at the page the user is already on
  const location = useLocation();
  useEffect(() => {
    if (location.hash !== `#${ADD_ANCHOR}`) return;
    const drop = document.getElementById(ADD_ANCHOR);
    drop?.scrollIntoView({ behavior: "smooth", block: "center" });
    drop?.focus({ preventScroll: true });
  }, [location.hash, location.key]);
  return (
    <section className="screen">
      <h1>Library</h1>
      <p className="sub">
        Every book under <span className="mono">books/</span> and <span className="mono">output/</span>, with its
        position on the signal path. Progress lives in the transport bar — cards only say what stage owns the book.
      </p>
      {books.isPending && <div className="loadline">reading the shelf…</div>}
      {books.isError && !backendDown && <PlainError error={books.error} title="couldn't load your books" />}
      {(empty || backendDown) && (
        <div className="panel">
          <div className="panel-h">
            <b>{backendDown ? "The Seiyuu backend isn't running" : "Before your first book"}</b>
          </div>
          <div className="panel-sub">
            {backendDown
              ? shelfCached
                ? "The shelf below is from before it stopped and can't change until it's back. Start it, then return to this page."
                : "Start it, then return to this page. Everything below should be green before you start."
              : "Everything below should be green before you start. Each fix says what to run."}
          </div>
          <SetupChecklist />
        </div>
      )}
      <div className="grid2">
        <UploadSlot books={books.data?.books} jobs={shelfJobs.data?.jobs} />
        {books.data?.books.map((b) => (
          <Card key={b.book_id} book={b} failed={unresolvedFailure(shelfJobs.data?.jobs, b.book_id)} />
        ))}
      </div>
    </section>
  );
}
