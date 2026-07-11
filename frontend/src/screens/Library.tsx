import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { useBooks, useDeleteBook, useIngest } from "../api/hooks";
import type { BookCard as BookCardT, JobOut, PaidArtifacts } from "../api/types";
import { KIND_STAGE, STAGES } from "../api/types";
import { TalkDialog } from "../components/Dialog";

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
  if (!job) return <div className="jobline text-ink-3">no job active</div>;
  return (
    <div className="jobline">
      <i className={`led ${job.state === "running" ? "run" : "q"}`} />
      {job.kind} {job.job_id} — {job.state}
      {job.state === "queued" && " · waiting for the worker"}
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
              {del.isPending ? "deleting…" : `discard ${paid.paid_segment_count} paid segment(s) & delete`}
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
          <span className="tag">conflicting_job</span>
          <p>
            a {conflict.kind} job is {conflict.state} for this book — cancel it in the transport bar below, then
            delete.
          </p>
        </div>
      ) : partial ? (
        <div className="refusal">
          <span className="tag">partial_delete</span>
          <p>
            the delete only partly completed — these paths survived and may need a retry or a manual sweep:
            <br />
            {partial.survivors.map((s) => (
              <span key={s} className="mono block text-caution">{s}</span>
            ))}
          </p>
        </div>
      ) : paid ? (
        <>
          <p className="mb-2.5 mt-0 text-ink-2">
            <b className="text-clip">{paid.paid_segment_count}</b> paid cloud segment(s) were rendered
            for <b>{title}</b> — deleting the book discards them, and reproducing them costs real money
            {paid.estimated_usd !== null && ` (~$${paid.estimated_usd.toFixed(2)})`}.
          </p>
          <div className="mono text-[11px] text-ink-3">
            engines: {paid.engines.join(", ") || "—"}
            {paid.paid_voice_ids.length > 0 && ` · voices: ${paid.paid_voice_ids.join(", ")}`}
          </div>
        </>
      ) : (
        <p className="m-0 text-ink-2">
          Permanently delete <b>{title}</b> — its ingest, attribution, casting, and any rendered/assembled audio.
          This cannot be undone.
        </p>
      )}
      {otherErr && <div className="errline mt-3">{otherErr.message}</div>}
    </TalkDialog>
  );
}

function Card({ book }: { book: BookCardT }) {
  const nav = useNavigate();
  const go = (screen: string) => nav(`/${screen}?book=${encodeURIComponent(book.book_id)}`);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const blocked = book.active_job !== null;
  return (
    <div className="bookcard">
      <h3>{book.title ?? book.book_id}</h3>
      <div className="by">{book.authors.join(", ") || "—"}</div>
      <SignalPath book={book} />
      <JobLine book={book} />
      <div className="acts">
        {book.rendered && (
          <button className="key" onClick={() => go("listen")}>
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

function UploadSlot() {
  const ingest = useIngest();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const submit = (file: File | undefined) => {
    if (file) ingest.mutate(file);
  };
  const error = ingest.error instanceof ApiError ? ingest.error.message : ingest.error?.message;

  return (
    <button
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
        onChange={(e) => submit(e.target.files?.[0] ?? undefined)}
      />
      <span className="tag">{ingest.isPending ? "ingesting…" : "drop epub or pdf / click to browse"}</span>
      <span>Ingest runs synchronously — the book appears here in seconds.</span>
      {ingest.isSuccess && (
        <span className="mono text-[11px] text-ok">
          {ingest.data.book.book_id} · {ingest.data.chapters} chapters, {ingest.data.blocks} blocks
        </span>
      )}
      {error && <span className="err">{error}</span>}
    </button>
  );
}

export function Library() {
  const books = useBooks();
  return (
    <section className="screen">
      <h1>Library</h1>
      <p className="sub">
        Every book under <span className="mono">books/</span> and <span className="mono">output/</span>, with its
        position on the signal path. Progress lives in the transport bar — cards only say what stage owns the book.
      </p>
      {books.isPending && <div className="loadline">reading the shelf…</div>}
      {books.isError && <div className="errline">{books.error.message}</div>}
      <div className="grid2">
        <UploadSlot />
        {books.data?.books.map((b) => <Card key={b.book_id} book={b} />)}
      </div>
    </section>
  );
}
