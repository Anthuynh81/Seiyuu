import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { useBooks, useIngest } from "../api/hooks";
import type { BookCard as BookCardT } from "../api/types";
import { KIND_STAGE, STAGES } from "../api/types";

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
  if (!job) return <div className="jobline" style={{ color: "var(--ink-3)" }}>no job active</div>;
  return (
    <div className="jobline">
      <i className={`led ${job.state === "running" ? "run" : "q"}`} />
      {job.kind} {job.job_id} — {job.state}
      {job.state === "queued" && " · waiting for the worker"}
    </div>
  );
}

function Card({ book }: { book: BookCardT }) {
  const nav = useNavigate();
  const go = (screen: string) => nav(`/${screen}?book=${encodeURIComponent(book.book_id)}`);
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
      </div>
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
        accept=".epub"
        hidden
        onChange={(e) => submit(e.target.files?.[0] ?? undefined)}
      />
      <span className="tag">{ingest.isPending ? "ingesting…" : "drop epub / click to browse"}</span>
      <span>Ingest runs synchronously — the book appears here in seconds.</span>
      {ingest.isSuccess && (
        <span className="mono" style={{ color: "var(--ok)", fontSize: 11 }}>
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
