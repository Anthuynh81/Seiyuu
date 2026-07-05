import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  useBook,
  useBookJobs,
  useBooks,
  useEstimate,
  useMintQuote,
  useRenderSummary,
  useStartJob,
  useValidation,
} from "../api/hooks";
import type {
  ChapterSummary,
  CostEstimateOut,
  JobOut,
  QuoteResponse,
  RenderMode,
  ValidationRow,
} from "../api/types";

/* -------------------------------------------------- steps strip */

function Steps() {
  return (
    <div className="steps">
      <div className="step">
        <span className="n">STEP 1</span>
        <b>Estimate</b>
        <span>free &amp; instant — see what's cached, what's local, what would bill</span>
      </div>
      <div className="step">
        <span className="n">STEP 2</span>
        <b>Approve</b>
        <span>paid work mints a quote: a single-use price ticket, valid 15 minutes</span>
      </div>
      <div className="step">
        <span className="n">STEP 3</span>
        <b>Render</b>
        <span>the job appears on the right, cancellable from the transport bar</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------- quote ticket */

type TicketState =
  | { kind: "none" }
  | { kind: "live"; quote: QuoteResponse }
  | { kind: "refused"; quote: QuoteResponse; stamp: "DRIFT" | "EXPIRED" | "USED"; message: string };

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

function QuoteTicket({
  state,
  onConfirm,
  onRemint,
  onExpired,
  busy,
}: {
  state: TicketState;
  onConfirm: () => void;
  onRemint: () => void;
  onExpired: () => void;
  busy: boolean;
}) {
  const now = useNow(state.kind === "live");
  const quote = state.kind === "none" ? null : state.quote;
  const remaining = quote ? Math.max(0, quote.expires_at - now) : 0;
  const expired = state.kind === "live" && remaining <= 0;
  useEffect(() => {
    if (expired) onExpired();
  }, [expired]); // eslint-disable-line react-hooks/exhaustive-deps
  if (state.kind === "none" || quote === null) return null;
  const dead = state.kind === "refused";
  const fusePct = (remaining / quote.ttl_seconds) * 100;
  const mmss = `${Math.floor(remaining / 60)}:${String(Math.floor(remaining % 60)).padStart(2, "0")}`;
  return (
    <div className={`paper quote ${remaining < 60 && !dead ? "hot" : ""} ${dead ? "dead" : ""}`}>
      <div className="fuse" style={{ background: `linear-gradient(90deg, currentColor 0 ${fusePct}%, transparent ${fusePct}%)` }} />
      {dead && <div className="stamp">{state.stamp}</div>}
      <div className="quote-inner">
        <div className="head">
          <span>cost quote · single use</span>
          <span style={{ color: remaining < 60 && !dead ? "#8a6d00" : undefined }}>
            {dead ? "refused" : `expires ${mmss}`}
          </span>
        </div>
        <div className="line"><span>book</span><span>{quote.book_id}</span></div>
        <div className="line"><span>chapters</span><span>{quote.chapters.length ? quote.chapters.join(", ") : "all"}</span></div>
        <div className="line"><span>paid segments</span><span>{quote.paid_segments}</span></div>
        <div className="line"><span>ceiling</span><span>${quote.max_usd_ceiling.toFixed(2)} ok</span></div>
        {dead && <div className="line" style={{ color: "var(--clip)" }}><span>refusal</span><span>{state.message}</span></div>}
        <div className="total">
          <span className="tag" style={{ color: "var(--paper-ink-2)" }}>
            {dead ? "token intact — mint a fresh quote" : "step 3 — the render never bills past this"}
          </span>
          <b>${quote.total_usd.toFixed(2)}</b>
        </div>
        {dead ? (
          <button className="confirm" onClick={onRemint} disabled={busy}>RE-MINT QUOTE</button>
        ) : (
          <button className="confirm" onClick={onConfirm} disabled={busy}>
            {busy ? "STARTING…" : "CONFIRM & RENDER"}
          </button>
        )}
        <div className="sig">
          <span>fingerprint {quote.fingerprint.slice(0, 4)}…{quote.fingerprint.slice(-4)}</span>
          <span>sig …{quote.token.slice(-8)}</span>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------- chapter scope */

type Scope = { kind: "whole" } | { kind: "range"; from: number; to: number };

function scopeChapters(scope: Scope, total: number): number[] {
  if (scope.kind === "whole") return [];
  const from = Math.max(1, Math.min(scope.from, total));
  const to = Math.max(from, Math.min(scope.to, total));
  return Array.from({ length: to - from + 1 }, (_, i) => from + i);
}

function ScopeRow({
  scope,
  setScope,
  chapters,
  renderedSet,
}: {
  scope: Scope;
  setScope: (s: Scope) => void;
  chapters: ChapterSummary[];
  renderedSet: Set<number>;
}) {
  const total = chapters.length;
  const firstUnrendered = chapters.find((c) => !renderedSet.has(c.index))?.index;
  const selected = scopeChapters(scope, total);
  const speakable = selected.length
    ? chapters.filter((c) => selected.includes(c.index)).reduce((a, c) => a + c.speakable_blocks, 0)
    : chapters.reduce((a, c) => a + c.speakable_blocks, 0);

  return (
    <div className="scoperow">
      <span className="tag">scope</span>
      <button className={`chap ${scope.kind === "whole" ? "on" : ""}`} onClick={() => setScope({ kind: "whole" })}>
        whole book
      </button>
      <button
        className={`chap ${scope.kind === "range" ? "on" : ""}`}
        onClick={() => setScope({ kind: "range", from: firstUnrendered ?? 1, to: Math.min((firstUnrendered ?? 1) + 9, total) })}
      >
        chapter range
      </button>
      {scope.kind === "range" && (
        <>
          <label className="rangelbl">
            ch
            <input
              type="number"
              min={1}
              max={total}
              value={scope.from}
              onChange={(e) => setScope({ ...scope, from: Number(e.target.value) || 1 })}
            />
          </label>
          <label className="rangelbl">
            to
            <input
              type="number"
              min={1}
              max={total}
              value={scope.to}
              onChange={(e) => setScope({ ...scope, to: Number(e.target.value) || scope.from })}
            />
          </label>
          {firstUnrendered !== undefined && renderedSet.size > 0 && (
            <button
              className="key quiet"
              style={{ padding: "3px 9px" }}
              title={`chapters 1–${firstUnrendered - 1} already have audio`}
              onClick={() =>
                setScope({ kind: "range", from: firstUnrendered, to: Math.min(firstUnrendered + 9, total) })
              }
            >
              continue · next 10 from ch {firstUnrendered}
            </button>
          )}
        </>
      )}
      <span className="mono scopehint">
        {selected.length ? `${selected.length} chapter(s)` : `all ${total} chapters`} · {speakable.toLocaleString()}{" "}
        segments
        {renderedSet.size > 0 && ` · ${renderedSet.size} ch already rendered`}
      </span>
    </div>
  );
}

/* -------------------------------------------------- estimate + money flow */

function EstimateCard({
  est,
  mode,
  onMint,
  onFreeRender,
  minting,
  error,
}: {
  est: CostEstimateOut;
  mode: RenderMode;
  onMint: () => void;
  onFreeRender: () => void;
  minting: boolean;
  error: string | null;
}) {
  const totalSegs = est.paid_segments + est.cached_segments + est.free_segments || 1;
  const paid = est.total_usd > 0;
  const human = paid
    ? `Most of this render is free. ${est.paid_segments} segment(s) use a paid cloud voice — about $${est.total_usd.toFixed(2)}.`
    : est.cached_segments > 0 && est.free_segments === 0 && est.paid_segments === 0
      ? "Everything is already cached — re-rendering is instant and free."
      : "This render is entirely free: local voices only.";
  return (
    <div className="panel est" style={{ marginBottom: 0 }}>
      <div className="panel-h">
        <b>Step 1 — Estimate</b>
        <span className="tag" style={{ marginLeft: "auto" }}>{mode} · {est.chapters.length ? `ch ${est.chapters.join(",")}` : "whole book"}</span>
      </div>
      <div className="humanline">{human}</div>
      <div className="estbar">
        <i className="paid" style={{ width: `${(est.paid_segments / totalSegs) * 100}%` }} />
        <i className="cached" style={{ width: `${(est.cached_segments / totalSegs) * 100}%` }} />
        <i className="free" style={{ flex: 1 }} />
      </div>
      <div className="estrow"><span>paid segments</span><b>{est.paid_segments.toLocaleString()}</b></div>
      <div className="estrow"><span>cached — free to reuse</span><b>{est.cached_segments.toLocaleString()}</b></div>
      <div className="estrow"><span>free segments (local)</span><b>{est.free_segments.toLocaleString()}</b></div>
      {est.edit_warnings.length > 0 && (
        <div className="estwarn">
          <div>edit overlay — {est.edit_warnings.length} warning(s) apply to this estimate:</div>
          {est.edit_warnings.map((w) => <div key={w}>· {w}</div>)}
        </div>
      )}
      <div className="est acts">
        {paid ? (
          <button className="key" onClick={onMint} disabled={minting}>
            {minting ? "minting…" : `step 2 · approve — mint quote for $${est.total_usd.toFixed(2)}`}
          </button>
        ) : (
          <button className="key" onClick={onFreeRender}>render {mode} — free, nothing to approve</button>
        )}
      </div>
      {error && <div className="errline" style={{ margin: "0 14px 12px" }}>{error}</div>}
    </div>
  );
}

/* -------------------------------------------------- jobs + validation */

const LED: Record<string, string> = { queued: "q", running: "run", succeeded: "ok", failed: "err", canceled: "off" };

function JobRow({ job }: { job: JobOut }) {
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
            <span className="mono" style={{ color: "var(--ink-3)", fontSize: 10.5, marginLeft: 6 }}>
              {job.params.mode}
            </span>
          )}
          {detail && <span className="more">why? ▸</span>}
        </span>
        <span className="id">{job.job_id}</span>
        <span className="when">{new Date(job.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
      </div>
      {job.state === "running" && job.progress_text && (
        <div className="mono" style={{ fontSize: 11, color: "var(--ink-2)", marginTop: 6 }}>{job.progress_text}</div>
      )}
      {detail && <div className="jdetail"><div className="errblock">{detail}</div></div>}
    </div>
  );
}

function ValidationFailure({ bookId, row }: { bookId: string; row: ValidationRow }) {
  const [playing, setPlaying] = useState(false);
  const src = `/api/books/${bookId}/segments/${row.block_id}/audio?segment=${row.segment_index}`;
  return (
    <div className="valrow">
      <div className="vhead">
        <i className="led warn" />
        ch{row.chapter_index} · {row.block_id}[{row.segment_index}] · score {row.score.toFixed(2)}
        {row.voice_id && <span style={{ color: "var(--ink-3)" }}>· {row.voice_id}</span>}
        <button className="key quiet" style={{ marginLeft: "auto", padding: "2px 8px" }} onClick={() => setPlaying(!playing)}>
          {playing ? "hide player" : "▶ play segment"}
        </button>
      </div>
      <div className="vdiff">
        <div className="paper exp"><span className="cap" style={{ color: "var(--paper-ink-2)" }}>expected (book)</span>{row.expected}</div>
        <div className="got"><span className="cap" style={{ color: "var(--ink-3)" }}>whisper heard</span>{row.transcript}</div>
      </div>
      {playing && <audio controls autoPlay src={src} style={{ width: "100%", marginTop: 8, height: 32 }} />}
    </div>
  );
}

/* -------------------------------------------------- confirm-full dialog */

function ConfirmFullDialog({
  detail,
  onConfirm,
  onCancel,
}: {
  detail: { speakable_blocks: number; runtime_estimate_seconds: number };
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const hours = detail.runtime_estimate_seconds / 3600;
  return (
    <div className="overlay on">
      <div className="dialog">
        <div className="dh"><b>Full-book render</b><button className="key quiet" onClick={onCancel}>esc</button></div>
        <div className="db">
          <p style={{ margin: 0, color: "var(--ink-2)" }}>
            This renders <b className="mono">{detail.speakable_blocks.toLocaleString()}</b> segments — roughly{" "}
            <b className="mono">{hours.toFixed(1)} h</b> of audio to synthesize. A long GPU job; you can cancel it
            anytime from the transport bar and re-runs resume from the cache.
          </p>
        </div>
        <div className="df">
          <button className="key quiet" onClick={onCancel}>not now</button>
          <button className="key" onClick={onConfirm}>render the whole book</button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------- the screen */

export function RenderJobs() {
  const [params, setParams] = useSearchParams();
  const books = useBooks();
  const bookId = params.get("book") ?? books.data?.books[0]?.book_id ?? null;
  const book = useBook(bookId);

  const status = book.data?.status;
  const defaultMode: RenderMode = status?.assigned ? "multivoice" : "single";
  const [modeChoice, setModeChoice] = useState<RenderMode | null>(null);
  useEffect(() => setModeChoice(null), [bookId]); // a mode picked for one book must not stick to another
  const mode = modeChoice ?? defaultMode;

  const [scope, setScope] = useState<Scope>({ kind: "whole" });
  const chapterCount = book.data?.chapters?.length ?? 0;
  const chapters = useMemo(() => scopeChapters(scope, chapterCount), [scope, chapterCount]);

  const ready = !!status?.ingested && (mode === "single" || (status?.attributed && status?.assigned));
  const estimate = useEstimate(bookId, mode, chapters, ready);
  const jobs = useBookJobs(bookId);
  const validation = useValidation(bookId, !!status?.rendered);
  const summary = useRenderSummary(bookId, !!status?.rendered);
  const renderedSet = useMemo(
    () => new Set(summary.data?.chapters.map((c) => c.index) ?? []),
    [summary.data],
  );

  const mint = useMintQuote(bookId ?? "");
  const render = useStartJob(bookId ?? "", "render");
  const assemble = useStartJob(bookId ?? "", "assemble");
  const master = useStartJob(bookId ?? "", "master");

  const [ticket, setTicket] = useState<TicketState>({ kind: "none" });
  const [confirmFull, setConfirmFull] = useState<{ speakable_blocks: number; runtime_estimate_seconds: number } | null>(null);
  const [flowError, setFlowError] = useState<string | null>(null);

  const singleSpec = useMemo(() => (mode === "single" ? { single: {} } : {}), [mode]);

  const doMint = () => {
    setFlowError(null);
    mint.mutate(
      { mode, chapters },
      {
        onSuccess: (quote) => setTicket({ kind: "live", quote }),
        onError: (e) => setFlowError(e instanceof ApiError ? e.message : String(e)),
      },
    );
  };

  const startRender = (opts: { token?: string; confirmFull?: boolean }) => {
    setFlowError(null);
    render.mutate(
      { mode, chapters, ...singleSpec, ...(opts.token ? { cost_token: opts.token } : {}), ...(opts.confirmFull ? { confirm_full: true } : {}) },
      {
        onSuccess: () => {
          setTicket({ kind: "none" });
          setConfirmFull(null);
        },
        onError: (e) => {
          if (!(e instanceof ApiError)) return setFlowError(String(e));
          if (e.code === "full_render_confirmation_required") {
            setConfirmFull(e.detail as { speakable_blocks: number; runtime_estimate_seconds: number });
          } else if (e.code === "quote_expired" && ticket.kind === "live") {
            // per the design contract: expiry re-mints silently
            setTicket({ kind: "none" });
            doMint();
          } else if ((e.code === "cost_drift" || e.code === "quote_mismatch" || e.code === "quote_used") && ticket.kind === "live") {
            setTicket({
              kind: "refused",
              quote: ticket.quote,
              stamp: e.code === "quote_used" ? "USED" : "DRIFT",
              message: e.message,
            });
          } else {
            setFlowError(e.message);
          }
        },
      },
    );
  };

  if (books.isPending) return <section className="screen"><div className="loadline">reading the shelf…</div></section>;
  if (!bookId || !books.data?.books.length) {
    return (
      <section className="screen">
        <h1>Render &amp; Jobs</h1>
        <p className="sub">No books yet — ingest an EPUB from the Library first.</p>
      </section>
    );
  }

  return (
    <section className="screen">
      <h1 style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
        Render &amp; Jobs
        <select
          className="bookpick"
          value={bookId}
          onChange={(e) => setParams({ book: e.target.value })}
          aria-label="book"
        >
          {books.data.books.map((b) => (
            <option key={b.book_id} value={b.book_id}>{b.title ?? b.book_id}</option>
          ))}
        </select>
      </h1>
      <p className="sub">
        Turn the reviewed book into audio. Free renders start with one click; anything that costs money asks you to
        approve the exact price first.
      </p>
      <Steps />
      <div className="rjgrid">
        <div>
          <div className="modewrap">
            <span className="tag">mode</span>
            <button className={`chap ${mode === "multivoice" ? "on" : ""}`} disabled={!status?.assigned}
              title={status?.assigned ? "" : "assign voices first (Character Review → cast)"}
              onClick={() => setModeChoice("multivoice")}>
              multivoice
            </button>
            <button className={`chap ${mode === "single" ? "on" : ""}`} onClick={() => setModeChoice("single")}>
              single voice
            </button>
          </div>
          {!ready && status && (
            <div className="refusal" style={{ marginBottom: 16 }}>
              <span className="tag">stage_prerequisite</span>
              <p>
                {!status.ingested
                  ? "this book has no normalized text yet — re-ingest it"
                  : "multivoice needs attribution + a voice assignment first — or switch to single voice"}
              </p>
            </div>
          )}
          {status?.assigned && mode === "single" && (
            <div className="refusal" style={{ marginBottom: 14 }}>
              <span className="tag">casting ignored</span>
              <p>
                this book HAS a casting, but single-voice renders everything with one voice —{" "}
                <button
                  className="link"
                  style={{ background: "none", border: "none", color: "var(--tungsten)", cursor: "pointer", padding: 0 }}
                  onClick={() => setModeChoice("multivoice")}
                >
                  switch to multivoice
                </button>
              </p>
            </div>
          )}
          {book.data?.chapters && (
            <ScopeRow scope={scope} setScope={setScope} chapters={book.data.chapters} renderedSet={renderedSet} />
          )}
          {estimate.isPending && ready && <div className="loadline">estimating against the segment cache…</div>}
          {estimate.isError && <div className="errline">{estimate.error.message}</div>}
          {estimate.data && (
            <EstimateCard
              est={estimate.data}
              mode={mode}
              onMint={doMint}
              onFreeRender={() => startRender({})}
              minting={mint.isPending}
              error={flowError}
            />
          )}
          <QuoteTicket
            state={ticket}
            busy={render.isPending}
            onConfirm={() => ticket.kind === "live" && startRender({ token: ticket.quote.token })}
            onRemint={doMint}
            onExpired={() =>
              ticket.kind === "live" &&
              setTicket({ kind: "refused", quote: ticket.quote, stamp: "EXPIRED", message: "the 15-minute window passed" })
            }
          />
          <details className="fine">
            <summary>what if the price changes or the ticket expires?</summary>
            You never get double-billed: the ticket is single-use and bound to the exact work you saw. If anything
            shifts before you confirm — the cache, an edit, the clock — the paper is stamped and the confirm key
            deadens; you simply mint a fresh quote.
          </details>
        </div>

        <div>
          <div className="panel joblist">
            <div className="panel-h"><b>Jobs</b><span className="tag" style={{ marginLeft: "auto" }}>live</span></div>
            <div className="panel-sub">Everything the machine is doing or has done for this book, newest first. Click a failed row for its reason.</div>
            {jobs.data?.jobs.length === 0 && <div className="loadline" style={{ padding: "14px" }}>no jobs yet</div>}
            {jobs.data?.jobs.map((j) => <JobRow key={j.job_id} job={j} />)}
          </div>

          <div className="panel dl">
            <div className="panel-h">
              <b>Outputs</b>
              <button className="key quiet" style={{ marginLeft: "auto" }} disabled={!status?.rendered || assemble.isPending}
                onClick={() => assemble.mutate({})}>
                assemble
              </button>
              <button className="key" style={{ marginLeft: 10 }} disabled={!status?.rendered || master.isPending}
                onClick={() => master.mutate({})}>
                master m4b
              </button>
            </div>
            <div className="panel-sub">
              After a render: <b>assemble</b> makes per-chapter MP3s, <b>master</b> makes the final chaptered .m4b audiobook.
            </div>
            {(assemble.error || master.error) && (
              <div className="errline" style={{ margin: 12 }}>{(assemble.error ?? master.error)?.message}</div>
            )}
            <table>
              <tbody>
                <tr>
                  <td className="mono">{bookId}.m4b</td>
                  <td className="mono" style={{ color: book.data?.downloads.m4b ? "var(--ink-2)" : "var(--ink-3)" }}>
                    {book.data?.downloads.m4b ? `${(book.data.downloads.m4b.bytes / 1048576).toFixed(1)} MB` : "not yet mastered"}
                  </td>
                  <td>{book.data?.downloads.m4b ? <a className="link" href={book.data.downloads.m4b.url}>download</a> : "—"}</td>
                </tr>
                {book.data?.downloads.chapter_mp3s.map((c) => (
                  <tr key={c.index}>
                    <td className="mono">chapters/ch{String(c.index).padStart(3, "0")}.mp3</td>
                    <td className="mono" style={{ color: "var(--ink-2)" }}>{(c.bytes / 1048576).toFixed(1)} MB</td>
                    <td><a className="link" href={c.url}>download</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {validation.data && validation.data.validated_segments > 0 && (
            <div className="panel">
              <div className="panel-h">
                <b>Audio checks</b>
                <span className="tag" style={{ marginLeft: "auto" }}>
                  {validation.data.validation_failures} failed of {validation.data.validated_segments} checked
                </span>
              </div>
              <div className="panel-sub">
                Cloned-voice segments are transcribed back by whisper; a mismatch means the voice may have misread.
                Listen before shipping.
              </div>
              <div style={{ padding: "0 14px 14px" }}>
                {validation.data.results.length === 0 && <div className="loadline">all checked segments read clean</div>}
                {validation.data.results.map((r) => (
                  <ValidationFailure key={`${r.block_id}-${r.segment_index}`} bookId={bookId} row={r} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      {confirmFull && (
        <ConfirmFullDialog
          detail={confirmFull}
          onConfirm={() => startRender({ confirmFull: true, token: ticket.kind === "live" ? ticket.quote.token : undefined })}
          onCancel={() => setConfirmFull(null)}
        />
      )}
    </section>
  );
}
