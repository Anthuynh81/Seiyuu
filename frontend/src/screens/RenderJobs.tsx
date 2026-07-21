import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  useAttribute,
  useBook,
  useBookJobs,
  useBooks,
  useDeleteCover,
  useEstimate,
  useMintQuote,
  useRenderSummary,
  useStartJob,
  useSwitchRenderMode,
  useSystem,
  useUploadCover,
  useValidation,
} from "../api/hooks";
import type {
  ActiveJobSummary,
  ArchivedRenderMode,
  BookCard,
  ChapterSummary,
  CostEstimateOut,
  CoverOut,
  JobOut,
  QuoteResponse,
  RenderMode,
  RenderSummaryOut,
  ValidationRow,
} from "../api/types";
import { KIND_STAGE, STAGES } from "../api/types";
import { TalkDialog } from "../components/Dialog";
import { TalkSelect } from "../components/Select";
import { StaleAudioBanner } from "../components/StaleAudio";
import { Tip } from "../components/Tooltip";
import { classifyRenderFailure } from "../lib/money";
import { continueRange, scopeChapters, type Scope } from "../lib/scope";

/* -------------------------------------------------- pipeline rail */

/** The whole signal path in one glance — which stages are done, which one is running,
    and what to do next. Replaces the old three-paragraph "steps" banner and answers
    "what do I click after render?" without the user memorizing the order. */
function StageRail({
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
        {dead && <div className="line text-clip"><span>refusal</span><span>{state.message}</span></div>}
        <div className="total">
          <span className="tag text-paper-ink-2">
            {dead ? "token intact — mint a fresh quote" : "the render never bills past this"}
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
  const cont = continueRange(renderedSet, total, 10);
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
        onClick={() => setScope(cont ?? { kind: "range", from: 1, to: Math.min(10, total) })}
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
          {cont && (
            <Tip content="the next ten chapters without rendered audio">
              <button className="key quiet px-[9px] py-[3px]" onClick={() => setScope(cont)}>
                continue · next 10 from ch {cont.from}
              </button>
            </Tip>
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

/* -------------------------------------------------- active render mode (instant fallback) */

/** Which archived render manifest.json points at. Completed renders are KEPT per mode
    (manifest.single.json / manifest.multi.json); switching is an atomic pointer move on the
    server — no synthesis, no cache touch — so falling back from multivoice to single voice
    (or back) is instant and free. Only mounted once at least one mode has been rendered. */
function ActiveModeControl({
  bookId,
  summary,
  conflict,
}: {
  bookId: string;
  summary: RenderSummaryOut;
  conflict: ActiveJobSummary | null;
}) {
  const sw = useSwitchRenderMode(bookId);
  const apiErr = sw.error instanceof ApiError ? sw.error : null;
  const modes: { mode: ArchivedRenderMode; label: string }[] = [
    { mode: "multi", label: "multivoice" },
    { mode: "single", label: "single voice" },
  ];
  return (
    <div className="mb-3.5">
      <div className="scoperow mb-0" role="group" aria-label="active render mode">
        <span className="tag">listening to</span>
        {modes.map(({ mode, label }) => {
          const onDisk = summary.available_modes.includes(mode);
          const active = summary.active_mode === mode;
          return (
            <button
              key={mode}
              className={`chap ${active ? "on" : ""}`}
              disabled={active || !onDisk || conflict !== null || sw.isPending}
              title={onDisk ? "" : `no ${label} render on disk yet — render it once to switch freely`}
              onClick={() => sw.mutate(mode)}
            >
              {label} · {active ? "active" : onDisk ? "switch" : "not rendered"}
            </button>
          );
        })}
        <span className="mono scopehint">
          {conflict
            ? `a ${conflict.kind} job is ${conflict.state} and owns the render — wait for it or cancel it first`
            : summary.available_modes.length > 1
              ? "both renders are kept — switching is instant, nothing re-renders"
              : "render the other mode once and you can switch between them instantly"}
        </span>
      </div>
      {apiErr &&
        (apiErr.code === "conflicting_job" ? (
          <div className="refusal mt-2">
            <span className="tag">conflicting_job</span>
            <p>{apiErr.message}</p>
          </div>
        ) : (
          <div className="errline mt-2">{apiErr.message}</div>
        ))}
      {sw.error && !apiErr && <div className="errline mt-2">{String(sw.error)}</div>}
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
    <div className="panel est mb-0">
      <div className="panel-h">
        <b>Estimate</b>
        <span className="tag ml-auto">{mode} · {est.chapters.length ? `ch ${est.chapters.join(",")}` : "whole book"}</span>
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
            {minting ? "minting…" : `approve — mint quote for $${est.total_usd.toFixed(2)}`}
          </button>
        ) : (
          <button className="key" onClick={onFreeRender}>render {mode} — free, nothing to approve</button>
        )}
      </div>
      {error && <div className="errline mx-3.5 mb-3">{error}</div>}
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

function ValidationFailure({ bookId, row }: { bookId: string; row: ValidationRow }) {
  const [playing, setPlaying] = useState(false);
  const src = `/api/books/${bookId}/segments/${row.block_id}/audio?segment=${row.segment_index}`;
  return (
    <div className="valrow">
      <div className="vhead">
        <i className="led warn" />
        ch{row.chapter_index} · {row.block_id}[{row.segment_index}] · score {row.score.toFixed(2)}
        {row.voice_id && <span className="text-ink-3">· {row.voice_id}</span>}
        <button className="key quiet ml-auto px-2 py-[2px]" onClick={() => setPlaying(!playing)}>
          {playing ? "hide player" : "▶ play segment"}
        </button>
      </div>
      <div className="vdiff">
        <div className="paper exp"><span className="cap text-paper-ink-2">expected (book)</span>{row.expected}</div>
        <div className="got"><span className="cap text-ink-3">whisper heard</span>{row.transcript}</div>
      </div>
      {playing && <audio controls autoPlay src={src} className="mt-2 h-8 w-full" />}
    </div>
  );
}

/* -------------------------------------------------- cover art (master + shelf) */

function CoverControl({ bookId, cover }: { bookId: string; cover: CoverOut | null }) {
  const upload = useUploadCover(bookId);
  const remove = useDeleteCover(bookId);
  const input = useRef<HTMLInputElement>(null);
  const [confirming, setConfirming] = useState(false);
  // GET /cover is a fixed URL, so REPLACING a cover would leave the browser's cached image
  // in place — bump a ?v= buster per successful upload (the read-along's audio_key trick)
  const [bust, setBust] = useState(0);

  const submit = (file: File | undefined) => {
    if (!file) return;
    remove.reset();
    upload.mutate(file, { onSuccess: () => setBust(Date.now()) });
  };
  const err = upload.error ?? remove.error;
  const apiErr = err instanceof ApiError ? err : null;

  return (
    <div className="border-t border-hairline px-3.5 py-3">
      <div className="flex items-center gap-3">
        {cover ? (
          <img
            src={`/api/books/${bookId}/cover${bust ? `?v=${bust}` : ""}`}
            alt="cover art"
            className="h-14 w-auto border border-hairline"
          />
        ) : (
          <span aria-hidden className="h-14 w-10 shrink-0 border border-dashed border-hairline" />
        )}
        <div className="min-w-0">
          <span className="tag">cover art</span>
          <div className="mono text-[11px] text-ink-2">
            {cover
              ? `${cover.content_type} · ${Math.max(1, Math.round(cover.bytes / 1024))} KB — lands in the .m4b and on the Listen shelf`
              : "none yet — a jpeg or png here lands in the .m4b and on the Listen shelf"}
          </div>
        </div>
        <input
          ref={input}
          type="file"
          accept="image/jpeg,image/png"
          hidden
          onChange={(e) => {
            submit(e.target.files?.[0] ?? undefined);
            e.target.value = ""; // re-picking the same file (after fixing a refusal) must fire change again
          }}
        />
        <button
          className="key quiet ml-auto shrink-0"
          disabled={upload.isPending}
          onClick={() => input.current?.click()}
        >
          {upload.isPending ? "uploading…" : cover ? "replace" : "upload cover"}
        </button>
        {cover &&
          (confirming ? (
            <>
              <button
                className="key danger shrink-0"
                disabled={remove.isPending}
                onClick={() => remove.mutate(undefined, { onSettled: () => setConfirming(false) })}
              >
                {remove.isPending ? "removing…" : "really remove"}
              </button>
              <button className="key quiet shrink-0" onClick={() => setConfirming(false)}>
                keep
              </button>
            </>
          ) : (
            <button
              className="key quiet shrink-0"
              onClick={() => {
                upload.reset();
                setConfirming(true);
              }}
            >
              remove
            </button>
          ))}
      </div>
      {apiErr &&
        (apiErr.code === "conflicting_job" ? (
          <div className="refusal mt-2.5">
            <span className="tag">conflicting_job</span>
            <p>
              a master job is running and reads the cover mid-run — wait for it or cancel it in the transport bar,
              then retry.
            </p>
          </div>
        ) : (
          <div className="errline mt-2.5">{apiErr.message}</div>
        ))}
      {err && !apiErr && <div className="errline mt-2.5">{String(err)}</div>}
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
    <TalkDialog
      title="Full-book render"
      onClose={onCancel}
      footer={
        <>
          <button className="key quiet" onClick={onCancel}>not now</button>
          <button className="key" onClick={onConfirm}>render the whole book</button>
        </>
      }
    >
      <p className="m-0 text-ink-2">
        This renders <b className="mono">{detail.speakable_blocks.toLocaleString()}</b> segments — roughly{" "}
        <b className="mono">{hours.toFixed(1)} h</b> of audio to synthesize. A long GPU job; you can cancel it
        anytime from the transport bar and re-runs resume from the cache.
      </p>
    </TalkDialog>
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

  const system = useSystem();
  // F2b: per-render emotion override. null = follow the server default; a bool overrides it.
  // Only meaningful for multivoice (single-voice has no attribution to tag), so single passes
  // undefined and its estimate/quote/render stay emotion-agnostic.
  const emotionDefault = system.data?.apply_emotion ?? false;
  const [emotionChoice, setEmotionChoice] = useState<boolean | null>(null);
  useEffect(() => setEmotionChoice(null), [bookId]); // a choice for one book must not stick
  const applyEmotion = emotionChoice ?? emotionDefault;
  const emotionArg = mode === "multivoice" ? applyEmotion : undefined;

  // Re-render: bypass the segment cache for the in-scope chapters (redo a bad chapter, or
  // re-hear a changed cast). Reset on a real book switch so it never sticks across books.
  const [force, setForce] = useState(false);
  useEffect(() => setForce(false), [bookId]);

  const ready = !!status?.ingested && (mode === "single" || (status?.attributed && status?.assigned));
  const estimate = useEstimate(bookId, mode, chapters, ready, emotionArg, force);
  const jobs = useBookJobs(bookId);
  const validation = useValidation(bookId, !!status?.rendered);
  const summary = useRenderSummary(bookId, !!status?.rendered);
  const renderedSet = useMemo(
    () => new Set(summary.data?.chapters.map((c) => c.index) ?? []),
    [summary.data],
  );
  // mirrors the backend's manifest owners: these job kinds make the mode switch 409
  const activeJob = book.data?.active_job ?? null;
  const manifestOwner =
    activeJob && ["render", "assemble", "master"].includes(activeJob.kind) ? activeJob : null;

  const mint = useMintQuote(bookId ?? "");
  const attribute = useAttribute(bookId ?? "");
  const [llm, setLlm] = useState<{ provider: "local" | "anthropic"; model: string } | null>(null);
  const [llmOpen, setLlmOpen] = useState(false);
  const attrDefaults = system.data?.attribution;
  const effLlm =
    llm ??
    (attrDefaults
      ? { provider: attrDefaults.provider as "local" | "anthropic", model: attrDefaults.model }
      : null);
  const startAttribute = (confirmPaid = false) =>
    attribute.mutate({
      chapters,
      ...(effLlm ? { provider: effLlm.provider, model: effLlm.model } : {}),
      ...(confirmPaid ? { confirm_paid: true } : {}),
    });
  const attrErr = attribute.error instanceof ApiError ? attribute.error : null;
  const render = useStartJob(bookId ?? "", "render");
  const assemble = useStartJob(bookId ?? "", "assemble");
  const master = useStartJob(bookId ?? "", "master");

  // One-click finish: run assemble, then master, advancing when the watched job lands
  // (the useBookJobs poll is the clock). Client-side chain — it stops if the page closes,
  // which the checkbox label says out loud.
  const [chain, setChain] = useState<{ watch: string | null; queue: ("assemble" | "master")[] } | null>(null);
  const [chainNote, setChainNote] = useState<string | null>(null);
  const [autoFinish, setAutoFinish] = useState(false);
  const chainStep = (queue: ("assemble" | "master")[]) => {
    const [next, ...rest] = queue;
    if (!next) {
      setChain(null);
      setChainNote("finished — the .m4b is ready below");
      return;
    }
    setChain({ watch: null, queue: rest }); // "starting" — blocks the watcher until the job id lands
    (next === "assemble" ? assemble : master).mutate(
      {},
      {
        onSuccess: (job) => setChain({ watch: job.job_id, queue: rest }),
        onError: (e) => {
          setChain(null);
          setChainNote(`auto-finish stopped at ${next}: ${e instanceof ApiError ? e.message : String(e)}`);
        },
      },
    );
  };
  useEffect(() => {
    if (!chain?.watch) return;
    const j = jobs.data?.jobs.find((x) => x.job_id === chain.watch);
    if (!j?.is_terminal) return;
    if (j.state === "succeeded") chainStep(chain.queue);
    else {
      setChain(null);
      setChainNote(`auto-finish stopped — the ${j.kind} job ${j.state}${j.error ? `: ${j.error}` : ""}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs.data, chain]);
  // Reset only on a REAL book switch. bookId flickers to null whenever the live-job count
  // changes (useBooks re-keys and refetches) — and that happens precisely when a chained job
  // lands, which must not kill the chain.
  const lastBook = useRef<string | null>(null);
  useEffect(() => {
    if (bookId === null || bookId === lastBook.current) return;
    lastBook.current = bookId;
    setChain(null);
    setChainNote(null);
    setAutoFinish(false);
  }, [bookId]);
  const chainBusy = chain !== null || assemble.isPending || master.isPending;

  const [ticket, setTicket] = useState<TicketState>({ kind: "none" });
  // Toggling force re-prices the render (cached segments become billable/free work), so any
  // live quote no longer matches — drop it and make the user mint against the new estimate.
  useEffect(() => setTicket({ kind: "none" }), [force]);
  const [confirmFull, setConfirmFull] = useState<{ speakable_blocks: number; runtime_estimate_seconds: number } | null>(null);
  const [flowError, setFlowError] = useState<string | null>(null);

  const singleSpec = useMemo(() => (mode === "single" ? { single: {} } : {}), [mode]);

  const doMint = () => {
    setFlowError(null);
    mint.mutate(
      { mode, chapters, applyEmotion: emotionArg, force },
      {
        onSuccess: (quote) => setTicket({ kind: "live", quote }),
        onError: (e) => setFlowError(e instanceof ApiError ? e.message : String(e)),
      },
    );
  };

  const startRender = (opts: { token?: string; confirmFull?: boolean }) => {
    setFlowError(null);
    render.mutate(
      {
        mode,
        chapters,
        ...singleSpec,
        ...(emotionArg === undefined ? {} : { apply_emotion: emotionArg }),
        ...(force ? { force: true } : {}),
        ...(opts.token ? { cost_token: opts.token } : {}),
        ...(opts.confirmFull ? { confirm_full: true } : {}),
      },
      {
        onSuccess: (job) => {
          setTicket({ kind: "none" });
          setConfirmFull(null);
          if (autoFinish) {
            setChainNote(null);
            setChain({ watch: job.job_id, queue: ["assemble", "master"] });
          }
        },
        onError: (e) => {
          const action = classifyRenderFailure(e, ticket.kind === "live");
          switch (action.kind) {
            case "confirm-full":
              setConfirmFull(action.detail);
              break;
            case "remint":
              // per the design contract: expiry re-mints silently
              setTicket({ kind: "none" });
              doMint();
              break;
            case "stamp":
              if (ticket.kind === "live") {
                setTicket({ kind: "refused", quote: ticket.quote, stamp: action.stamp, message: action.message });
              }
              break;
            case "error":
              setFlowError(action.message);
          }
        },
      },
    );
  };

  if (books.isPending) return <section className="screen"><div className="loadline">reading the shelf…</div></section>;
  if (books.isError) return <section className="screen"><h1>Render &amp; Jobs</h1><div className="errline">{books.error.message}</div></section>;
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
      <h1 className="flex items-baseline gap-3.5">
        Render &amp; Jobs
        <TalkSelect
          className="bookpick"
          ariaLabel="book"
          value={bookId}
          onChange={(v) => setParams({ book: v })}
          options={books.data.books.map((b) => ({ value: b.book_id, label: b.title ?? b.book_id }))}
        />
      </h1>
      <p className="sub">
        Free renders start with one click; paid work always shows its exact price first.
      </p>
      {book.isError && <div className="errline">{book.error.message}</div>}
      <StageRail status={status} activeJob={book.data?.active_job ?? null} />
      <StaleAudioBanner status={status} />
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
            <div className="refusal mb-4">
              <span className="tag">stage_prerequisite</span>
              <p>
                {!status.ingested
                  ? "this book has no normalized text yet — re-ingest it"
                  : "multivoice needs attribution + a voice assignment first — or switch to single voice"}
              </p>
            </div>
          )}
          {status?.assigned && mode === "single" && (
            <div className="refusal mb-3.5">
              <span className="tag">casting ignored</span>
              <p>
                this book HAS a casting, but single-voice renders everything with one voice —{" "}
                <button className="link" onClick={() => setModeChoice("multivoice")}>
                  switch to multivoice
                </button>
              </p>
            </div>
          )}
          {book.data?.chapters && (
            <ScopeRow scope={scope} setScope={setScope} chapters={book.data.chapters} renderedSet={renderedSet} />
          )}
          {status?.rendered && (
            <div className="scoperow">
              <span className="tag">re-render</span>
              <label
                className="mono flex items-center gap-1.5 text-xs"
                title="Re-synthesize the in-scope chapters from scratch, overwriting the cached audio. Use it to redo a bad chapter (a misread that slipped the check) — pick a chapter range above, then force. Paid voices re-bill, so the estimate re-prices what force will re-synthesize."
              >
                <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
                force re-render — ignore cached audio
              </label>
              <span className="mono scopehint">
                {force
                  ? "every in-scope segment is re-synthesized fresh — a bad chapter gets redone"
                  : "cached segments are reused — nothing already rendered is re-synthesized"}
              </span>
            </div>
          )}
          {summary.data && (
            <ActiveModeControl bookId={bookId} summary={summary.data} conflict={manifestOwner} />
          )}
          {mode === "multivoice" &&
            summary.data?.rendered_assignment_hash &&
            estimate.data?.assignment_hash &&
            summary.data.rendered_assignment_hash !== estimate.data.assignment_hash && (
              <div className="refusal mb-3.5">
                <span className="tag">casting changed</span>
                <p>
                  the current cast differs from the rendered audio — re-render (multivoice) to hear
                  the new voices. Only the reassigned characters' lines re-synthesize; every
                  unchanged voice is reused from cache, so this is fast and mostly free.
                </p>
              </div>
            )}
          {mode === "multivoice" && (
            <div className="scoperow">
              <span className="tag">emotion</span>
              <label
                className="mono flex items-center gap-1.5 text-xs"
                title="voice each dialogue line with the emotion attribution tagged it. Off keeps delivery flat and the render cache byte-identical to a no-emotion render."
              >
                <input
                  type="checkbox"
                  checked={applyEmotion}
                  onChange={(e) => setEmotionChoice(e.target.checked)}
                />
                voice per-line emotion
              </label>
              <span className="mono scopehint">
                {applyEmotion
                  ? "dialogue lines are voiced with their tagged emotion"
                  : "flat delivery — emotion tags ignored (cache-stable)"}
                {emotionChoice === null && ` · system default: ${emotionDefault ? "on" : "off"}`}
              </span>
            </div>
          )}
          {status?.ingested && !status.attributed && (
            book.data?.active_job?.kind === "attribute" ? (
              <div className="drainstrip">
                <span className="state"><i className="led run" />attribution running</span>
                <span className="mono text-[11.5px] text-ink-2">
                  follow it in the transport bar — Character Review and the Listen read-along unlock when it lands
                </span>
              </div>
            ) : (
              <div className="refusal mb-3.5">
                <span className="tag">not attributed</span>
                <p>multivoice, Character Review, and the read-along need speaker attribution — single voice doesn't.</p>
                <p className="mt-2">
                  <button className="key" disabled={attribute.isPending} onClick={() => startAttribute()}>
                    {attribute.isPending
                      ? "starting…"
                      : `▶ attribute ${chapters.length ? `ch ${chapters[0]}–${chapters[chapters.length - 1]}` : "the whole book"}`}
                  </button>
                </p>
                <p className="mono mt-1.5 text-[11px] text-ink-2">
                  reader: {effLlm ? `${effLlm.provider} · ${effLlm.model}` : "…"}
                  {effLlm?.provider === "local" && " (ollama must be running)"}{" "}
                  <button className="link" onClick={() => setLlmOpen(!llmOpen)}>
                    {llmOpen ? "done" : "change"}
                  </button>
                </p>
                {llmOpen && effLlm && attrDefaults && (
                  <p className="mt-1.5 flex items-center gap-2">
                    <TalkSelect
                      ariaLabel="attribution provider"
                      value={effLlm.provider}
                      onChange={(v) => {
                        const p = v as "local" | "anthropic";
                        setLlm({ provider: p, model: p === "local" ? attrDefaults.model : attrDefaults.anthropic_model });
                      }}
                      options={[
                        { value: "local", label: "local (ollama) — free" },
                        {
                          value: "anthropic",
                          label: `anthropic — paid${system.data?.keys.anthropic_configured ? "" : " (no key configured)"}`,
                          disabled: !system.data?.keys.anthropic_configured,
                        },
                      ]}
                    />
                    <input
                      type="text"
                      value={effLlm.model}
                      aria-label="model id"
                      onChange={(e) => setLlm({ provider: effLlm.provider, model: e.target.value })}
                      className="mono w-[180px] border border-hairline bg-booth-0 px-2 py-[5px] text-[11.5px] text-ink"
                    />
                  </p>
                )}
                {attrErr && (
                  <p className="mono mt-1.5 text-[11px] text-clip">
                    {attrErr.message}
                    {attrErr.code === "payment_confirmation_required" && (
                      <>
                        {" — "}
                        <button className="link" onClick={() => startAttribute(true)}>
                          confirm the paid run
                        </button>
                      </>
                    )}
                  </p>
                )}
              </div>
            )
          )}
          {estimate.isPending && ready && <div className="loadline">estimating against the segment cache…</div>}
          {estimate.isError && <div className="errline">{estimate.error.message}</div>}
          {estimate.data && (
            <>
              <EstimateCard
                est={estimate.data}
                mode={mode}
                onMint={doMint}
                onFreeRender={() => startRender({})}
                minting={mint.isPending}
                error={flowError}
              />
              <label className="mono mt-2 flex items-center gap-1.5 text-[11px] text-ink-2">
                <input type="checkbox" checked={autoFinish} onChange={(e) => setAutoFinish(e.target.checked)} />
                when the render lands, assemble + master automatically (keep this page open)
              </label>
            </>
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
            <div className="panel-h"><b>Jobs</b><span className="tag ml-auto">live</span></div>
            <div className="panel-sub">Newest first — click a failed row for its reason.</div>
            {jobs.data?.jobs.length === 0 && <div className="loadline p-3.5">no jobs yet</div>}
            {jobs.data?.jobs.map((j) => <JobRow key={j.job_id} job={j} />)}
          </div>

          <div className="panel dl">
            <div className="panel-h">
              <b>Outputs</b>
              <button className="key quiet ml-auto" disabled={!status?.rendered || chainBusy}
                onClick={() => assemble.mutate({})}>
                assemble
              </button>
              <button className="key quiet ml-2.5" disabled={!status?.rendered || chainBusy}
                onClick={() => master.mutate({})}>
                master
              </button>
              <button
                className="key ml-2.5"
                disabled={!status?.rendered || chainBusy}
                title="run assemble, then master — one click to the .m4b"
                onClick={() => {
                  setChainNote(null);
                  chainStep(["assemble", "master"]);
                }}
              >
                {chain ? "finishing…" : "finish · mp3s + m4b"}
              </button>
            </div>
            <div className="panel-sub">
              <b>assemble</b> → per-chapter MP3s · <b>master</b> → the chaptered .m4b — or <b>finish</b> runs both.
            </div>
            {chain && (
              <div className="drainstrip mx-3 my-2">
                <span className="state"><i className="led run" />auto-finish</span>
                <span className="mono text-[11px] text-ink-2">
                  {chain.watch ? "waiting for the current job" : "starting"}
                  {chain.queue.length > 0 && ` · then ${chain.queue.join(" → ")}`} — keep this page open
                </span>
              </div>
            )}
            {chainNote && <div className="mono mx-3.5 my-2 text-[11px] text-ink-2">{chainNote}</div>}
            {(assemble.error || master.error) && !chainNote && (
              <div className="errline m-3">{(assemble.error ?? master.error)?.message}</div>
            )}
            <table>
              <tbody>
                <tr>
                  <td className="mono">{bookId}.m4b</td>
                  <td className={`mono ${book.data?.downloads.m4b ? "text-ink-2" : "text-ink-3"}`}>
                    {book.data?.downloads.m4b ? `${(book.data.downloads.m4b.bytes / 1048576).toFixed(1)} MB` : "not yet mastered"}
                  </td>
                  <td>{book.data?.downloads.m4b ? <a className="link" href={book.data.downloads.m4b.url}>download</a> : "—"}</td>
                </tr>
                {book.data?.downloads.chapter_mp3s.map((c) => (
                  <tr key={c.index}>
                    <td className="mono">chapters/ch{String(c.index).padStart(3, "0")}.mp3</td>
                    <td className="mono text-ink-2">{(c.bytes / 1048576).toFixed(1)} MB</td>
                    <td><a className="link" href={c.url}>download</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {book.data && <CoverControl bookId={bookId} cover={book.data.cover} />}
          </div>

          {validation.data && validation.data.validated_segments > 0 && (
            <div className="panel">
              <div className="panel-h">
                <b>Audio checks</b>
                <span className="tag ml-auto">
                  {validation.data.validation_failures} failed of {validation.data.validated_segments} checked
                </span>
              </div>
              <div className="panel-sub">
                Cloned-voice segments are transcribed back by whisper; a mismatch means the voice may have misread.
                Listen before shipping.
              </div>
              <div className="px-3.5 pb-3.5">
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
