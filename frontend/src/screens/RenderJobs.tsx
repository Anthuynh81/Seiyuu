import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  useAttribute,
  useBook,
  useBookJobs,
  useBooks,
  useEstimate,
  useMintQuote,
  useRenderSummary,
  useStartJob,
  useSystem,
  useValidation,
} from "../api/hooks";
import type { RenderMode } from "../api/types";
import { TalkSelect } from "../components/Select";
import { StaleAudioBanner } from "../components/StaleAudio";
import { classifyRenderFailure } from "../lib/money";
import { scopeChapters, type Scope } from "../lib/scope";
import { ActiveModeControl } from "./render/ActiveModeControl";
import { ConfirmFullDialog } from "./render/ConfirmFullDialog";
import { CoverControl } from "./render/CoverControl";
import { EstimateCard } from "./render/EstimateCard";
import { JobRow } from "./render/JobRow";
import { QuoteTicket, type TicketState } from "./render/QuoteTicket";
import { ScopeRow } from "./render/ScopeRow";
import { StageRail } from "./render/StageRail";
import { ValidationFailure } from "./render/ValidationFailure";

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
