import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useSearchParams } from "react-router-dom";

import {
  useBook,
  useBookJobs,
  useBooks,
  useEstimate,
  useKokoroPresets,
  useLexicon,
  useMintQuote,
  useRenderSummary,
  useSettings,
  useStartJob,
  useSystem,
  useValidation,
  useVoices,
} from "../api/hooks";
import type { RenderMode } from "../api/types";
import { useScreenBook } from "../app/currentBook";
import { TalkDialog } from "../components/Dialog";
import { NextStep } from "../components/NextStep";
import { PlainError } from "../components/PlainError";
import { TalkSelect } from "../components/Select";
import { StaleAudioBanner } from "../components/StaleAudio";
import { failureLine, SCREENS } from "../lib/glossary";
import { stepName } from "../lib/jobCopy";
import { explainJobError, unresolvedFailure } from "../lib/jobErrors";
import { classifyRenderFailure, type FlowFailure } from "../lib/money";
import { ADD_A_BOOK, RENDER_ANCHOR } from "../lib/nextStep";
import {
  choiceForVoice,
  choiceVoiceId,
  MODE_LABEL,
  narratorSpec,
  respelledWords,
  wholeBookConfirm,
  type NarratorChoice,
  type WholeBookDetail,
} from "../lib/render";
import { scopeChapters, type Scope } from "../lib/scope";
import { ActiveModeControl } from "./render/ActiveModeControl";
import { AttributePanel } from "./render/AttributePanel";
import { ConfirmFullDialog } from "./render/ConfirmFullDialog";
import { CoverControl } from "./render/CoverControl";
import { EstimateCard } from "./render/EstimateCard";
import { JobRow } from "./render/JobRow";
import { NarratorPicker } from "./render/NarratorPicker";
import { QuoteTicket, type TicketState } from "./render/QuoteTicket";
import { ScopeRow } from "./render/ScopeRow";
import { StageRail } from "./render/StageRail";
import { ValidationFailure } from "./render/ValidationFailure";

type FinishStep = "assemble" | "master";

/** What each finish step makes, in the words the strip uses. */
const MAKES: Record<FinishStep, string> = { assemble: "the chapter MP3s", master: "the .m4b" };

/* -------------------------------------------------- the screen */

export function RenderJobs() {
  const [, setParams] = useSearchParams();
  const books = useBooks();
  // no ?book= (a typed URL or an old bookmark): the book this viewer last worked on, else the first
  const bookId = useScreenBook(
    books.data?.books.map((b) => b.book_id),
    books.data?.books[0]?.book_id,
  );
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

  const jobs = useBookJobs(bookId);
  const validation = useValidation(bookId, !!status?.rendered);
  const summary = useRenderSummary(bookId, !!status?.rendered);
  const renderedSet = useMemo(
    () => new Set(summary.data?.chapters.map((c) => c.index) ?? []),
    [summary.data],
  );

  // One narrator: which voice reads the book, and how fast. Fetched only when they matter
  // (single mode, or naming the voices on failed audio checks).
  const single = mode === "single";
  const hasFailures = (validation.data?.results.length ?? 0) > 0;
  const voices = useVoices({ enabled: single || hasFailures });
  // the words the lexicon already respells are never suggested again on a failed check
  const lexicon = useLexicon(hasFailures ? bookId : null);
  const presets = useKokoroPresets({ enabled: single });
  const settings = useSettings({ enabled: single });
  const [narratorPick, setNarratorPick] = useState<NarratorChoice | null>(null);
  const [speed, setSpeed] = useState(1);
  useEffect(() => {
    setNarratorPick(null);
    setSpeed(1);
  }, [bookId]);
  const presetIds = useMemo(() => presets.data?.voices.map((p) => p.id), [presets.data]);
  // until the user picks, keep the narrator the existing one-narrator recording used, so
  // re-making a chapter doesn't try to mix two narrators (the server refuses that)
  const recordedVoice = summary.data?.mode === "single" ? summary.data.voice_id : null;
  const recordedChoice = choiceForVoice(recordedVoice, voices.data?.voices, presetIds);
  const narrator: NarratorChoice = narratorPick ?? recordedChoice ?? "default";
  const singleSpec = useMemo(
    () => (single ? narratorSpec(narrator, speed, voices.data?.voices) : undefined),
    [single, narrator, speed, voices.data],
  );
  // Until the user picks, the earlier narrator can only be matched once the summary and BOTH lists have
  // answered (it may be a library voice or a preset) — before that "default" would be a
  // guess, so nothing is priced or recorded. A library voice's engine comes from the library:
  // don't price it before that has loaded either.
  const settled = (q: { data?: unknown; isError: boolean }) => q.data !== undefined || q.isError;
  const findingRecorded =
    single &&
    narratorPick === null &&
    // a recorded book: its summary says which narrator it used
    ((!!status?.rendered && !settled(summary)) || (!!recordedVoice && !(settled(voices) && settled(presets))));
  const narratorReady =
    !single || (!findingRecorded && (!narrator.startsWith("voice:") || voices.data !== undefined));
  // both lists answered and the earlier narrator is in neither: say so instead of switching silently
  const recordedLost = single && !!recordedVoice && narratorPick === null && !findingRecorded && !recordedChoice;
  const defaultVoiceId = settings.data?.kokoro_default_voice;
  const defaultName = presets.data?.voices.find((p) => p.id === defaultVoiceId)?.name ?? defaultVoiceId;
  const narratorVoiceId = choiceVoiceId(narrator, defaultVoiceId);
  const narratorClash =
    single && scope.kind === "range" && !!recordedVoice && narratorVoiceId !== null && narratorVoiceId !== recordedVoice;

  const ready =
    !!status?.ingested && narratorReady && (mode === "single" || (status?.attributed && status?.assigned));
  const estimate = useEstimate(bookId, mode, chapters, !!ready, emotionArg, force, singleSpec);
  // mirrors the backend's manifest owners: these job kinds make the mode switch 409
  const activeJob = book.data?.active_job ?? null;
  const manifestOwner =
    activeJob && ["render", "assemble", "master"].includes(activeJob.kind) ? activeJob : null;

  const mint = useMintQuote(bookId ?? "");
  const render = useStartJob(bookId ?? "", "render");
  const assemble = useStartJob(bookId ?? "", "assemble");
  const master = useStartJob(bookId ?? "", "master");

  // One-click finish: build the .m4b (and, when asked, the chapter MP3s first), advancing when
  // the watched job lands (the useBookJobs poll is the clock). Client-side chain — it stops if
  // the page closes, which the labels say out loud. The .m4b needs only the recording, so the
  // MP3s are an option, off by default.
  const [chain, setChain] = useState<{ watch: string | null; queue: FinishStep[] } | null>(null);
  const [chainNote, setChainNote] = useState<string | null>(null);
  const [autoFinish, setAutoFinish] = useState(false);
  const [withMp3s, setWithMp3s] = useState(false);
  const finishQueue = (): FinishStep[] => (withMp3s ? ["assemble", "master"] : ["master"]);
  const chainStep = (queue: FinishStep[]) => {
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
          setChainNote(`stopped before ${MAKES[next]}: ${failureLine(e)}`);
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
      const explained = explainJobError(j.error);
      const why = j.error ? `: ${explained.known ? explained.summary : j.error}` : "";
      setChainNote(`stopped — ${stepName(j.kind)} ${j.state === "failed" ? "failed" : "was stopped"}${why}`);
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
    setWithMp3s(false);
  }, [bookId]);
  const chainBusy = chain !== null || assemble.isPending || master.isPending;

  // A next-step button lands here with #find-speakers / #record / #build: scroll that box into
  // view once the book has loaded (location.key re-fires it when the same link is clicked again).
  const location = useLocation();
  const loaded = book.data !== undefined;
  useEffect(() => {
    const id = location.hash.slice(1);
    if (!id || !loaded) return;
    const box = document.getElementById(id);
    box?.scrollIntoView({ behavior: "smooth", block: "start" });
    // move keyboard focus too, so the next Tab goes into the box the user was sent to
    box?.focus({ preventScroll: true });
  }, [location.hash, location.key, loaded]);

  const title = book.data?.status.title ?? books.data?.books.find((b) => b.book_id === bookId)?.title ?? null;

  const [ticket, setTicket] = useState<TicketState>({ kind: "none" });
  // The one whole-book sheet (free: its own key; paid: the live ticket inside it).
  const [sheet, setSheet] = useState<WholeBookDetail | null>(null);
  const [flowError, setFlowError] = useState<FlowFailure | null>(null);
  // A quote prices exactly one setup. ANY change to what would be recorded — the book, mode,
  // chapters, emotion, a forced re-record, the narrator or speed — leaves a live ticket
  // describing different work: drop it (and any open sheet) so the user gets a fresh price.
  const pricingKey = JSON.stringify([bookId, mode, chapters, emotionArg ?? null, force, singleSpec ?? null]);
  // the setup on screen NOW — a price that lands after a change described the old one
  const pricingNow = useRef(pricingKey);
  useEffect(() => {
    pricingNow.current = pricingKey;
    setTicket({ kind: "none" });
    setSheet(null);
  }, [pricingKey]);

  // The whole-book stop, asked BEFORE the request (limits.full_render_confirm_blocks), so
  // confirm_full goes out with the first POST. Unknown limit -> the server's 409 still asks.
  const wholeBook = wholeBookConfirm(
    chapters,
    book.data?.chapters,
    system.data?.limits?.full_render_confirm_blocks,
    book.data?.runtime_estimate_seconds,
  );

  // Re-make a chapter (from an audio-check row): confirm first, then set up a forced range.
  const [remake, setRemake] = useState<number | null>(null);
  const focusRecord = useRef(false);
  const remakeChapter = (n: number) => {
    setScope({ kind: "range", from: n, to: n });
    setForce(true);
    if (summary.data) setModeChoice(summary.data.active_mode === "single" ? "single" : "multivoice");
    focusRecord.current = true;
    setRemake(null);
  };
  // Move to the recording controls only AFTER the dialog has gone: closing it hands focus back
  // to its trigger (the row's Re-make button), which would undo a move made in the handler.
  useEffect(() => {
    if (remake !== null || !focusRecord.current) return;
    focusRecord.current = false;
    const frame = requestAnimationFrame(() => {
      const box = document.getElementById(RENDER_ANCHOR.record);
      box?.scrollIntoView({ behavior: "smooth", block: "start" });
      box?.focus({ preventScroll: true });
    });
    return () => cancelAnimationFrame(frame);
  }, [remake]);

  const doMint = (opts: { openSheet?: boolean } = {}) => {
    setFlowError(null);
    const priced = pricingKey;
    mint.mutate(
      { mode, chapters, applyEmotion: emotionArg, force, single: singleSpec },
      {
        onSuccess: (quote) => {
          if (priced !== pricingNow.current) return; // a setting changed while pricing: stale
          setTicket({ kind: "live", quote });
          // a paid whole-book recording: the ticket opens inside the one whole-book sheet
          if (opts.openSheet && wholeBook) setSheet(wholeBook);
        },
        onError: (e) => {
          if (priced !== pricingNow.current) return;
          setFlowError({ error: e, lead: "Couldn't get the price" });
        },
      },
    );
  };

  const startRender = (opts: { token?: string; confirmFull?: boolean }) => {
    setFlowError(null);
    render.mutate(
      {
        mode,
        chapters,
        ...(singleSpec ? { single: singleSpec } : {}),
        ...(emotionArg === undefined ? {} : { apply_emotion: emotionArg }),
        ...(force ? { force: true } : {}),
        ...(opts.token ? { cost_token: opts.token } : {}),
        ...(opts.confirmFull ? { confirm_full: true } : {}),
      },
      {
        onSuccess: (job) => {
          setTicket({ kind: "none" });
          setSheet(null);
          if (autoFinish) {
            setChainNote(null);
            setChain({ watch: job.job_id, queue: finishQueue() });
          }
        },
        onError: (e) => {
          const action = classifyRenderFailure(e, ticket.kind === "live");
          switch (action.kind) {
            case "confirm-full":
              // fallback only (the limit wasn't known up front): ask once, same sheet
              setSheet(action.detail);
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
              setFlowError({ error: action.error, lead: "Couldn't start the recording" });
          }
        },
      },
    );
  };

  /** The ticket's own confirm: inside the whole-book sheet it also carries confirm_full. */
  const ticketEl = (inSheet: boolean) => (
    <QuoteTicket
      state={ticket}
      bookTitle={title}
      busy={render.isPending}
      onConfirm={() => {
        if (ticket.kind !== "live") return;
        if (inSheet) startRender({ token: ticket.quote.token, confirmFull: true });
        else if (wholeBook) setSheet(wholeBook);
        else startRender({ token: ticket.quote.token });
      }}
      onRemint={() => doMint()}
      onExpired={() =>
        ticket.kind === "live" &&
        setTicket({ kind: "refused", quote: ticket.quote, stamp: "EXPIRED", message: "the 15-minute window passed" })
      }
    />
  );

  if (books.isPending) return <section className="screen"><div className="loadline">reading the shelf…</div></section>;
  if (books.isError) return <section className="screen"><h1>{SCREENS.render}</h1><PlainError error={books.error} title="couldn't load your books" /></section>;
  if (!bookId || !books.data?.books.length) {
    return (
      <section className="screen">
        <h1>{SCREENS.render}</h1>
        <p className="sub">No books yet. Add one in the {SCREENS.library} first.</p>
        <NextStep step={ADD_A_BOOK} />
      </section>
    );
  }

  const castChanged =
    mode === "multivoice" &&
    !!summary.data?.rendered_assignment_hash &&
    !!estimate.data?.assignment_hash &&
    summary.data.rendered_assignment_hash !== estimate.data.assignment_hash;
  const voiceNames = new Map(voices.data?.voices.map((v) => [v.voice_id, v.name]) ?? []);
  const respelled = respelledWords(lexicon.data?.entries);
  // how much a re-make redoes: the recorded lines of that chapter, else its paragraphs
  const remakeLines = remake !== null ? summary.data?.chapters.find((c) => c.index === remake)?.segments : undefined;
  const remakeBlocks = remake !== null ? book.data?.chapters?.find((c) => c.index === remake)?.speakable_blocks : undefined;
  const remakeSize =
    remakeLines !== undefined ? `${remakeLines.toLocaleString()} lines` : remakeBlocks !== undefined ? `${remakeBlocks.toLocaleString()} paragraphs` : null;

  return (
    <section className="screen">
      <h1 className="flex items-baseline gap-3.5">
        {SCREENS.render}
        <TalkSelect
          className="bookpick"
          ariaLabel="book"
          value={bookId}
          onChange={(v) => setParams({ book: v })}
          options={books.data.books.map((b) => ({ value: b.book_id, label: b.title ?? b.book_id }))}
        />
      </h1>
      <p className="sub">
        Free recordings start with one click; paid voices always show their exact price first.
      </p>
      {book.isError && <PlainError error={book.error} title="couldn't load this book" />}
      <StageRail
        status={status}
        activeJob={book.data?.active_job ?? null}
        failed={unresolvedFailure(jobs.data?.jobs, bookId)}
      />
      <StaleAudioBanner status={status} />
      <div className="rjgrid">
        <div>
          {/* step 1 of the full-cast path: a step card, done or not, before the recording controls */}
          {status?.ingested && (
            <AttributePanel
              key={bookId}
              id={RENDER_ANCHOR.findSpeakers}
              book={{ book_id: bookId, attributed: status.attributed, active_job: activeJob }}
              chapterCount={chapterCount}
              chapters={chapters}
            />
          )}
          <div className="modewrap" id={RENDER_ANCHOR.record} tabIndex={-1} role="group" aria-label="how it sounds">
            <span className="tag">version</span>
            <button className={`chap ${mode === "multivoice" ? "on" : ""}`} disabled={!status?.assigned}
              title={status?.assigned ? "every character gets their own voice" : `cast the voices first (${SCREENS.review})`}
              onClick={() => setModeChoice("multivoice")}>
              {MODE_LABEL.multivoice}
            </button>
            <button className={`chap ${mode === "single" ? "on" : ""}`} title="one voice reads the whole book"
              onClick={() => setModeChoice("single")}>
              {MODE_LABEL.single}
            </button>
          </div>
          {!ready && status && narratorReady && (
            <div className="refusal mb-4">
              <span className="tag">not ready yet</span>
              <p>
                {!status.ingested
                  ? "this book has no readable text yet — add it again in the Library"
                  : `${MODE_LABEL.multivoice} needs the speakers found and the voices cast first — or switch to ${MODE_LABEL.single}`}
              </p>
            </div>
          )}
          {status?.assigned && mode === "single" && (
            <div className="refusal mb-3.5">
              <span className="tag">cast not used</span>
              <p>
                this book HAS a cast, but {MODE_LABEL.single} reads everything in one voice —{" "}
                <button className="link" onClick={() => setModeChoice("multivoice")}>
                  switch to {MODE_LABEL.multivoice}
                </button>
              </p>
            </div>
          )}
          {single && (
            <NarratorPicker
              choice={narrator}
              setChoice={setNarratorPick}
              speed={speed}
              setSpeed={setSpeed}
              voices={voices.data?.voices}
              presets={presets.data?.voices}
              defaultName={defaultName}
            />
          )}
          {findingRecorded && <div className="loadline">finding the narrator of the earlier recording…</div>}
          {recordedLost && !narratorClash && (
            <div className="refusal mb-3.5">
              <span className="tag">narrator not found</span>
              <p>
                the earlier recording&apos;s narrator isn&apos;t in your voices any more, so the standard narrator is
                picked. Pick another above, or record the whole book so every chapter matches.
              </p>
            </div>
          )}
          {book.data?.chapters && (
            <ScopeRow scope={scope} setScope={setScope} chapters={book.data.chapters} renderedSet={renderedSet} />
          )}
          {narratorClash && (
            <div className="refusal mb-3.5">
              <span className="tag">narrator changed</span>
              <p>
                the other chapters were recorded with a different narrator, and a chapter range can&apos;t mix
                two narrators — the recording would stop. Pick the same narrator, or{" "}
                <button className="link" onClick={() => setScope({ kind: "whole" })}>
                  record the whole book instead
                </button>
              </p>
            </div>
          )}
          {status?.rendered && (
            <div className="scoperow">
              <span className="tag">re-record</span>
              <label
                className="mono flex items-center gap-1.5 text-xs"
                title="Record the chosen chapters again from scratch, replacing the saved audio. Use it to redo a bad chapter (a misread the audio check missed) — pick a chapter range above first. Paid voices bill again, and the estimate shows that."
              >
                <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
                record again from scratch — ignore saved audio
              </label>
              <span className="mono scopehint">
                {force
                  ? "every line in scope is recorded fresh — paid voices bill again"
                  : "saved audio is reused — nothing already recorded is redone"}
              </span>
            </div>
          )}
          {castChanged && (
            <div className="refusal mb-3.5">
              <span className="tag">cast changed</span>
              {scope.kind === "range" ? (
                <p>
                  the cast changed since the last recording. A chapter range can&apos;t use the new cast — the
                  chapters outside it still have the old voices, so the recording would stop.{" "}
                  <button className="link" onClick={() => setScope({ kind: "whole" })}>
                    record the whole book instead
                  </button>{" "}
                  — only the recast characters&apos; lines are recorded again; everything else is reused.
                </p>
              ) : (
                <p>
                  the current cast differs from the recorded audio — record the whole book again to hear the new
                  voices. Only the recast characters&apos; lines are recorded again; every unchanged voice is reused,
                  so this is fast and mostly free.
                </p>
              )}
            </div>
          )}
          {mode === "multivoice" && (
            <div className="scoperow">
              <span className="tag">emotion</span>
              <label
                className="mono flex items-center gap-1.5 text-xs"
                title="voice each dialogue line with the emotion speaker-finding tagged it. Off keeps delivery flat and the saved audio reusable."
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
                  : "flat delivery — emotion tags ignored"}
                {emotionChoice === null && ` · default: ${emotionDefault ? "on" : "off"}`}
              </span>
              <span className="mono scopehint basis-full">
                Kokoro voices always read flat — emotion only changes cloned (Chatterbox, IndexTTS-2) and ElevenLabs
                voices.
              </span>
            </div>
          )}
          {estimate.isPending && ready && <div className="loadline">estimating against the saved audio…</div>}
          {estimate.isError && <PlainError error={estimate.error} title="couldn't estimate" />}
          {estimate.data && (
            <>
              <EstimateCard
                est={estimate.data}
                mode={mode}
                onMint={() => doMint({ openSheet: true })}
                onFreeRender={() => (wholeBook ? setSheet(wholeBook) : startRender({}))}
                minting={mint.isPending}
                error={flowError}
              />
              <label className="mono mt-2 flex items-center gap-1.5 text-[11px] text-ink-2">
                <input type="checkbox" checked={autoFinish} onChange={(e) => setAutoFinish(e.target.checked)} />
                when the recording lands, build the .m4b automatically (keep this page open)
              </label>
            </>
          )}
          {/* inline only while the whole-book sheet is closed — never two live confirm keys */}
          {!sheet && ticketEl(false)}
          <details className="fine">
            <summary>what if the price changes or the ticket expires?</summary>
            You never get double-billed: the ticket is single-use and bound to the exact work you saw. If anything
            shifts before you confirm — a setting here, the saved audio, an edit, the clock — the ticket is cleared
            or stamped and its confirm key deadens; you simply get a fresh price.
          </details>
        </div>

        <div>
          <div className="panel joblist">
            <div className="panel-h"><b>Jobs</b><span className="tag ml-auto">live</span></div>
            <div className="panel-sub">Newest first — click a failed row for its reason.</div>
            {jobs.data?.jobs.length === 0 && <div className="loadline p-3.5">no jobs yet</div>}
            {jobs.data?.jobs.map((j) => <JobRow key={j.job_id} job={j} />)}
          </div>

          <div className="panel dl" id={RENDER_ANCHOR.build} tabIndex={-1}>
            <div className="panel-h">
              <b>Outputs</b>
              <button
                className="key ml-auto"
                disabled={!status?.rendered || chainBusy}
                title="join the recorded chapters into one audiobook file"
                onClick={() => {
                  setChainNote(null);
                  chainStep(finishQueue());
                }}
              >
                {chain ? "building…" : "Build the .m4b"}
              </button>
            </div>
            <div className="panel-sub">
              The .m4b is the audiobook: every recorded chapter in one file, with chapter marks and the cover.
            </div>
            <label className="mono mx-3.5 mb-2 flex items-center gap-1.5 text-[11px] text-ink-2">
              <input type="checkbox" checked={withMp3s} onChange={(e) => setWithMp3s(e.target.checked)} />
              also make chapter MP3s (optional)
            </label>
            {chain && (
              <div className="drainstrip mx-3 my-2">
                <span className="state"><i className="led run" />building</span>
                <span className="mono text-[11px] text-ink-2">
                  {chain.watch ? "waiting for the current step" : "starting"}
                  {chain.queue.length > 0 && ` · then ${chain.queue.map((q) => MAKES[q]).join(" → ")}`} — keep this
                  page open
                </span>
              </div>
            )}
            {chainNote && <div className="mono mx-3.5 my-2 text-[11px] text-ink-2">{chainNote}</div>}
            {(assemble.error || master.error) && !chainNote && (
              <PlainError className="m-3" error={assemble.error ?? master.error} title="couldn't start" />
            )}
            {/* the version Listen plays (and the .m4b is built from), beside its downloads */}
            {summary.data && (
              <ActiveModeControl bookId={bookId} summary={summary.data} conflict={manifestOwner} />
            )}
            <table>
              <tbody>
                <tr>
                  <td className="mono">{title ? `${title}.m4b` : "Audiobook (.m4b)"}</td>
                  <td className={`mono ${book.data?.downloads.m4b ? "text-ink-2" : "text-ink-3"}`}>
                    {book.data?.downloads.m4b ? `${(book.data.downloads.m4b.bytes / 1048576).toFixed(1)} MB` : "not built yet"}
                  </td>
                  <td>{book.data?.downloads.m4b ? <a className="link" href={book.data.downloads.m4b.url}>download</a> : "—"}</td>
                </tr>
                {book.data?.downloads.chapter_mp3s.map((c) => (
                  <tr key={c.index}>
                    <td className="mono">Chapter {c.index} MP3</td>
                    <td className="mono text-ink-2">{(c.bytes / 1048576).toFixed(1)} MB</td>
                    <td><a className="link" href={c.url}>download</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mx-3.5 my-2">
              <button
                className="key quiet px-2 py-[2px] text-[11px]"
                disabled={!status?.rendered || chainBusy}
                title="only the per-chapter MP3 files, without building the .m4b"
                onClick={() => assemble.mutate({})}
              >
                chapter MP3s only
              </button>
            </div>
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
                Lines read by cloned voices are listened back and compared with the book; a mismatch means the voice
                may have misread. Listen before you share the audiobook.
              </div>
              <div className="px-3.5 pb-3.5">
                {validation.data.results.length === 0 && <div className="loadline">every checked line read clean</div>}
                {validation.data.results.map((r) => (
                  <ValidationFailure
                    key={`${r.block_id}-${r.segment_index}`}
                    bookId={bookId}
                    row={r}
                    voiceName={r.voice_id ? voiceNames.get(r.voice_id) : undefined}
                    onRemake={setRemake}
                    respelled={respelled}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      {sheet && (
        <ConfirmFullDialog
          detail={sheet}
          // paid comes from the estimate, not from whether a ticket exists: while an expired
          // ticket is re-priced (or that failed) the free key must never appear
          paid={(estimate.data?.total_usd ?? 0) > 0 || ticket.kind !== "none"}
          ticket={ticket.kind === "none" ? undefined : ticketEl(true)}
          minting={mint.isPending}
          busy={render.isPending}
          error={flowError}
          onConfirm={() => startRender({ confirmFull: true })}
          onRemint={() => doMint()}
          onCancel={() => setSheet(null)}
        />
      )}
      {remake !== null && (
        <TalkDialog
          title={`Re-make chapter ${remake}`}
          onClose={() => setRemake(null)}
          footer={
            <>
              <button className="key quiet" onClick={() => setRemake(null)}>not now</button>
              <button className="key" onClick={() => remakeChapter(remake)}>set up the re-make</button>
            </>
          }
        >
          <p className="m-0 text-ink-2">
            Chapter {remake}
            {remakeSize && (
              <>
                {" "}(<b className="mono">{remakeSize}</b>)
              </>
            )}{" "}
            is recorded again from scratch, replacing its saved audio.
          </p>
          <p className="mb-0 mt-2 text-caution">
            Lines read by a paid voice are billed again. Nothing records yet — you&apos;ll see the estimate and, for
            paid voices, the exact price before you start.
          </p>
        </TalkDialog>
      )}
    </section>
  );
}
