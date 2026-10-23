import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  segmentsQuery,
  useAssignment,
  useBook,
  useBooks,
  useChapterAttribution,
  useCharacters,
  useDraftAssignment,
  useEditLog,
  useRecordEdit,
  useSaveAssignment,
  useSegments,
  useSuggestCast,
  useVoices,
  type SuggestInput,
} from "../api/hooks";
import type { CharacterSummary, SegmentRow, SuggestCastResponse } from "../api/types";
import { chapterOfBlock } from "../api/types";
import { useScreenBook } from "../app/currentBook";
import { usePlayer } from "../app/usePlayer";
import { NextStep } from "../components/NextStep";
import { PlainError, PlainErrorLine } from "../components/PlainError";
import { TalkSelect } from "../components/Select";
import { Tip } from "../components/Tooltip";
import { auditionLine } from "../lib/audition";
import {
  castingDiffers,
  castingFromServer,
  castingVoicesDiffer,
  genderLabel,
  type CastingState,
} from "../lib/casting";
import { buildEmotionMap, emotionKey } from "../lib/emotion";
import { cliHint, SCREENS, term } from "../lib/glossary";
import { nextStep } from "../lib/nextStep";
import { findLine, matchesSpeaker, readLocation, walkToNextCheck } from "../lib/reviewNav";
import { canPlay, describeEditWarning, mergeSuggestions, type LineTarget } from "../lib/reviewTools";
import { CharacterEditor } from "./review/CharacterEditor";
import { isReviewable, useFrontier, useSpoilerSafe } from "./review/helpers";
import { HistoryDrawer } from "./review/HistoryDrawer";
import { MergeSuggestions } from "./review/MergeSuggestions";
import { EmotionPopover, ReassignPopover } from "./review/popovers";
import { ResetCastDialog } from "./review/ResetCastDialog";
import { RosterRow } from "./review/RosterRow";
import { AttributePanel } from "./render/AttributePanel";
import { SegmentPair } from "./review/SegmentPair";
import { SuggestCastPanel } from "./review/SuggestCastPanel";
import { PlayableRow } from "./review/PlayableRow";
import { ViewMenu } from "./review/ViewMenu";
import { VoicePicker } from "./review/VoicePicker";
import { AddVoiceDialog } from "./voices/AddVoiceDialog";

/** "Next to check" progress: idle, walking chapters, or stopped with a plain message. */
type Walk =
  | { state: "idle" }
  | { state: "looking"; chapter: number }
  | { state: "end"; fromStart: boolean }
  | { state: "error"; message: string }
  /** landed on a line: focus stays on the button, the polite region says where it went */
  | { state: "landed"; message: string };
const IDLE: Walk = { state: "idle" };

/* -------------------------------------------------- the screen */

/** The shared player's title suffix for Review's one-line playlist (▶ on a line). */
const ONE_LINE = " · one line";
const isOneLine = (p: { bookId: string | null; clips: readonly unknown[]; chapterTitle: string }, bookId: string | null) =>
  p.bookId === bookId && p.clips.length === 1 && p.chapterTitle.endsWith(ONE_LINE);

export function Review() {
  const [params, setParams] = useSearchParams();
  const qc = useQueryClient();
  const books = useBooks();
  // no ?book=: the book this viewer last worked on, else the first one with speakers found
  const bookId = useScreenBook(
    books.data?.books.map((b) => b.book_id),
    books.data?.books.find((b) => b.attributed)?.book_id,
  );
  const book = useBook(bookId);
  const attributed = !!book.data?.status.attributed;

  // Where the page is lives in the URL (?ch=, ?block=&seg=, ?speaker=), so a line can be linked
  // to and a reload keeps the place. Moves within the screen REPLACE the history entry.
  const loc = useMemo(() => readLocation(params), [params]);
  const chapter = loc.chapter;
  const speakerFilter = loc.speaker;
  const target = loc.target;
  const patchParams = useCallback(
    (patch: Record<string, string | null>) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [k, v] of Object.entries(patch)) {
            if (v === null) next.delete(k);
            else next.set(k, v);
          }
          return next;
        },
        { replace: true },
      ),
    [setParams],
  );
  const chapterCount = book.data?.chapters?.length ?? 1;
  const chapterList = useMemo(
    () => book.data?.chapters?.map((c) => c.index) ?? Array.from({ length: chapterCount }, (_, i) => i + 1),
    [book.data, chapterCount],
  );

  const overview = useCharacters(bookId, attributed);
  const segments = useSegments(bookId, chapter, attributed);
  // "show only <Name>": the server's ?speaker= view of the same chapter (the full chapter
  // above still feeds the queue counts, the emotion tags and the narrator's sample line)
  const filtered = useSegments(bookId, chapter, attributed && speakerFilter !== null, {
    speaker: speakerFilter,
  });
  const shown = speakerFilter !== null ? filtered : segments;
  // F2/F2a: the per-segment emotion tags the model captured, now EDITABLE via the set_emotion
  // edit overlay (EmotionPopover). Keyed block_id:ordinal to line up with each row.
  const attribution = useChapterAttribution(bookId, chapter, attributed);
  const emotionMap = useMemo(
    () => buildEmotionMap(attribution.data?.report.chapters.find((c) => c.index === chapter)),
    [attribution.data, chapter],
  );
  const editLog = useEditLog(attributed ? bookId : null);
  const [historyOpen, setHistoryOpen] = useState(false);
  useEffect(() => setHistoryOpen(false), [bookId]);

  // casting: server truth + a local editable copy
  const assignment = useAssignment(bookId, attributed);
  const voicesQ = useVoices();
  const draftCast = useDraftAssignment(bookId ?? "");
  const suggestCast = useSuggestCast(bookId ?? "");
  const saveCast = useSaveAssignment(bookId ?? "");
  const [casting, setCasting] = useState<CastingState | null>(null);
  useEffect(() => {
    const a = assignment.data;
    setCasting(a ? castingFromServer(a) : null);
  }, [assignment.data]);
  const castingDirty = useMemo(
    () => (assignment.data && casting ? castingDiffers(assignment.data, casting) : false),
    [assignment.data, casting],
  );
  // A voice-changing save on a rendered book makes the audio stale. Tracked (not derived from
  // saveCast.isSuccess) so a stage-only draft->final save never claims the audio is stale, and
  // set-true-only so a later stage-only save can't clear a still-pending voice change. Cleared on
  // book switch (the ?book= param), since staleness is per-book.
  const [staleAfterSave, setStaleAfterSave] = useState(false);
  useEffect(() => setStaleAfterSave(false), [bookId]);
  // the "reset cast…" confirm (a keep_existing:false draft) — the only path that drops hand picks
  const [resetOpen, setResetOpen] = useState(false);
  useEffect(() => setResetOpen(false), [bookId]);
  // "Suggest a cast": the last preview the server returned. Held here (not read off the
  // mutation) so the table stays up while a row/narrator/accent change re-requests it; the
  // per-call onSuccess only fires for the LATEST request, so an out-of-order reply never wins.
  const [castPreview, setCastPreview] = useState<SuggestCastResponse | null>(null);
  const resetSuggest = suggestCast.reset;
  useEffect(() => {
    setCastPreview(null);
    resetSuggest(); // a late reply for the previous book must not open its preview here
  }, [bookId, resetSuggest]);
  // a preview/apply repeats the saved stage, so it never quietly demotes a "final" cast
  const savedStage = assignment.data?.stage ?? "draft";
  // `done` (the panel's row/narrator/accent changes) hears the outcome so a refused change can
  // be rolled back; like onSuccess it only fires for the latest request.
  const requestPreview = (input: SuggestInput, done?: (error: Error | null) => void) =>
    suggestCast.mutate(
      { ...input, stage: savedStage },
      {
        onSuccess: (d) => {
          setCastPreview(d);
          done?.(null);
        },
        onError: (e) => done?.(e),
      },
    );
  const openSuggest = () => {
    setCastPreview(null); // a fresh suggestion: the panel remounts with no leftover changes
    draftCast.reset();
    requestPreview({});
  };
  const closeSuggest = () => {
    setCastPreview(null);
    suggestCast.reset();
  };

  // tag filter for the voice pickers: AND semantics, narrows every picker in the roster.
  // The assigned voice always stays visible (VoicePicker falls back to the full pool).
  const [tagFilter, setTagFilter] = useState<string[]>([]);
  const allVoices = useMemo(() => voicesQ.data?.voices ?? [], [voicesQ.data]);
  const allTags = useMemo(
    () => [...new Set(allVoices.flatMap((v) => v.tags))].sort(),
    [allVoices],
  );
  const filteredVoices = useMemo(
    () => allVoices.filter((v) => tagFilter.every((t) => v.tags.includes(t))),
    [allVoices, tagFilter],
  );
  const voiceById = useMemo(() => new Map(allVoices.map((v) => [v.voice_id, v])), [allVoices]);
  // "+ new voice for <Name>": the Add voice dialog opens OVER this screen, so the unsaved
  // casting in `casting` survives; the new voice is cast for that character (still unsaved).
  const [newVoiceFor, setNewVoiceFor] = useState<CharacterSummary | null>(null);
  useEffect(() => setNewVoiceFor(null), [bookId]);
  const toggleTag = (t: string) =>
    setTagFilter((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));

  const [frontier, setFrontier] = useFrontier(bookId);
  const [spoilerSafe, setSpoilerSafe] = useSpoilerSafe();
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const [editingChar, setEditingChar] = useState<CharacterSummary | null>(null);
  const [popoverAt, setPopoverAt] = useState<string | null>(null); // `${block_id}:${idx}`
  const [emoAt, setEmoAt] = useState<string | null>(null); // emotion editor, same key shape

  const pageRef = useRef<HTMLDivElement>(null);

  // Front matter (title page, copyright) is often chapter 1 and never attributed —
  // skip forward once instead of opening the screen on a 404.
  const autoSkipped = useRef(false);
  // The one-shot front-matter skip is per-book state (a book switch drops ?ch= with ?book=).
  // Only an implicit chapter 1 skips: a chapter or line someone asked for stays put.
  useEffect(() => {
    autoSkipped.current = false;
  }, [bookId]);
  const implicitChapter = !loc.explicitChapter && target === null;
  useEffect(() => {
    if (
      !autoSkipped.current &&
      implicitChapter &&
      chapter === 1 &&
      segments.error instanceof ApiError &&
      segments.error.code === "not_found" &&
      chapterCount > 1
    ) {
      autoSkipped.current = true;
      patchParams({ ch: "2" });
    }
  }, [segments.error, chapter, chapterCount, implicitChapter, patchParams]);

  const threshold = overview.data?.confidence_threshold ?? 0.7;
  const lowConfInChapter = useMemo(
    () => segments.data?.segments.filter((s) => isReviewable(s, threshold)) ?? [],
    [segments.data, threshold],
  );

  const isMasked = (c: CharacterSummary) => {
    if (!spoilerSafe || revealed.has(c.id)) return false;
    const debut = chapterOfBlock(c.first_appearance);
    return debut !== null && debut > frontier;
  };
  const cast = useMemo(() => overview.data?.characters ?? [], [overview.data]);
  // Set-ified isMasked for the segment loop: the per-row `cast.some(...)` was
  // O(segments × cast) on every render, and this screen re-renders on each live-jobs tick.
  // (Hooks, so they live ABOVE the isPending/no-book early returns.)
  const maskedIds = useMemo(() => {
    const ids = new Set<string>();
    if (!spoilerSafe) return ids;
    for (const c of cast) {
      if (revealed.has(c.id)) continue;
      const debut = chapterOfBlock(c.first_appearance);
      if (debut !== null && debut > frontier) ids.add(c.id);
    }
    return ids;
  }, [cast, spoilerSafe, revealed, frontier]);
  // Stable per-key handlers so the memoized SegmentPair rows skip re-rendering when
  // nothing about THEM changed (fresh closures per row would defeat the memo).
  const handleChip = useCallback((key: string) => {
    setEmoAt(null);
    setPopoverAt((cur) => (cur === key ? null : key));
  }, []);
  const handleEmotion = useCallback((key: string) => {
    setPopoverAt(null);
    setEmoAt((cur) => (cur === key ? null : key));
  }, []);

  // "Looks right ✓": a same-speaker reassign. The server records it at confidence 1.0, so the
  // line leaves the queue on the refetch. One shared mutation: `variables` says which line, so
  // every line's button waits while one saves (a refusal always lands under its own line).
  const confirmLine = useRecordEdit(bookId ?? "");
  const { mutate: recordConfirm, reset: resetConfirm } = confirmLine;
  useEffect(() => resetConfirm(), [bookId, resetConfirm]);
  // the button leaves with the line's "to check" state: keyboard focus moves to that line's
  // speaker chip instead of falling to the page
  const [focusChipOf, setFocusChipOf] = useState<string | null>(null);
  useEffect(() => {
    if (!focusChipOf) return;
    const line = [...(pageRef.current?.querySelectorAll<HTMLElement>("[data-seg]") ?? [])].find(
      (n) => n.dataset.seg === focusChipOf,
    );
    line?.nextElementSibling?.querySelector<HTMLElement>(".chip")?.focus();
    setFocusChipOf(null);
  }, [focusChipOf]);
  const handleConfirm = useCallback(
    (row: SegmentRow) =>
      recordConfirm(
        {
          op: "reassign",
          block_id: row.block_id,
          segment_index: row.segment_index,
          speaker: row.speaker,
        },
        { onSuccess: () => setFocusChipOf(`${row.block_id}:${row.segment_index}`) },
      ),
    [recordConfirm],
  );
  const confirmVars = confirmLine.variables;
  const confirmKey =
    confirmVars && confirmVars.op === "reassign" ? `${confirmVars.block_id}:${confirmVars.segment_index}` : null;
  const confirmError = confirmLine.error;
  const confirmNote = useMemo(() => {
    if (!confirmError) return null;
    return (
      <PlainErrorLine className="linenote" error={confirmError} fallback="Couldn't save that" />
    );
  }, [confirmError]);

  // ▶ on a recorded line: the shared player plays that one wav (file serving, no GPU). The
  // player object changes as it plays, so the stable handler reads it through a ref.
  const player = usePlayer();
  const playerRef = useRef(player);
  useEffect(() => {
    playerRef.current = player;
  });
  const chapterTitle = segments.data?.title || `Chapter ${chapter}`;
  // ▶ pauses and replays only Review's own one-line playlist, never, say, a Listen chapter
  // that happens to sit on the same line (that would run on through the chapter)
  const playingKey =
    player && player.playing && isOneLine(player, bookId) ? (player.clips[0]?.key ?? null) : null;
  // spoiler-safe: the transport bar shows the clip's speaker, so a hidden one stays hidden there
  const maskedRef = useRef<Set<string>>(new Set());
  const handlePlay = useCallback(
    (row: SegmentRow) => {
      const p = playerRef.current;
      if (!p || !bookId || row.audio_segment === null) return;
      const key = `${row.block_id}:${row.audio_segment}`;
      const buster = row.audio_key ? `&v=${row.audio_key}` : "";
      const src = `/api/books/${bookId}/segments/${row.block_id}/audio?segment=${row.audio_segment}${buster}`;
      if (isOneLine(p, bookId) && p.clips[0].src === src) {
        if (p.playing) p.toggle(); // the same line again pauses it
        else p.seekClip(0, 0); // ...and once more plays it from the top
        return;
      }
      const speaker =
        row.speaker === null
          ? "narrator"
          : maskedRef.current.has(row.speaker)
            ? "a hidden character"
            : (row.speaker_name ?? "a character");
      p.load(
        bookId,
        `${chapterTitle}${ONE_LINE}`,
        [
          {
            src,
            duration: row.duration_seconds ?? 0,
            key,
            speaker,
            words: [],
          },
        ],
        { autoplay: true },
      );
    },
    [bookId, chapterTitle],
  );

  useEffect(() => {
    maskedRef.current = maskedIds;
  }, [maskedIds]);

  // spoiler-safe: the reassign list offers only the characters met so far (the merge list in
  // CharacterEditor filters the same way)
  const visibleCast = useMemo(() => cast.filter((c) => !maskedIds.has(c.id)), [cast, maskedIds]);

  /* ---- the linked line: scroll to it and move focus there, once per arrival */
  const shownRows = shown.data?.segments;
  const targetRow = useMemo(
    () => (target && shownRows ? findLine(shownRows, target) : undefined),
    [target, shownRows],
  );
  const targetKey = targetRow ? `${targetRow.block_id}:${targetRow.segment_index}` : null;
  const landedOn = useRef<string | null>(null);
  // bumped by every "Go to line": arriving on the line the page is already on still scrolls
  // there and moves focus (after the one-character filter is cleared, say)
  const [goNonce, setGoNonce] = useState(0);
  // A landing "Next to check" asked for: scroll there but leave focus on the button, so the
  // press-check-press loop never needs a long Shift+Tab back past the roster.
  const quietArrival = useRef<string | null>(null);
  useEffect(() => {
    if (!targetKey) {
      landedOn.current = null; // left the line: arriving on it again later scrolls again
      return;
    }
    const arrival = `${bookId}|${chapter}|${targetKey}|${speakerFilter}|${goNonce}`;
    // a live refetch (job ticks, an edit) re-renders the page; only a NEW target scrolls
    if (landedOn.current === arrival) return;
    const el = [...(pageRef.current?.querySelectorAll<HTMLElement>("[data-seg]") ?? [])].find(
      (n) => n.dataset.seg === targetKey,
    );
    if (!el) return;
    landedOn.current = arrival;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    // only a real deep-link arrival (a first load, a link from another screen) moves focus
    if (quietArrival.current !== arrival) el.focus({ preventScroll: true });
    quietArrival.current = null;
  }, [targetKey, bookId, chapter, speakerFilter, goNonce]);

  /* ---- Next to check: forward through the lines worth a look, across chapters */
  const [walk, setWalk] = useState<Walk>(IDLE);
  const walkRun = useRef(0);
  const nextRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    walkRun.current += 1; // abandons an in-flight walk for the old book / filter
    setWalk(IDLE);
  }, [bookId, speakerFilter]);
  const nextToCheck = async (fromStart = false) => {
    if (!bookId) return;
    const run = ++walkRun.current;
    const from = fromStart ? { chapter: chapterList[0] ?? 1, after: null } : { chapter, after: target };
    setWalk({ state: "looking", chapter: from.chapter });
    const result = await walkToNextCheck({
      chapters: chapterList,
      from,
      // the open chapter is already loaded in full; later ones are fetched lazily, and only
      // their ?low_confidence= lines
      startRows: from.chapter === chapter ? (segments.data?.segments ?? null) : null,
      keep: (s: SegmentRow) => isReviewable(s, threshold) && matchesSpeaker(s, speakerFilter),
      laterRows: async (ch) => {
        try {
          // retry: false. The app-wide default (retry: 1) would ask twice for every front-matter
          // 404 and every failure; a refusal stops the walk and is never re-asked on its own.
          const page = await qc.fetchQuery({
            ...segmentsQuery(bookId, ch, { speaker: speakerFilter, lowConfidence: true }),
            retry: false,
          });
          return page.segments;
        } catch (e) {
          if (e instanceof ApiError && e.code === "not_found") return null; // nothing to read there
          throw e;
        }
      },
      cancelled: () => walkRun.current !== run,
      onChapter: (ch) => setWalk({ state: "looking", chapter: ch }),
    });
    if (walkRun.current !== run || result.kind === "cancelled") return;
    if (result.kind === "found") {
      const excerpt = result.row.text.replace(/\s+/g, " ").trim();
      setWalk({
        state: "landed",
        message: `Next line to check, chapter ${result.chapter}: ${
          excerpt.length > 80 ? `${excerpt.slice(0, 79)}…` : excerpt
        }`,
      });
      quietArrival.current = `${bookId}|${result.chapter}|${result.row.block_id}:${result.row.segment_index}|${speakerFilter}|${goNonce}`;
      patchParams({
        ch: String(result.chapter),
        block: result.row.block_id,
        seg: String(result.row.segment_index),
      });
    } else if (result.kind === "end") {
      setWalk({ state: "end", fromStart });
    } else {
      setWalk({ state: "error", message: `Couldn't read chapter ${result.chapter}: ${result.message}` });
    }
  };
  const setChapter = (n: number) => {
    walkRun.current += 1;
    setWalk(IDLE);
    patchParams({ ch: String(n), block: null, seg: null });
  };
  /** "Go to line" (flagged paragraphs, edit warnings, History): open the line's chapter with
      the one-character filter off, scroll to it and move focus there. Asking again for the line
      the page is already on scrolls back to it (the nonce makes every press a new arrival). */
  const goToLine = (t: LineTarget) => {
    walkRun.current += 1;
    setWalk(IDLE);
    quietArrival.current = null;
    setGoNonce((n) => n + 1);
    patchParams({ ch: String(t.chapter), block: t.block, seg: t.seg === null ? null : String(t.seg), speaker: null });
  };

  if (books.isPending) return <section className="screen"><div className="loadline">reading the shelf…</div></section>;
  if (books.isError) return <section className="screen"><PlainError error={books.error} title="couldn't load your books" /></section>;
  if (!bookId) {
    // no book has its speakers found yet: point at the first one that could
    const candidate = books.data?.books.find((b) => b.ingested && !b.attributed) ?? books.data?.books[0];
    return (
      <section className="screen">
        <h1>{SCREENS.review}</h1>
        <p className="sub">
          {candidate ? (
            <>
              No book has its speakers found yet. Once Seiyuu has worked out who says each line, you can check it
              here.
              {candidate.ingested && (
                <>
                  {" "}
                  Next up: <b>{candidate.title ?? "your book"}</b>.
                </>
              )}
            </>
          ) : (
            "No books yet. Add one in the Library first."
          )}
        </p>
        {/* this screen is about speakers: offer that step even for a book recorded in one voice,
            and do it right here when the book has text to read */}
        {candidate?.ingested ? (
          <AttributePanel key={candidate.book_id} book={candidate} />
        ) : (
          <NextStep step={nextStep({ status: candidate, activeJob: candidate?.active_job, want: "attribute" })} />
        )}
      </section>
    );
  }

  const filterChar = speakerFilter !== null ? cast.find((c) => c.id === speakerFilter) : undefined;
  const filterName =
    speakerFilter === "narration"
      ? "narration"
      : filterChar
        ? maskedIds.has(filterChar.id)
          ? "a hidden character"
          : filterChar.name
        : "one character";
  const filterSuffix = speakerFilter !== null ? ` for ${filterName}` : "";
  // aria-disabled (still focusable), not `disabled`: a keyboard press must not push focus to
  // <body> while the walk looks ahead or the landed chapter loads
  const nextBusy = walk.state === "looking" || segments.isPending;
  // nothing in the whole book needs a look: the walk would only report the end. The server's
  // low-confidence count already holds the unattributed quotes still to check; its unattributed
  // count also keeps the ones confirmed as narration, so it can't decide this.
  const nothingToCheck =
    !!overview.data &&
    overview.data.low_confidence_segments === 0 &&
    lowConfInChapter.length === 0;
  const chapterOptions = (() => {
    const opts = (book.data?.chapters ?? []).map((c) => ({
      value: String(c.index),
      label: c.title ? `Ch ${c.index} · ${c.title}` : `Chapter ${c.index}`,
    }));
    if (opts.length === 0) {
      for (let i = 1; i <= chapterCount; i++) opts.push({ value: String(i), label: `Chapter ${i}` });
    }
    // a link to a chapter the book doesn't list still shows where the page is
    if (!opts.some((o) => o.value === String(chapter))) opts.push({ value: String(chapter), label: `Chapter ${chapter}` });
    return opts;
  })();
  const warnings = segments.data?.edit_warnings ?? [];
  const flaggedHere = overview.data?.flagged.filter((f) => f.chapter_index === chapter) ?? [];

  return (
    <section className="screen">
      <h1 className="flex items-baseline gap-3.5">
        {SCREENS.review}
        <TalkSelect
          className="bookpick"
          ariaLabel="book"
          value={bookId}
          onChange={(v) => setParams({ book: v })}
          options={(books.data?.books ?? [])
            .filter((b) => b.attributed)
            .map((b) => ({ value: b.book_id, label: b.title ?? b.book_id }))}
        />
      </h1>
      <p className="sub">
        The machine's attributions in the margin, the book on the page. Fixes are durable edits replayed over every
        re-attribution — anchors record exactly what you were looking at.
      </p>

      {!attributed && book.data && (
        <>
          <p className="mb-2 text-ink-2">
            Seiyuu hasn't worked out who says each line in this book yet, so there is nothing to review.
          </p>
          {book.data.status.ingested ? (
            <AttributePanel
              key={bookId}
              book={{ book_id: bookId, attributed: false, active_job: book.data.active_job }}
            />
          ) : (
            <NextStep
              step={nextStep({ status: book.data.status, activeJob: book.data.active_job, want: "attribute" })}
            />
          )}
        </>
      )}

      {attributed && (
        <>
          <div className="drainstrip">
            <span className="state"><i className={`led ${overview.data && overview.data.low_confidence_segments > 0 ? "warn" : "ok"}`} />review queue</span>
            <span className="mono text-[11.5px] text-ink-2">
              {overview.data?.low_confidence_segments ?? "…"} to check in the book ·{" "}
              {overview.data && overview.data.unattributed_quote_segments > 0
                ? `${overview.data.unattributed_quote_segments} quote(s) with no speaker set · `
                : ""}
              {lowConfInChapter.length} in this chapter
            </span>
            {overview.data && (
              <Tip
                content={`the AI that found the speakers: ${overview.data.provider_id} · ${overview.data.model_id} · prompt ${overview.data.prompt_version} ${cliHint("attribute")}`}
              >
                <span className="mono text-[10.5px] text-ink-3" tabIndex={0}>
                  speakers found by {overview.data.model_id}
                </span>
              </Tip>
            )}
            <Tip content="go to the next line worth checking — it carries on into later chapters">
              <button
                ref={nextRef}
                className="key quiet"
                onClick={() => {
                  if (!nextBusy && !nothingToCheck) void nextToCheck();
                }}
                aria-disabled={nextBusy || nothingToCheck || undefined}
                aria-busy={walk.state === "looking" || undefined}
              >
                {walk.state === "looking" ? `looking in chapter ${walk.chapter}…` : "Next to check ▸"}
              </button>
            </Tip>
            <span className="flex-1" />
            <Tip content="every fix you made to this book, with undo for the newest">
              <button className="key quiet" onClick={() => setHistoryOpen(true)}>
                History · {editLog.data?.ops.length ?? 0}
              </button>
            </Tip>
          </div>

          {/* "Next to check" speaks here. Always mounted, so screen readers hear the end, a
              failure or a landing arrive (a region inserted together with its text is often
              not announced). A bare polite region, not role=status, like PlayableRow's. */}
          <div aria-live="polite">
            {walk.state === "looking" && (
              <span className="sr-only">Looking for the next line to check in chapter {walk.chapter}…</span>
            )}
            {walk.state === "landed" && <span className="sr-only">{walk.message}</span>}
            {walk.state === "end" && (
              <p className="navnote -mt-2 mb-3.5">
                {walk.fromStart
                  ? `Nothing to check${filterSuffix} in this book.`
                  : `Nothing left to check${filterSuffix} from here to the end of the book.`}{" "}
                {!walk.fromStart && (
                  <button
                    className="link"
                    onClick={() => {
                      // this link goes away as the walk starts: keep focus on Next instead
                      nextRef.current?.focus();
                      void nextToCheck(true);
                    }}
                  >
                    start again from the first chapter
                  </button>
                )}
              </p>
            )}
            {walk.state === "error" && (
              <div className="refusal -mt-2 mb-3.5">
                <span className="tag">couldn't look further</span>
                <p>{walk.message}</p>
              </div>
            )}
          </div>

          <div className="review">
            <div className="panel roster" style={{ margin: 0 }}>
              <div className="panel-h">
                <b>Cast</b>
                <span className="tag ml-2">{cast.length}</span>
                <ViewMenu spoilerSafe={spoilerSafe} onSpoilerSafe={setSpoilerSafe} />
              </div>
              {spoilerSafe && (
                <div className="frontier">
                  <span className="tag">frontier</span> read through ch{" "}
                  <input
                    className="frontin"
                    aria-label="read through chapter"
                    type="number"
                    min={1}
                    max={chapterCount}
                    value={frontier}
                    onChange={(e) => setFrontier(Number(e.target.value) || 1)}
                  />{" "}
                  — later characters stay hidden
                </div>
              )}
              <div className="caststrip">
                {/* One "Suggest a cast" in both states, at the same position, so it keeps its
                    identity when the saved cast loads under the pointer. It also replaces the old
                    re-draft: with re-cast off, applying a suggestion fills newly found characters
                    and keeps every existing voice and hand pick. */}
                {!casting && (
                  <span className="text-xs text-ink-2">
                    no cast yet — Seiyuu suggests a voice for every character, and you can change any of them before
                    applying
                  </span>
                )}
                <Tip
                  content={
                    castPreview
                      ? "a suggestion is open below — change it there, or dismiss it to start over"
                      : casting
                        ? "preview a cast: newly found characters get their own voices, existing voices and your picks stay — nothing changes until you apply"
                        : "preview a cast: every character a distinct voice — nothing changes until you apply"
                  }
                >
                  <button
                    className={casting ? "key quiet px-[9px] py-[3px]" : "key ml-auto px-3 py-[3px]"}
                    // while a preview is open its own controls change it; a fresh suggestion here
                    // would silently throw away the user's row, narrator and accent changes
                    disabled={suggestCast.isPending || castPreview !== null}
                    onClick={openSuggest}
                  >
                    {suggestCast.isPending && !castPreview ? "thinking…" : "Suggest a cast"}
                  </button>
                </Tip>
                {casting && (
                  <>
                    <Tip content="start over: re-cast everyone with auto voices, discarding your picks (asks first)">
                      <button
                        className="key quiet px-[7px] py-[3px] text-[10.5px] text-ink-3"
                        disabled={draftCast.isPending || !assignment.data}
                        onClick={() => setResetOpen(true)}
                      >
                        reset cast…
                      </button>
                    </Tip>
                    <button
                      className="key ml-auto px-3 py-[3px]"
                      disabled={!castingDirty || saveCast.isPending}
                      onClick={() => {
                        // capture BEFORE the save overwrites the server copy: did the voices
                        // actually change (vs a stage-only edit)? Only that makes audio stale.
                        const voicesChanged = assignment.data
                          ? castingVoicesDiffer(assignment.data, casting)
                          : true;
                        saveCast.mutate(
                          {
                            stage: casting.stage,
                            narrator_voice_id: casting.narrator,
                            assignments: casting.map,
                            thought_voice_id: casting.thought,
                          },
                          { onSuccess: () => voicesChanged && setStaleAfterSave(true) },
                        );
                      }}
                    >
                      {saveCast.isPending ? "saving…" : castingDirty ? "save casting" : "saved ✓"}
                    </button>
                  </>
                )}
              </div>
              {casting && (
                <div className="caststrip">
                  <details className="fine">
                    <summary>advanced · cast stage: {casting.stage}</summary>
                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      {(["draft", "final"] as const).map((s) => (
                        <button
                          key={s}
                          className={`chap px-[9px] py-[2px] ${casting.stage === s ? "on" : ""}`}
                          onClick={() => setCasting({ ...casting, stage: s })}
                        >
                          {s}
                        </button>
                      ))}
                      <span className="mono text-[10.5px] text-ink-3">
                        mark the cast final once you have signed it off (for example before paid cloud voices).
                        The stage alone re-records nothing; save to keep it.
                      </span>
                    </div>
                  </details>
                </div>
              )}
              {casting && allTags.length > 0 && (
                <div className="caststrip flex-wrap gap-1.5">
                  <Tip content={`narrow every voice picker to voices carrying ALL the selected tags — tag voices on the ${SCREENS.voices} screen`}>
                    <span className="tag">voice tags</span>
                  </Tip>
                  {allTags.map((t) => (
                    <button
                      key={t}
                      className={`chap px-2 py-[2px] ${tagFilter.includes(t) ? "on" : ""}`}
                      onClick={() => toggleTag(t)}
                    >
                      {t}
                    </button>
                  ))}
                  {tagFilter.length > 0 && (
                    <>
                      <span className="mono text-[10.5px] text-ink-3">
                        {filteredVoices.length} of {allVoices.length} voices
                      </span>
                      <button className="key quiet px-2 py-[2px]" onClick={() => setTagFilter([])}>
                        clear
                      </button>
                    </>
                  )}
                </div>
              )}
              {castPreview && (
                <SuggestCastPanel
                  preview={castPreview}
                  saved={assignment.data ?? null}
                  characters={cast}
                  isMasked={(id) => maskedIds.has(id)}
                  voices={allVoices}
                  refreshing={suggestCast.isPending}
                  unsavedEdits={castingDirty}
                  applying={draftCast.isPending}
                  applyError={draftCast.error instanceof ApiError ? draftCast.error : null}
                  onPreview={requestPreview}
                  onApply={({ input, recast, useLlm, confirmPaid }) =>
                    draftCast.mutate(
                      {
                        strategy: "smart",
                        stage: savedStage,
                        ...input,
                        recast,
                        use_llm: useLlm,
                        ...(confirmPaid ? { confirm_paid: true } : {}),
                      },
                      { onSuccess: closeSuggest },
                    )
                  }
                  onDismiss={closeSuggest}
                />
              )}
              {/* only the FIRST request's failure lands here; an open panel reports its own */}
              {suggestCast.error && !castPreview && (
                <PlainError className="mx-3 my-2" error={suggestCast.error} title="couldn't suggest a cast" />
              )}
              {draftCast.data && draftCast.data.created_voice_ids.length > 0 && (
                <div className="caststrip mono text-[11px] text-ok">
                  made {draftCast.data.created_voice_ids.length} new voice
                  {draftCast.data.created_voice_ids.length === 1 ? "" : "s"} for this cast
                </div>
              )}
              {(draftCast.error || saveCast.error) && (
                <div className="refusal mx-3 my-2">
                  <span className="tag">
                    {draftCast.error instanceof ApiError && draftCast.error.code === "payment_confirmation_required"
                      ? "the AI caster is paid — approve it below to continue"
                      : draftCast.error
                        ? "couldn't apply the cast"
                        : "couldn't save casting"}
                  </span>
                  {!(draftCast.error instanceof ApiError && draftCast.error.code === "payment_confirmation_required") && (
                    <PlainErrorLine as="div" error={draftCast.error ?? saveCast.error} />
                  )}
                </div>
              )}
              {staleAfterSave && !castingDirty && book.data?.status.rendered && (
                <div className="caststrip flex-wrap gap-2 text-[11px]">
                  <span className="tag">audio is stale</span>
                  <span className="mono text-ink-2">
                    voices changed — the rendered audio still uses the old ones.
                  </span>
                  <Link className="link" to={`/render?book=${encodeURIComponent(bookId)}`}>
                    re-record on {SCREENS.render} →
                  </Link>
                </div>
              )}
              {overview.isPending && <div className="loadline p-3.5">reading the registry…</div>}
              {overview.isError && (
                <PlainError className="m-3" error={overview.error} title={`couldn't load the ${term("registry")}`} />
              )}
              <table>
                <tbody>
                  <tr>
                    <th>character</th>
                    <th>lines</th>
                    <th>voice</th>
                  </tr>
                  <PlayableRow
                    colSpan={3}
                    voice={casting ? voiceById.get(casting.narrator) : undefined}
                    // the narrator reads the open chapter's first stretch of narration
                    text={auditionLine(segments.data?.segments.find((s) => s.type === "narration")?.text)}
                    who="the narrator's voice"
                    cells={
                      <>
                        <td>Narration<span className="sample">the book's own voice</span></td>
                        <td>{overview.data?.narration_segments.toLocaleString()}</td>
                      </>
                    }
                    picker={
                      casting && (
                        <VoicePicker
                          value={casting.narrator}
                          onChange={(v) => v && setCasting({ ...casting, narrator: v })}
                          voices={filteredVoices}
                          pool={allVoices}
                        />
                      )
                    }
                  />
                  {casting && (
                    <tr>
                      <td className="text-ink-2">
                        Thoughts<span className="sample">inner voice for thought segments</span>
                      </td>
                      <td>—</td>
                      <td className="vcell">
                        <VoicePicker
                          value={casting.thought}
                          onChange={(v) => setCasting({ ...casting, thought: v })}
                          voices={filteredVoices}
                          pool={allVoices}
                          allowOwn
                        />
                      </td>
                    </tr>
                  )}
                  {cast.map((c) => (
                    <RosterRow
                      key={c.id}
                      char={c}
                      masked={isMasked(c)}
                      onReveal={() => setRevealed(new Set(revealed).add(c.id))}
                      onEdit={() => setEditingChar(c)}
                      selected={speakerFilter === c.id}
                      onSelect={() =>
                        patchParams({ speaker: speakerFilter === c.id ? null : c.id, block: null, seg: null })
                      }
                      voice={casting?.map[c.id] ? voiceById.get(casting.map[c.id]) : undefined}
                      voiceCell={
                        casting ? (
                          <VoicePicker
                            value={casting.map[c.id] ?? null}
                            onChange={(v) => v && setCasting({ ...casting, map: { ...casting.map, [c.id]: v } })}
                            voices={filteredVoices}
                            pool={allVoices}
                            newVoiceFor={c.name}
                            onNewVoice={() => setNewVoiceFor(c)}
                          />
                        ) : (
                          <span className="mono text-ink-3">—</span>
                        )
                      }
                    />
                  ))}
                </tbody>
              </table>
              <MergeSuggestions
                key={bookId}
                bookId={bookId}
                // spoiler-safe: a pair naming a character not met yet stays out of sight
                suggestions={mergeSuggestions(overview.data?.notes ?? [], cast).filter(
                  (m) => !maskedIds.has(m.loser.id) && !maskedIds.has(m.winner.id),
                )}
              />
            </div>

            <div className="page-wrap" ref={pageRef}>
              <div className="flex items-center gap-2.5 mb-3.5">
                <button className="chap" disabled={chapter <= 1} onClick={() => setChapter(chapter - 1)}>‹</button>
                <TalkSelect
                  ariaLabel="chapter"
                  value={String(chapter)}
                  onChange={(v) => setChapter(Number(v))}
                  options={chapterOptions}
                />
                <span className="tag">of {chapterCount}</span>
                <button className="chap" disabled={chapter >= chapterCount} onClick={() => setChapter(chapter + 1)}>›</button>
                {emotionMap.size > 0 && (
                  <Tip content="the emotion the model tagged each line with — voiced only when a recording has emotion turned on">
                    <span className="mono text-[10.5px] text-ink-3">
                      {emotionMap.size} emotion tag(s) in this chapter
                    </span>
                  </Tip>
                )}
                {warnings.length > 0 && (
                  <span className="mono ml-auto text-[10.5px] text-caution">
                    {warnings.length} edit warning(s)
                  </span>
                )}
              </div>
              {warnings.map((w) => {
                const d = describeEditWarning(w);
                const to = d.target;
                return (
                  <div className="mwarn mb-2.5" key={w}>
                    {d.text}
                    {to && (
                      <button className="link goto" onClick={() => goToLine(to)}>
                        Go to line
                      </button>
                    )}
                    <span className="raw">{d.raw}</span>
                  </div>
                );
              })}
              {flaggedHere.map((f) => (
                <div className="mwarn mb-2.5" key={f.block_id}>
                  Seiyuu couldn't work out who speaks in a paragraph here, so the narrator reads all of it for now. Check
                  it and set the speakers by hand.
                  <button
                    className="link goto"
                    onClick={() => goToLine({ chapter: f.chapter_index, block: f.block_id, seg: null })}
                  >
                    Go to line
                  </button>
                  <span className="raw">{f.reason}</span>
                </div>
              ))}
              {speakerFilter !== null && (
                <div className="filterbar">
                  <span>
                    showing only <b>{filterName}</b>
                    {shown.data &&
                      ` · ${shown.data.segments.length} line${shown.data.segments.length === 1 ? "" : "s"} in this chapter`}
                  </span>
                  <button
                    className="link"
                    onClick={() => patchParams({ speaker: null, block: null, seg: null })}
                  >
                    show all
                  </button>
                </div>
              )}
              {target && shown.data && !targetRow && (
                <p className="navnote mb-2.5" role="status">
                  The linked line isn't on this page
                  {speakerFilter !== null ? " (it may be hidden by the filter)" : ""} — it may have moved since the
                  link was made.
                </p>
              )}
              {shown.isPending && <div className="loadline">setting the page…</div>}
              {shown.isError && <PlainError error={shown.error} title="couldn't load this chapter" />}
              {shown.data && shown.data.segments.length === 0 && speakerFilter !== null && (
                <p className="navnote">No lines for {filterName} in this chapter.</p>
              )}
              {shown.data && (shown.data.segments.length > 0 || speakerFilter === null) && (
                <div className="paper page">
                  {shown.data.segments.map((s) => {
                    const key = `${s.block_id}:${s.segment_index}`;
                    const low = isReviewable(s, threshold);
                    const dimmed = s.speaker !== null && maskedIds.has(s.speaker);
                    const emotion = emotionMap.get(emotionKey(s.block_id, s.segment_index)) ?? null;
                    return (
                      <SegmentPair
                        key={key}
                        segKey={key}
                        row={s}
                        target={key === targetKey}
                        low={low}
                        maskedName={dimmed}
                        emotion={emotion}
                        open={popoverAt === key}
                        onChip={handleChip}
                        popover={
                          popoverAt === key ? (
                            <ReassignPopover
                              row={s}
                              cast={visibleCast}
                              speakerHidden={dimmed}
                              bookId={bookId}
                              onClose={() => setPopoverAt(null)}
                            />
                          ) : null
                        }
                        emoOpen={emoAt === key}
                        onEmotion={handleEmotion}
                        onConfirm={handleConfirm}
                        confirming={confirmLine.isPending && confirmKey === key}
                        confirmLocked={confirmLine.isPending}
                        note={confirmKey === key ? confirmNote : null}
                        onPlay={handlePlay}
                        playing={canPlay(s) && playingKey === `${s.block_id}:${s.audio_segment}`}
                        emotionPopover={
                          emoAt === key ? (
                            <EmotionPopover row={s} current={emotion} bookId={bookId} onClose={() => setEmoAt(null)} />
                          ) : null
                        }
                      />
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </>
      )}
      {historyOpen && bookId && (
        <HistoryDrawer
          bookId={bookId}
          log={editLog.data}
          nameOf={(id) => {
            const c = cast.find((x) => x.id === id);
            return c ? (maskedIds.has(c.id) ? "a hidden character" : c.name) : null;
          }}
          isHidden={(id) => maskedIds.has(id)}
          onGoTo={(t) => {
            setHistoryOpen(false);
            goToLine(t);
          }}
          onClose={() => setHistoryOpen(false)}
        />
      )}
      {editingChar && (
        <CharacterEditor
          char={editingChar}
          others={cast.filter((c) => c.id !== editingChar.id && !isMasked(c))}
          onClose={() => setEditingChar(null)}
          bookId={bookId}
        />
      )}
      {resetOpen && assignment.data && (
        <ResetCastDialog
          assignment={assignment.data}
          characterName={(id) => {
            const c = cast.find((x) => x.id === id);
            // spoiler-safe: a pick for a not-yet-met character is counted, never named
            return c ? (isMasked(c) ? "a hidden character" : c.name) : id;
          }}
          voiceName={(id) => allVoices.find((v) => v.voice_id === id)?.name ?? id}
          unsavedEdits={castingDirty}
          resetting={draftCast.isPending}
          onConfirm={() => {
            setResetOpen(false);
            // the reset starts over with the collision-free caster (the hash one is CLI-only now)
            draftCast.mutate({ strategy: "smart", stage: savedStage, keep_existing: false });
          }}
          onClose={() => setResetOpen(false)}
        />
      )}
      {newVoiceFor && (
        <AddVoiceDialog
          prefill={{
            forCharacter: newVoiceFor.name,
            gender: genderLabel(newVoiceFor.gender),
          }}
          // functional update: the dialog may have been open across other casting changes
          onCreated={(v) =>
            setCasting((cur) => (cur ? { ...cur, map: { ...cur.map, [newVoiceFor.id]: v.voice_id } } : cur))
          }
          onClose={() => setNewVoiceFor(null)}
        />
      )}
    </section>
  );
}
