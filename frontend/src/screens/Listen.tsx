import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  useAssignment,
  useBook,
  useBooks,
  useRenderSummary,
  useSegments,
  useSegmentWords,
  useVoices,
} from "../api/hooks";
import type { SegmentWordsClip } from "../api/hooks";
import type { BookCard, SegmentRow } from "../api/types";
import { useScreenBook } from "../app/currentBook";
import { usePlayer, type PlayClip, type PlayWord } from "../app/usePlayer";
import { NextStep } from "../components/NextStep";
import { StaleAudioBanner } from "../components/StaleAudio";
import {
  clock,
  fileSize,
  parseResume,
  resumeKey,
  resumeTarget,
  voiceChangedSince,
  type ResumePoint,
} from "../lib/listen";
import { ADD_A_BOOK, bookHref, nextStep, RENDER_ANCHOR } from "../lib/nextStep";
import { readStored, removeStored, writeStored } from "../lib/storage";
import { alignWordTimings, buildClipWords, groupPlayableRows } from "../lib/words";
import { holdListenChapter, listenHeldChapter } from "./listen/held";
import { LineMenu } from "./listen/LineMenu";
import { ResumeRecorder } from "./listen/ResumeRecorder";

/* -------------------------------------------------- reading preferences */

type ReadingTheme = "paper" | "sepia" | "dark";
type ReadingSize = "s" | "m" | "l";

function usePrefs() {
  const [prefs, setPrefs] = useState<{ theme: ReadingTheme; size: ReadingSize }>(() => {
    try {
      return { theme: "paper", size: "m", ...JSON.parse(localStorage.getItem("seiyuu.reading") ?? "{}") };
    } catch {
      return { theme: "paper", size: "m" };
    }
  });
  const update = (next: Partial<{ theme: ReadingTheme; size: ReadingSize }>) => {
    const merged = { ...prefs, ...next };
    setPrefs(merged);
    writeStored("seiyuu.reading", JSON.stringify(merged));
  };
  return [prefs, update] as const;
}

/* -------------------------------------------------- the shelf (cover picker) */

function CoverTile({ book, onPick }: { book: BookCard; onPick: () => void }) {
  const [imgOk, setImgOk] = useState(true);
  return (
    <button className="covertile" onClick={onPick} disabled={!book.rendered} title={book.rendered ? "" : "no audio yet — render it first"}>
      {imgOk ? (
        <img src={`/api/books/${book.book_id}/cover`} alt="" onError={() => setImgOk(false)} />
      ) : (
        <span className="coverfb">
          <b className="serif">{book.title ?? book.book_id}</b>
          <i />
          <span>{book.authors.join(", ") || "—"}</span>
        </span>
      )}
      {!book.rendered && <span className="coverbadge">not rendered</span>}
    </button>
  );
}

/* -------------------------------------------------- the screen */

export function Listen() {
  const [, setParams] = useSearchParams();
  const books = useBooks();
  const player = usePlayer();
  // The imperative word-span handlers below outlive the render that created them, and the
  // player api is rebuilt on every state change — a captured `player` still holds the clips
  // from BEFORE load() ran (empty on first build), so seekClip would silently no-op. Same
  // latest-ref idiom as hlRef.
  const playerRef = useRef(player);
  playerRef.current = player;
  // no ?book= on arrival: open the book this viewer last worked on (no default — the bare
  // URL is the shelf, and "‹ shelf" must stay there)
  const bookId = useScreenBook(books.data?.books.map((b) => b.book_id), null, { once: true });
  const book = useBook(bookId);
  const rendered = !!book.data?.status.rendered;

  const summary = useRenderSummary(bookId, rendered);
  // the margin credits each line's recorded voice by NAME, and notes when today's cast would
  // give the line a different voice (its speaker or that speaker's voice changed since)
  const voices = useVoices();
  const voiceNames = useMemo(() => new Map(voices.data?.voices.map((v) => [v.voice_id, v.name] as const) ?? []), [voices.data]);
  const assignment = useAssignment(bookId, !!book.data?.status.attributed);
  const renderedChapters = useMemo(() => new Set(summary.data?.chapters.map((c) => c.index) ?? []), [summary.data]);

  // Coming BACK to this book while the player still holds its Listen chapter (a fix link to
  // Review or Pronunciation and back) stays on that chapter: reloading chapter 1 would stop
  // and rewind the listener mid-book. Nothing to resume then — they never left their place.
  // Read during render too (not only in the effect below): the build effect runs in the same
  // commit as that effect and must already see the held chapter, or it loads chapter 1.
  const [chapter, setChapterRaw] = useState<number | null>(() => listenHeldChapter(player, bookId));
  const heldOnArrival = chapter === null ? listenHeldChapter(player, bookId) : null;

  /* resume where you left off: read once per book open; offered until this book plays here */
  const savedResume = useMemo(() => (bookId ? parseResume(readStored(resumeKey(bookId))) : null), [bookId]);
  const [resumeDoneFor, setResumeDoneFor] = useState<string | null>(() =>
    listenHeldChapter(player, bookId) !== null ? bookId : null,
  );
  // a resume into another chapter, waiting for that chapter's clips to reach the player
  const pendingResume = useRef<typeof savedResume>(null);

  // Chapter is per-book; without this a shelf switch resumes at the OLD book's number.
  useEffect(() => {
    const held = listenHeldChapter(playerRef.current, bookId);
    setChapterRaw(held);
    setResumeDoneFor(held !== null ? bookId : null);
    pendingResume.current = null;
  }, [bookId]);
  // Latch the deep-link default ONCE. Deriving it live means a ch-1 render landing
  // mid-listen changes chapters[0] and yanks playback to chapter 1. An updater, so the held
  // chapter set in the same commit above is never overwritten.
  useEffect(() => {
    const first = summary.data?.chapters[0]?.index;
    if (chapter === null && first !== undefined) setChapterRaw((c) => c ?? first);
  }, [summary.data, chapter]);
  // listening started here (on this book's Listen clips, not a Review one-liner): the offer goes
  useEffect(() => {
    if (bookId && player?.playing && listenHeldChapter(player, bookId) !== null) setResumeDoneFor(bookId);
  }, [player, bookId]);
  const effectiveChapter = chapter ?? heldOnArrival ?? summary.data?.chapters[0]?.index ?? 1;
  const segments = useSegments(bookId, effectiveChapter, rendered);

  const [prefs, setPrefs] = usePrefs();
  const [tocOpen, setTocOpen] = useState(false);
  const pageRef = useRef<HTMLDivElement>(null);
  const autoplayNext = useRef(false);
  // the word last clicked, per line: "Fix pronunciation" offers it when nothing is selected
  const lastWord = useRef<{ seg: string; text: string } | null>(null);

  const setChapter = (c: number) => {
    // switching mid-listen carries the playback over — the new chapter starts on its own
    // instead of landing paused (the old stop-dead-on-switch complaint)
    autoplayNext.current = !!playerRef.current?.playing;
    // picking another chapter cancels a resume still waiting for its chapter to load
    if (pendingResume.current?.chapter !== c) pendingResume.current = null;
    setChapterRaw(c);
    setTocOpen(false);
  };

  // nearest rendered neighbours for the ‹ › chapter keys
  const prevRendered = Math.max(...[...renderedChapters].filter((i) => i < effectiveChapter), -Infinity);
  const nextRendered = Math.min(...[...renderedChapters].filter((i) => i > effectiveChapter), Infinity);

  const playableRows = useMemo<SegmentRow[]>(
    () => segments.data?.segments.filter((s) => s.has_audio && s.duration_seconds !== null && s.audio_segment !== null) ?? [],
    [segments.data],
  );

  // rows sharing one rendered wav collapse into ONE clip; remember where each row
  // starts inside its clip so clicking a segment seeks to ITS words, not the block top
  const rowSeek = useRef(new Map<string, { clip: number; offset: number }>());

  // one clip per rendered wav; fetch whisper word-timings for each, busted by audio_key
  const wordClips = useMemo<SegmentWordsClip[]>(
    () =>
      groupPlayableRows(playableRows).map((g) => {
        const sep = g.key.lastIndexOf(":");
        return {
          key: g.key,
          blockId: g.key.slice(0, sep),
          segment: Number(g.key.slice(sep + 1)),
          audioKey: g.rows[0].audio_key,
        };
      }),
    [playableRows],
  );
  const words = useSegmentWords(bookId, effectiveChapter, wordClips);

  // built clip structure (span elements + per-row token counts) so whisper timings can be
  // applied onto the EXISTING spans without a player.load() that would interrupt playback
  const builtRef = useRef<
    { ci: number; key: string; clipWords: PlayWord[]; rows: { rowKey: string; count: number }[]; duration: number }[]
  >([]);
  // Bumped every time the build effect rebuilds builtRef (a same-chapter reassignment, alias
  // merge, or refetch gives playableRows a new identity and resets every clip to interpolation).
  // The apply effect depends on this, so it re-applies the cached whisper timings onto the fresh
  // spans even when `words.sig` didn't change — otherwise the read-along drifts permanently.
  const [buildGen, setBuildGen] = useState(0);

  useEffect(() => {
    if (!player || !bookId || !pageRef.current || playableRows.length === 0) return;
    const groups = groupPlayableRows(playableRows);
    rowSeek.current.clear();
    builtRef.current = [];
    const clips: PlayClip[] = groups.map((g, ci) => {
      const pairs = g.rows
        .map((row) => ({
          row,
          el: pageRef.current!.querySelector<HTMLElement>(`[data-seg="${row.block_id}:${row.segment_index}"] .segtext`),
        }))
        .filter((p): p is { row: SegmentRow; el: HTMLElement } => p.el !== null);
      const clipWords = buildClipWords(pairs.map((p) => ({ text: p.row.text, el: p.el })), g.duration);
      // first word of each row = that row's seek point; also record token counts so a later
      // whisper-timing pass can recompute these seek points against the real word offsets
      const rows: { rowKey: string; count: number }[] = [];
      let w = 0;
      for (const p of pairs) {
        const rowKey = `${p.row.block_id}:${p.row.segment_index}`;
        rowSeek.current.set(rowKey, { clip: ci, offset: clipWords[w]?.offset ?? 0 });
        const count = p.row.text.split(/\s+/).filter(Boolean).length;
        rows.push({ rowKey, count });
        w += count;
      }
      clipWords.forEach((word) => {
        word.el.style.cursor = "pointer";
        // reads word.offset LIVE at click time, so it picks up whisper-refined offsets
        word.el.onclick = (e) => {
          e.stopPropagation();
          const seg = word.el.closest("[data-seg]")?.getAttribute("data-seg");
          if (seg) lastWord.current = { seg, text: word.el.textContent ?? "" };
          playerRef.current?.seekClip(ci, word.offset);
        };
      });
      builtRef.current.push({ ci, key: g.key, clipWords, rows, duration: g.duration });
      const [blockId, audioSegment] = g.key.split(":");
      // v= is the wav's SegmentKey hash: a re-render changes it, so the browser can
      // never serve a stale clip from before the re-render
      const buster = g.rows[0].audio_key ? `&v=${g.rows[0].audio_key}` : "";
      return {
        src: `/api/books/${bookId}/segments/${blockId}/audio?segment=${audioSegment}${buster}`,
        duration: g.duration,
        key: g.key,
        speaker: g.speaker,
        words: clipWords,
      };
    });
    const chapterDone = () => {
      // advance the spoiler frontier, then roll into the next rendered chapter
      const key = `seiyuu.frontier.${bookId}`;
      const prev = Number(readStored(key)) || 1;
      writeStored(key, String(Math.max(prev, effectiveChapter)));
      const next = effectiveChapter + 1;
      if (renderedChapters.has(next)) {
        autoplayNext.current = true;
        if (pendingResume.current?.chapter !== next) pendingResume.current = null;
        setChapterRaw(next);
        return;
      }
      // Playback stops here, so the last remembered place is a second before this chapter's
      // end — never worth offering. Point at the start of the next recorded chapter instead,
      // or forget the place when the listener reached the end of what is recorded.
      const later = Math.min(...[...renderedChapters].filter((i) => i > effectiveChapter), Infinity);
      if (Number.isFinite(later)) {
        const start: ResumePoint = { chapter: later, clip: "", offset: 0, seconds: 0 };
        writeStored(resumeKey(bookId), JSON.stringify(start));
      } else {
        removeStored(resumeKey(bookId));
      }
    };
    player.load(bookId, segments.data?.title ?? `Chapter ${effectiveChapter}`, clips, {
      autoplay: autoplayNext.current,
      onEnded: chapterDone,
    });
    autoplayNext.current = false;
    holdListenChapter(bookId, effectiveChapter, clips);
    // signal the apply effect that clips were rebuilt so it re-applies whisper timings even
    // when the resolved-clip signature is unchanged (same chapter, new playableRows identity)
    setBuildGen((g) => g + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bookId, effectiveChapter, playableRows]);

  // When a clip's whisper words arrive, drive its highlight from the real (start,end) times
  // instead of length interpolation — mutating the existing spans in place (no player.load,
  // so playback never hitches) and refreshing that clip's per-row seek points. Clips still
  // loading or 404'd keep their interpolated fallback. Re-runs on `words.sig` (timings arrive
  // or a clip's audio identity changes) AND on `buildGen` (the build effect rebuilt every clip
  // back to interpolation for the same chapter — reassignment, alias merge, or refetch).
  useEffect(() => {
    for (const clip of builtRef.current) {
      const sw = words.byKey.get(clip.key);
      if (!sw || sw.words.length === 0) continue;
      const displayTokens = clip.clipWords.map((cw) => cw.el.textContent ?? "");
      const timings = alignWordTimings(displayTokens, sw.words, sw.audio_duration || clip.duration);
      clip.clipWords.forEach((cw, k) => {
        if (timings[k]) {
          cw.offset = timings[k].offset;
          cw.end = timings[k].end;
        }
      });
      let idx = 0;
      for (const r of clip.rows) {
        rowSeek.current.set(r.rowKey, { clip: clip.ci, offset: clip.clipWords[idx]?.offset ?? 0 });
        idx += r.count;
      }
    }
    // buildGen re-fires this on every rebuild (a same-chapter reassignment/refetch resets the
    // clips to interpolation without changing words.sig); words.sig re-fires it when timings load
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [words.sig, effectiveChapter, buildGen]);

  // A resume into another chapter waits for that chapter's clips to reach the player (the
  // build effect bumps buildGen in the same commit as player.load), then seeks into them.
  useEffect(() => {
    const point = pendingResume.current;
    const p = playerRef.current;
    if (!point || !p || listenHeldChapter(p, bookId) !== point.chapter) return;
    pendingResume.current = null;
    const at = resumeTarget(p.clips, point);
    if (at) p.seekClip(at.index, at.offset);
  }, [buildGen, bookId]);

  // Highlight loop: rAF reading the audio element's clock directly — smooth and never
  // behind the 250ms timeupdate events.
  const hlRef = useRef<{ clips: PlayClip[]; index: number }>({ clips: [], index: 0 });
  hlRef.current = { clips: player?.clips ?? [], index: player?.index ?? 0 };
  useEffect(() => {
    let raf = 0;
    let last: HTMLElement | null = null;
    const loop = () => {
      const audio = player?.audio;
      const { clips, index } = hlRef.current;
      const clip = clips[index];
      if (audio && clip && clip.words.length) {
        const t = audio.currentTime;
        // exact active word: the first whose end is still ahead of the playhead
        let i = clip.words.findIndex((word) => t < word.end);
        if (i < 0) i = clip.words.length - 1;
        const el = clip.words[i]?.el ?? null;
        if (el !== last) {
          last?.classList.remove("now");
          last?.closest(".seg")?.classList.remove("now-seg");
          el?.classList.add("now");
          el?.closest(".seg")?.classList.add("now-seg");
          el?.closest(".seg")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
          last = el;
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [player?.audio]);

  /* ---------------- shelf (no book chosen) ---------------- */
  if (books.isPending) return <section className="screen"><div className="loadline">reading the shelf…</div></section>;
  if (books.isError) return <section className="screen"><h1>Listen</h1><div className="errline">{books.error.message}</div></section>;
  if (!bookId) {
    return (
      <section className="screen">
        <h1>Listen</h1>
        <p className="sub">Pick a book from the shelf — upload cover art in Render &amp; Jobs to make this prettier.</p>
        <div className="shelf">
          {books.data?.books.map((b) => (
            <CoverTile key={b.book_id} book={b} onPick={() => setParams({ book: b.book_id })} />
          ))}
        </div>
        {books.data?.books.length === 0 && (
          <>
            <div className="loadline">No books yet.</div>
            <NextStep step={ADD_A_BOOK} />
          </>
        )}
      </section>
    );
  }

  const chapterTitles = new Map(book.data?.chapters?.map((c) => [c.index, c.title] as const) ?? []);
  const activeJob = book.data?.active_job ?? null;
  // a job already working on the book outranks any nudge: show what it is doing, disabled
  const recordThisChapter = activeJob
    ? nextStep({ status: book.data?.status, activeJob })
    : { label: "Record this chapter", to: bookHref("render", bookId, RENDER_ANCHOR.record) };
  const m4b = book.data?.downloads.m4b ?? null;
  // no .m4b yet: the "Build the .m4b" step sits where the download would (a running job wins)
  const buildStep = nextStep({ status: book.data?.status, activeJob });
  const fullCast = summary.data?.mode === "multivoice";
  const unreadableVoices = new Set(voices.data?.unreadable.map((u) => u.voice_id) ?? []);
  // The margin's voice credit: the library name. A voice whose files can't be read is still in
  // the library (Voices can repair it), so it is not called deleted; and when the library
  // itself failed to load, the raw id beats showing nothing.
  const voiceLabel = (id: string): { text: string; broken: boolean } | null => {
    const name = voiceNames.get(id);
    if (name) return { text: name, broken: false };
    if (voices.isError) return { text: id, broken: false };
    if (!voices.data) return null;
    if (unreadableVoices.has(id)) return { text: "a voice that can't be opened right now", broken: true };
    return { text: "a voice no longer in your library", broken: false };
  };

  // offered until this book plays (and not for "the very start of the first chapter")
  const firstRendered = summary.data?.chapters[0]?.index ?? null;
  const resumeOffer =
    savedResume &&
    resumeDoneFor !== bookId &&
    renderedChapters.has(savedResume.chapter) &&
    !(savedResume.chapter === firstRendered && savedResume.seconds < 3)
      ? savedResume
      : null;
  const onResume = () => {
    if (!resumeOffer) return;
    setResumeDoneFor(bookId);
    const p = playerRef.current;
    if (p && listenHeldChapter(p, bookId) === resumeOffer.chapter) {
      const at = resumeTarget(p.clips, resumeOffer);
      if (at) p.seekClip(at.index, at.offset);
    } else {
      setChapter(resumeOffer.chapter);
      pendingResume.current = resumeOffer;
    }
  };

  return (
    <section className="screen">
      <ResumeRecorder bookId={bookId} />
      <div className="readerhead">
        <button className="key quiet" onClick={() => setParams({})}>‹ shelf</button>
        <h1 className="m-0">{book.data?.status.title ?? bookId}</h1>
        <button
          className="chap"
          disabled={!Number.isFinite(prevRendered)}
          title="previous rendered chapter"
          onClick={() => setChapter(prevRendered)}
        >
          ‹
        </button>
        <button
          className="chap"
          disabled={!Number.isFinite(nextRendered)}
          title="next rendered chapter"
          onClick={() => setChapter(nextRendered)}
        >
          ›
        </button>
        <div className="tocwrap">
          <button className="key quiet" onClick={() => setTocOpen(!tocOpen)}>
            contents ▾
          </button>
          {tocOpen && (
            <div className="tocmenu">
              {(book.data?.chapters ?? []).map((c) => (
                <button
                  key={c.index}
                  className={`tocrow ${c.index === effectiveChapter ? "on" : ""}`}
                  disabled={!renderedChapters.has(c.index)}
                  onClick={() => setChapter(c.index)}
                >
                  <i className={`led ${renderedChapters.has(c.index) ? "ok" : "off"}`} />
                  <span className="mono w-[34px]">{c.index}</span>
                  <span className="toctitle">{c.title}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <span className="flex-1" />
        <span className="tag">page</span>
        {(["paper", "sepia", "dark"] as const).map((t) => (
          <button key={t} className={`swatch sw-${t} ${prefs.theme === t ? "on" : ""}`} title={t} onClick={() => setPrefs({ theme: t })} />
        ))}
        <span className="tag ml-2.5">text</span>
        {(["s", "m", "l"] as const).map((s) => (
          <button key={s} className={`chap px-2 py-[2px] ${prefs.size === s ? "on" : ""}`} onClick={() => setPrefs({ size: s })}>
            {s === "s" ? "A" : s === "m" ? "A+" : "A++"}
          </button>
        ))}
      </div>
      <p className="sub mt-1.5">
        {chapterTitles.get(effectiveChapter) ?? `Chapter ${effectiveChapter}`} — click any word to play from there; the
        transport below seeks, pauses, and holds the volume. Click a speaker to fix that line.
      </p>
      {resumeOffer && (
        <div className="mb-3 flex flex-wrap items-center gap-2.5">
          <button type="button" className="key" onClick={onResume}>
            Resume chapter {resumeOffer.chapter} at {clock(resumeOffer.seconds)}
          </button>
          <span className="mono text-[11px] text-ink-3">
            where you stopped{chapterTitles.get(resumeOffer.chapter) ? ` · ${chapterTitles.get(resumeOffer.chapter)}` : ""}
          </span>
        </div>
      )}
      {summary.data && (
        <div className="provenance flex-wrap">
          <span className="tag">audio</span>
          <span className="state">
            <i className={`led ${summary.data.mode === "multivoice" ? "ok" : "off"}`} />
            {summary.data.mode === "multivoice"
              ? `multivoice · ${Object.keys(summary.data.voices_used ?? {}).length || "cast"} voices`
              : "single voice"}
          </span>
          {summary.data.mode === "single" && book.data?.status.assigned && (
            <span className="flex items-center gap-2">
              <span className="mono text-[11px] text-caution">this audio predates your casting</span>
              <NextStep
                step={{ label: "Record again with your cast", to: bookHref("render", bookId, RENDER_ANCHOR.record) }}
                showReason={false}
              />
            </span>
          )}
          <span className="flex-1" />
          {m4b ? (
            <a className="key" href={m4b.url} download title="the whole book as one audiobook file, with chapters">
              Download .m4b <span className="font-normal text-ink-3">· {fileSize(m4b.bytes)}</span>
            </a>
          ) : (
            buildStep.stepId === "build" && <NextStep step={buildStep} showReason={false} />
          )}
        </div>
      )}
      <StaleAudioBanner status={book.data?.status} />
      {segments.isPending && (rendered || !book.data) && <div className="loadline">setting the page…</div>}
      {segments.isError && <div className="refusal"><span className="tag">not attributed</span><p>{segments.error.message}</p></div>}
      {book.data && !rendered && (
        <div className="refusal">
          <span className="tag">no audio yet</span>
          <p>This book hasn't been recorded yet, so there is nothing to listen to.</p>
          <NextStep step={nextStep({ status: book.data.status, activeJob })} />
        </div>
      )}
      {segments.data && playableRows.length === 0 && (
        <div className="refusal">
          <span className="tag">no audio yet</span>
          <p>This chapter has no audio yet. You can record just this chapter (pick it in the chapter range).</p>
          <NextStep step={recordThisChapter} />
        </div>
      )}
      <div className={`page-wrap rt-${prefs.theme} sz-${prefs.size}`} ref={pageRef}>
        {segments.data && (
          <div className="paper page grid-cols-[minmax(auto,66ch)_150px]">
            {segments.data.segments.map((row) => {
              const segKey = `${row.block_id}:${row.segment_index}`;
              const seek = rowSeek.current.get(segKey);
              const playable = row.has_audio && row.duration_seconds !== null;
              const recordedWith = row.voice_id ? voiceLabel(row.voice_id) : null;
              const nowVoice = voiceChangedSince(row, assignment.data, fullCast);
              return (
                <div key={`${row.block_id}:${row.segment_index}`} className="contents">
                  <p
                    className={`seg serif ${row.type !== "narration" ? "dlg" : ""} ${playable ? "cursor-pointer" : "cursor-default opacity-50"}`}
                    data-seg={`${row.block_id}:${row.segment_index}`}
                    onClick={() => playable && seek && player?.seekClip(seek.clip, seek.offset)}
                  >
                    <span className="segtext">{row.text}</span>
                  </p>
                  <div className="margin">
                    <LineMenu
                      row={row}
                      bookId={bookId}
                      chapter={segments.data.chapter_index}
                      label={row.speaker === null ? "narration" : (row.speaker_name ?? row.speaker)}
                      pickedWord={() => (lastWord.current?.seg === segKey ? lastWord.current.text : null)}
                    />
                    {recordedWith &&
                      (recordedWith.broken ? (
                        <Link
                          to="/voices"
                          className="voicelabel"
                          title={`This voice's files can't be read (${row.voice_id}). Open Voices to repair it.`}
                        >
                          {recordedWith.text}
                        </Link>
                      ) : (
                        <span className="voicelabel" title={`recorded with ${recordedWith.text} (${row.voice_id})`}>
                          {recordedWith.text}
                        </span>
                      ))}
                    {nowVoice && (
                      <span
                        className="voicenote"
                        title="This line's speaker, or that speaker's voice, changed after it was recorded. Record the chapter again to hear the new voice."
                      >
                        changed since recording — now {voiceLabel(nowVoice)?.text ?? "another voice"}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
