import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useBook, useBooks, useRenderSummary, useSegments } from "../api/hooks";
import type { BookCard, SegmentRow } from "../api/types";
import { usePlayer, type PlayClip } from "../app/usePlayer";
import { buildClipWords, groupPlayableRows } from "../lib/words";

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
    localStorage.setItem("seiyuu.reading", JSON.stringify(merged));
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
  const [params, setParams] = useSearchParams();
  const books = useBooks();
  const player = usePlayer();
  const bookId = params.get("book");
  const book = useBook(bookId);
  const rendered = !!book.data?.status.rendered;

  const summary = useRenderSummary(bookId, rendered);
  const renderedChapters = useMemo(() => new Set(summary.data?.chapters.map((c) => c.index) ?? []), [summary.data]);

  const [chapter, setChapterRaw] = useState<number | null>(null);
  const effectiveChapter = chapter ?? summary.data?.chapters[0]?.index ?? 1;
  const segments = useSegments(bookId, effectiveChapter, rendered);

  const [prefs, setPrefs] = usePrefs();
  const [tocOpen, setTocOpen] = useState(false);
  const pageRef = useRef<HTMLDivElement>(null);
  const autoplayNext = useRef(false);

  const setChapter = (c: number) => {
    setChapterRaw(c);
    setTocOpen(false);
  };

  const playableRows = useMemo<SegmentRow[]>(
    () => segments.data?.segments.filter((s) => s.has_audio && s.duration_seconds !== null && s.audio_segment !== null) ?? [],
    [segments.data],
  );

  // rows sharing one rendered wav collapse into ONE clip; remember where each row
  // starts inside its clip so clicking a segment seeks to ITS words, not the block top
  const rowSeek = useRef(new Map<string, { clip: number; offset: number }>());

  useEffect(() => {
    if (!player || !bookId || !pageRef.current || playableRows.length === 0) return;
    const groups = groupPlayableRows(playableRows);
    rowSeek.current.clear();
    const clips: PlayClip[] = groups.map((g, ci) => {
      const pairs = g.rows
        .map((row) => ({
          row,
          el: pageRef.current!.querySelector<HTMLElement>(`[data-seg="${row.block_id}:${row.segment_index}"] .segtext`),
        }))
        .filter((p): p is { row: SegmentRow; el: HTMLElement } => p.el !== null);
      const words = buildClipWords(pairs.map((p) => ({ text: p.row.text, el: p.el })), g.duration);
      // first word of each row = that row's seek point
      let w = 0;
      for (const p of pairs) {
        rowSeek.current.set(`${p.row.block_id}:${p.row.segment_index}`, {
          clip: ci,
          offset: words[w]?.offset ?? 0,
        });
        w += p.row.text.split(/\s+/).filter(Boolean).length;
      }
      words.forEach((word) => {
        word.el.style.cursor = "pointer";
        word.el.onclick = (e) => {
          e.stopPropagation();
          player.seekClip(ci, word.offset);
        };
      });
      const [blockId, audioSegment] = g.key.split(":");
      // v= is the wav's SegmentKey hash: a re-render changes it, so the browser can
      // never serve a stale clip from before the re-render
      const buster = g.rows[0].audio_key ? `&v=${g.rows[0].audio_key}` : "";
      return {
        src: `/api/books/${bookId}/segments/${blockId}/audio?segment=${audioSegment}${buster}`,
        duration: g.duration,
        key: g.key,
        speaker: g.speaker,
        words,
      };
    });
    const chapterDone = () => {
      // advance the spoiler frontier, then roll into the next rendered chapter
      const key = `seiyuu.frontier.${bookId}`;
      const prev = Number(localStorage.getItem(key)) || 1;
      localStorage.setItem(key, String(Math.max(prev, effectiveChapter)));
      const next = effectiveChapter + 1;
      if (renderedChapters.has(next)) {
        autoplayNext.current = true;
        setChapterRaw(next);
      }
    };
    player.load(bookId, segments.data?.title ?? `Chapter ${effectiveChapter}`, clips, {
      autoplay: autoplayNext.current,
      onEnded: chapterDone,
    });
    autoplayNext.current = false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bookId, effectiveChapter, playableRows]);

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
        let i = clip.words.findIndex((_, k) => t < (clip.words[k + 1]?.offset ?? clip.duration));
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
        {books.data?.books.length === 0 && <div className="loadline">no books yet — ingest one from the Library</div>}
      </section>
    );
  }

  const chapterTitles = new Map(book.data?.chapters?.map((c) => [c.index, c.title] as const) ?? []);

  return (
    <section className="screen">
      <div className="readerhead">
        <button className="key quiet" onClick={() => setParams({})}>‹ shelf</button>
        <h1 style={{ margin: 0 }}>{book.data?.status.title ?? bookId}</h1>
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
                  <span className="mono" style={{ width: 34 }}>{c.index}</span>
                  <span className="toctitle">{c.title}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <span style={{ flex: 1 }} />
        <span className="tag">page</span>
        {(["paper", "sepia", "dark"] as const).map((t) => (
          <button key={t} className={`swatch sw-${t} ${prefs.theme === t ? "on" : ""}`} title={t} onClick={() => setPrefs({ theme: t })} />
        ))}
        <span className="tag" style={{ marginLeft: 10 }}>text</span>
        {(["s", "m", "l"] as const).map((s) => (
          <button key={s} className={`chap ${prefs.size === s ? "on" : ""}`} style={{ padding: "2px 8px" }} onClick={() => setPrefs({ size: s })}>
            {s === "s" ? "A" : s === "m" ? "A+" : "A++"}
          </button>
        ))}
      </div>
      <p className="sub" style={{ marginTop: 6 }}>
        {chapterTitles.get(effectiveChapter) ?? `Chapter ${effectiveChapter}`} — click any word to play from there; the
        transport below seeks, pauses, and holds the volume.
      </p>
      {summary.data && (
        <div className="provenance">
          <span className="tag">audio</span>
          <span className="state">
            <i className={`led ${summary.data.mode === "multivoice" ? "ok" : "off"}`} />
            {summary.data.mode === "multivoice"
              ? `multivoice · ${Object.keys(summary.data.voices_used ?? {}).length || "cast"} voices`
              : "single voice"}
          </span>
          {summary.data.mode === "single" && book.data?.status.assigned && (
            <span className="mono" style={{ color: "var(--caution)", fontSize: 11 }}>
              this audio predates your casting — re-render in multivoice to hear it
            </span>
          )}
        </div>
      )}
      {segments.isPending && <div className="loadline">setting the page…</div>}
      {segments.isError && <div className="refusal"><span className="tag">not attributed</span><p>{segments.error.message}</p></div>}
      {segments.data && playableRows.length === 0 && (
        <div className="refusal"><span className="tag">not rendered</span><p>this chapter has no audio yet — render it from Render &amp; Jobs (a chapter range works)</p></div>
      )}
      <div className={`page-wrap rt-${prefs.theme} sz-${prefs.size}`} ref={pageRef}>
        {segments.data && (
          <div className="paper page" style={{ gridTemplateColumns: "minmax(auto,66ch) 150px" }}>
            {segments.data.segments.map((row) => {
              const seek = rowSeek.current.get(`${row.block_id}:${row.segment_index}`);
              const playable = row.has_audio && row.duration_seconds !== null;
              return (
                <div key={`${row.block_id}:${row.segment_index}`} style={{ display: "contents" }}>
                  <p
                    className={`seg serif ${row.type !== "narration" ? "dlg" : ""}`}
                    data-seg={`${row.block_id}:${row.segment_index}`}
                    onClick={() => playable && seek && player?.seekClip(seek.clip, seek.offset)}
                    style={{ cursor: playable ? "pointer" : "default", opacity: playable ? 1 : 0.5 }}
                  >
                    <span className="segtext">{row.text}</span>
                  </p>
                  <div className="margin">
                    <span className={`chip ${row.speaker === null ? "narr" : ""}`}>
                      {row.speaker === null ? "narration" : (row.speaker_name ?? row.speaker).toUpperCase()}
                    </span>
                    {row.voice_id && <span className="voicelabel">{row.voice_id}</span>}
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
