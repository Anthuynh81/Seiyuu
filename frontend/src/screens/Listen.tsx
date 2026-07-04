import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useBook, useBooks, useSegments } from "../api/hooks";
import type { SegmentRow } from "../api/types";
import { usePlayer, type PlayClip } from "../app/usePlayer";

/** Split a segment's text into word spans; interpolate each word's start offset within
    the clip by character length — the same algorithm the design demo used, now driving
    the real rendered audio. */
function buildWords(text: string, duration: number, container: HTMLElement): { el: HTMLElement; offset: number }[] {
  const parts = text.split(/\s+/).filter(Boolean);
  const weight = parts.reduce((a, w) => a + w.length + 1, 0) || 1;
  container.textContent = "";
  const out: { el: HTMLElement; offset: number }[] = [];
  let t = 0;
  parts.forEach((w, i) => {
    const span = document.createElement("span");
    span.className = "w";
    span.textContent = w;
    container.appendChild(span);
    container.appendChild(document.createTextNode(" "));
    out.push({ el: span, offset: t });
    t += (duration * (parts[i].length + 1)) / weight;
  });
  return out;
}

export function Listen() {
  const [params, setParams] = useSearchParams();
  const books = useBooks();
  const player = usePlayer();
  const bookId = params.get("book") ?? books.data?.books.find((b) => b.rendered)?.book_id ?? null;
  const book = useBook(bookId);
  const rendered = !!book.data?.status.rendered;

  const [chapter, setChapter] = useState(2);
  const segments = useSegments(bookId, chapter, rendered);
  const pageRef = useRef<HTMLDivElement>(null);
  const [activeWord, setActiveWord] = useState<HTMLElement | null>(null);

  // Which chapters actually have audio (a segment with a duration).
  const chapterCount = book.data?.chapters?.length ?? 1;

  // Build the clip list + word spans once the DOM is laid out for this chapter.
  const playableRows = useMemo<SegmentRow[]>(
    () => segments.data?.segments.filter((s) => s.has_audio && s.duration_seconds !== null && s.audio_segment !== null) ?? [],
    [segments.data],
  );

  useEffect(() => {
    if (!player || !bookId || playableRows.length === 0 || !pageRef.current) return;
    const clips: PlayClip[] = playableRows.map((row) => {
      const el = pageRef.current!.querySelector<HTMLElement>(`[data-seg="${row.block_id}:${row.segment_index}"] .segtext`);
      const words = el ? buildWords(row.text, row.duration_seconds!, el) : [];
      return {
        src: `/api/books/${bookId}/segments/${row.block_id}/audio?segment=${row.audio_segment}`,
        duration: row.duration_seconds!,
        blockId: row.block_id,
        speaker: row.speaker_name ?? (row.speaker === null ? "narration" : row.speaker),
        words,
      };
    });
    // click any word to seek playback to it (offset within its clip)
    clips.forEach((clip, ci) => {
      clip.words.forEach((w) => {
        w.el.style.cursor = "pointer";
        w.el.onclick = (e) => {
          e.stopPropagation();
          player.seekClip(ci, w.offset);
        };
      });
    });
    player.load(bookId, segments.data?.title ?? `Chapter ${chapter}`, clips);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bookId, chapter, playableRows]);

  // Highlight the current word as the shared player advances.
  useEffect(() => {
    if (!player || !player.clips.length) return;
    const clip = player.clips[player.index];
    if (!clip) return;
    let idx = clip.words.findIndex((_, i) => player.clipElapsed < (clip.words[i + 1]?.offset ?? clip.duration));
    if (idx < 0) idx = clip.words.length - 1;
    const el = clip.words[idx]?.el ?? null;
    if (el !== activeWord) {
      activeWord?.classList.remove("now");
      activeWord?.closest(".seg")?.classList.remove("now-seg");
      el?.classList.add("now");
      el?.closest(".seg")?.classList.add("now-seg");
      el?.closest(".seg")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      setActiveWord(el);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [player?.index, player?.clipElapsed]);

  if (books.isPending) return <section className="screen"><div className="loadline">reading the shelf…</div></section>;
  if (!bookId || !books.data?.books.some((b) => b.rendered)) {
    return (
      <section className="screen">
        <h1>Listen</h1>
        <p className="sub">Nothing rendered yet — render a book (or a few chapters) from Render &amp; Jobs, then come back to read along.</p>
      </section>
    );
  }

  const clipForRow = (row: SegmentRow) =>
    player?.clips.findIndex((c) => c.blockId === row.block_id) ?? -1;

  return (
    <section className="screen">
      <h1 style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
        Listen
        <select className="bookpick" value={bookId} onChange={(e) => setParams({ book: e.target.value })} aria-label="book">
          {books.data.books.filter((b) => b.rendered).map((b) => (
            <option key={b.book_id} value={b.book_id}>{b.title ?? b.book_id}</option>
          ))}
        </select>
      </h1>
      <p className="sub">
        Read along while it plays: the current word lights as the rendered audio runs, and clicking any word seeks the
        playback there. The transport bar below is the audio transport on this screen — volume included.
      </p>
      <div className="chapters">
        {Array.from({ length: chapterCount }, (_, i) => i + 1).map((c) => (
          <button key={c} className={`chap ${c === chapter ? "on" : ""}`} onClick={() => setChapter(c)}>
            ch {c}
          </button>
        ))}
      </div>
      {segments.isPending && <div className="loadline">setting the page…</div>}
      {segments.data && playableRows.length === 0 && (
        <div className="refusal"><span className="tag">not rendered</span><p>this chapter has no audio yet — render it from Render &amp; Jobs (chapter range works)</p></div>
      )}
      <div className="page-wrap" ref={pageRef}>
        {segments.data && (
          <div className="paper page" style={{ gridTemplateColumns: "minmax(auto,66ch) 150px" }}>
            {segments.data.segments.map((row) => {
              const ci = clipForRow(row);
              const playable = row.has_audio && row.duration_seconds !== null;
              return (
                <div key={`${row.block_id}:${row.segment_index}`} style={{ display: "contents" }}>
                  <p
                    className={`seg serif ${row.type !== "narration" ? "dlg" : ""}`}
                    data-seg={`${row.block_id}:${row.segment_index}`}
                    onClick={() => playable && ci >= 0 && player?.seekClip(ci, 0)}
                    style={{ cursor: playable ? "pointer" : "default", opacity: playable ? 1 : 0.5 }}
                  >
                    <span className="segtext">{row.text}</span>
                  </p>
                  <div className="margin">
                    <span className={`chip ${row.speaker === null ? "narr" : ""}`}>
                      {row.speaker === null ? "narration" : (row.speaker_name ?? row.speaker).toUpperCase()}
                    </span>
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
