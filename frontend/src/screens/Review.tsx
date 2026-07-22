import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
  useAssignment,
  useBook,
  useBooks,
  useChapterAttribution,
  useCharacters,
  useDraftAssignment,
  useEditLog,
  useSaveAssignment,
  useSegments,
  useSuggestCast,
  useUndoEdit,
  useVoices,
} from "../api/hooks";
import type { CharacterSummary } from "../api/types";
import { chapterOfBlock } from "../api/types";
import { TalkSelect } from "../components/Select";
import { Tip } from "../components/Tooltip";
import { castingDiffers, castingFromServer, castingVoicesDiffer, type CastingState } from "../lib/casting";
import { buildEmotionMap, emotionKey } from "../lib/emotion";
import { CharacterEditor } from "./review/CharacterEditor";
import { isReviewable, useFrontier } from "./review/helpers";
import { EmotionPopover, ReassignPopover } from "./review/popovers";
import { RosterRow } from "./review/RosterRow";
import { SegmentPair } from "./review/SegmentPair";
import { SuggestCastPanel } from "./review/SuggestCastPanel";
import { VoicePicker } from "./review/VoicePicker";

/* -------------------------------------------------- the screen */

export function Review() {
  const [params, setParams] = useSearchParams();
  const books = useBooks();
  const bookId = params.get("book") ?? books.data?.books.find((b) => b.attributed)?.book_id ?? null;
  const book = useBook(bookId);
  const attributed = !!book.data?.status.attributed;

  const [chapter, setChapter] = useState(1);
  const chapterCount = book.data?.chapters?.length ?? 1;

  const overview = useCharacters(bookId, attributed);
  const segments = useSegments(bookId, chapter, attributed);
  // F2/F2a: the per-segment emotion tags the model captured, now EDITABLE via the set_emotion
  // edit overlay (EmotionPopover). Keyed block_id:ordinal to line up with each row.
  const attribution = useChapterAttribution(bookId, chapter, attributed);
  const emotionMap = useMemo(
    () => buildEmotionMap(attribution.data?.report.chapters.find((c) => c.index === chapter)),
    [attribution.data, chapter],
  );
  const editLog = useEditLog(attributed ? bookId : null);
  const undo = useUndoEdit(bookId ?? "");

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
  const toggleTag = (t: string) =>
    setTagFilter((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));

  const [frontier, setFrontier] = useFrontier(bookId);
  const [spoilerSafe, setSpoilerSafe] = useState(true);
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const [selectedChar, setSelectedChar] = useState<string | null>(null);
  const [editingChar, setEditingChar] = useState<CharacterSummary | null>(null);
  const [popoverAt, setPopoverAt] = useState<string | null>(null); // `${block_id}:${idx}`
  const [emoAt, setEmoAt] = useState<string | null>(null); // emotion editor, same key shape

  const pageRef = useRef<HTMLDivElement>(null);

  // Front matter (title page, copyright) is often chapter 1 and never attributed —
  // skip forward once instead of opening the screen on a 404.
  const autoSkipped = useRef(false);
  // Chapter position and the one-shot front-matter skip are per-book state.
  useEffect(() => {
    setChapter(1);
    autoSkipped.current = false;
  }, [bookId]);
  useEffect(() => {
    if (
      !autoSkipped.current &&
      chapter === 1 &&
      segments.error instanceof ApiError &&
      segments.error.code === "not_found" &&
      chapterCount > 1
    ) {
      autoSkipped.current = true;
      setChapter(2);
    }
  }, [segments.error, chapter, chapterCount]);

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

  const jumpToLowConf = () => {
    const first = lowConfInChapter[0];
    if (!first) return;
    pageRef.current
      ?.querySelector(`[data-seg="${first.block_id}:${first.segment_index}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  if (books.isPending) return <section className="screen"><div className="loadline">reading the shelf…</div></section>;
  if (books.isError) return <section className="screen"><div className="errline">{books.error.message}</div></section>;
  if (!bookId) {
    return (
      <section className="screen">
        <h1>Character Review</h1>
        <p className="sub">No attributed books yet — run attribution from Render &amp; Jobs first.</p>
      </section>
    );
  }

  const warnings = segments.data?.edit_warnings ?? [];
  const flaggedHere = overview.data?.flagged.filter((f) => f.chapter_index === chapter) ?? [];

  return (
    <section className="screen">
      <h1 className="flex items-baseline gap-3.5">
        Character Review
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
        <div className="refusal"><span className="tag">stage_prerequisite</span><p>this book has no attribution yet — start one from Render &amp; Jobs</p></div>
      )}

      {attributed && (
        <>
          <div className="drainstrip">
            <span className="state"><i className={`led ${overview.data && overview.data.low_confidence_segments > 0 ? "warn" : "ok"}`} />review queue</span>
            <span className="mono text-[11.5px] text-ink-2">
              {overview.data?.low_confidence_segments ?? "…"} low-confidence in the book ·{" "}
              {overview.data && overview.data.unattributed_quote_segments > 0
                ? `${overview.data.unattributed_quote_segments} unattributed quote(s) · `
                : ""}
              {lowConfInChapter.length} in this chapter
            </span>
            {overview.data && (
              <Tip content="the LLM that produced this attribution">
                <span className="mono text-[10.5px] text-ink-3">
                  read by {overview.data.provider_id} · {overview.data.model_id} · {overview.data.prompt_version}
                </span>
              </Tip>
            )}
            <button className="key quiet" onClick={jumpToLowConf} disabled={lowConfInChapter.length === 0}>next ▸</button>
            <span className="flex-1" />
            <span className="tag">edits</span>
            <span className="mono text-[11.5px] text-ink-2">{editLog.data?.ops.length ?? 0}</span>
            <button
              className="key quiet"
              disabled={undo.isPending || (editLog.data?.ops.length ?? 0) === 0}
              onClick={() => undo.mutate()}
            >
              undo last
            </button>
            {undo.error && <span className="mono text-[11px] text-clip">{undo.error.message}</span>}
          </div>

          <div className="review">
            <div className="panel roster" style={{ margin: 0 }}>
              <div className="panel-h">
                <b>Cast</b>
                <span className="tag ml-2">{cast.length}</span>
                <Tip content="hide characters that first appear beyond your reading frontier">
                  <button
                    className="key quiet ml-auto px-[9px] py-[3px]"
                    onClick={() => setSpoilerSafe(!spoilerSafe)}
                  >
                    spoiler-safe {spoilerSafe ? "✓" : "✗"}
                  </button>
                </Tip>
              </div>
              <div className="frontier">
                <span className="tag">frontier</span> read through ch{" "}
                <input
                  className="frontin"
                  type="number"
                  min={1}
                  max={chapterCount}
                  value={frontier}
                  onChange={(e) => setFrontier(Number(e.target.value) || 1)}
                />{" "}
                — later characters stay hidden
              </div>
              <div className="caststrip">
                {!casting ? (
                  <>
                    <span className="text-xs text-ink-2">
                      no casting yet — auto-cast gives every character a distinct voice blend
                    </span>
                    <Tip content="preview a smart cast: every character a distinct voice, no collisions">
                      <button
                        className="key quiet ml-auto px-[9px] py-[3px]"
                        disabled={suggestCast.isPending}
                        onClick={() => suggestCast.mutate()}
                      >
                        {suggestCast.isPending ? "thinking…" : "suggest cast"}
                      </button>
                    </Tip>
                    <button
                      className="key px-3 py-[3px]"
                      disabled={draftCast.isPending}
                      onClick={() => draftCast.mutate({})}
                    >
                      {draftCast.isPending ? "casting…" : "auto-cast"}
                    </button>
                  </>
                ) : (
                  <>
                    {(["draft", "final"] as const).map((s) => (
                      <button
                        key={s}
                        className={`chap px-[9px] py-[2px] ${casting.stage === s ? "on" : ""}`}
                        onClick={() => setCasting({ ...casting, stage: s })}
                      >
                        {s}
                      </button>
                    ))}
                    <Tip content="preview a smart cast: every character a distinct voice, no collisions">
                      <button
                        className="key quiet ml-auto px-[9px] py-[3px]"
                        disabled={suggestCast.isPending}
                        onClick={() => suggestCast.mutate()}
                      >
                        {suggestCast.isPending ? "thinking…" : "suggest cast"}
                      </button>
                    </Tip>
                    <Tip content="re-run the deterministic draft — fills newly-discovered characters, keeps existing voices">
                      <button
                        className="key quiet px-[9px] py-[3px]"
                        disabled={draftCast.isPending}
                        onClick={() => draftCast.mutate({})}
                      >
                        re-draft
                      </button>
                    </Tip>
                    <button
                      className="key px-3 py-[3px]"
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
              {casting && allTags.length > 0 && (
                <div className="caststrip flex-wrap gap-1.5">
                  <Tip content="narrow every voice picker to voices carrying ALL the selected tags — tag voices in Voice Studio">
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
              {suggestCast.data && (
                <SuggestCastPanel
                  preview={suggestCast.data}
                  applying={draftCast.isPending}
                  applyError={draftCast.error instanceof ApiError ? draftCast.error : null}
                  onApply={({ recast, useLlm, confirmPaid }) =>
                    draftCast.mutate(
                      {
                        strategy: "smart",
                        recast,
                        use_llm: useLlm,
                        ...(confirmPaid ? { confirm_paid: true } : {}),
                      },
                      { onSuccess: () => suggestCast.reset() },
                    )
                  }
                  onDismiss={() => suggestCast.reset()}
                />
              )}
              {suggestCast.error && (
                <div className="refusal mx-3 my-2">
                  <span className="tag">
                    {suggestCast.error instanceof ApiError ? suggestCast.error.code : "error"}
                  </span>
                  <p>{suggestCast.error.message}</p>
                </div>
              )}
              {draftCast.data && draftCast.data.created_voice_ids.length > 0 && (
                <div className="caststrip mono text-[11px] text-ok">
                  created {draftCast.data.created_voice_ids.length} voice(s) — tune them in Voice Studio
                </div>
              )}
              {(draftCast.error || saveCast.error) && (
                <div className="refusal mx-3 my-2">
                  <span className="tag">
                    {(draftCast.error ?? saveCast.error) instanceof ApiError
                      ? ((draftCast.error ?? saveCast.error) as ApiError).code
                      : "error"}
                  </span>
                  <p>{(draftCast.error ?? saveCast.error)?.message}</p>
                </div>
              )}
              {staleAfterSave && !castingDirty && book.data?.status.rendered && (
                <div className="caststrip flex-wrap gap-2 text-[11px]">
                  <span className="tag">audio is stale</span>
                  <span className="mono text-ink-2">
                    voices changed — the rendered audio still uses the old ones.
                  </span>
                  <Link className="link" to={`/render?book=${encodeURIComponent(bookId)}`}>
                    re-render in Render &amp; Jobs →
                  </Link>
                </div>
              )}
              {overview.isPending && <div className="loadline p-3.5">reading the registry…</div>}
              {overview.isError && <div className="errline p-3.5">{overview.error.message}</div>}
              <table>
                <tbody>
                  <tr>
                    <th>character</th>
                    <th>lines</th>
                    <th>voice</th>
                  </tr>
                  <tr>
                    <td>Narration<span className="sample">the book's own voice</span></td>
                    <td>{overview.data?.narration_segments.toLocaleString()}</td>
                    <td className="vcell">
                      {casting && (
                        <VoicePicker
                          value={casting.narrator}
                          onChange={(v) => v && setCasting({ ...casting, narrator: v })}
                          voices={filteredVoices}
                          pool={allVoices}
                        />
                      )}
                    </td>
                  </tr>
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
                      selected={selectedChar === c.id}
                      onSelect={() => setSelectedChar(selectedChar === c.id ? null : c.id)}
                      voiceCell={
                        casting ? (
                          <VoicePicker
                            value={casting.map[c.id] ?? null}
                            onChange={(v) => v && setCasting({ ...casting, map: { ...casting.map, [c.id]: v } })}
                            voices={filteredVoices}
                            pool={allVoices}
                          />
                        ) : (
                          <span className="mono text-ink-3">—</span>
                        )
                      }
                    />
                  ))}
                </tbody>
              </table>
            </div>

            <div className="page-wrap" ref={pageRef}>
              <div className="flex items-center gap-2.5 mb-3.5">
                <button className="chap" disabled={chapter <= 1} onClick={() => setChapter(chapter - 1)}>‹</button>
                <span className="tag">
                  chapter {chapter} of {chapterCount}
                  {segments.data && ` · ${segments.data.title}`}
                </span>
                <button className="chap" disabled={chapter >= chapterCount} onClick={() => setChapter(chapter + 1)}>›</button>
                {emotionMap.size > 0 && (
                  <Tip content="the emotion the model tagged each line with — voiced at render only when emotion rendering is enabled">
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
              {warnings.map((w) => (
                <div className="mwarn mb-2.5" key={w}>edit overlay — {w}</div>
              ))}
              {flaggedHere.map((f) => (
                <div className="mwarn mb-2.5" key={f.block_id}>
                  flagged {f.block_id} — {f.reason}
                </div>
              ))}
              {segments.isPending && <div className="loadline">setting the page…</div>}
              {segments.isError && <div className="errline">{segments.error.message}</div>}
              {segments.data && (
                <div className="paper page">
                  {segments.data.segments.map((s) => {
                    const key = `${s.block_id}:${s.segment_index}`;
                    const low = isReviewable(s, threshold);
                    const dimmed = s.speaker !== null && maskedIds.has(s.speaker);
                    const emotion = emotionMap.get(emotionKey(s.block_id, s.segment_index)) ?? null;
                    return (
                      <SegmentPair
                        key={key}
                        segKey={key}
                        row={s}
                        low={low}
                        maskedName={dimmed}
                        emotion={emotion}
                        open={popoverAt === key}
                        onChip={handleChip}
                        popover={
                          popoverAt === key ? (
                            <ReassignPopover row={s} cast={cast} bookId={bookId} onClose={() => setPopoverAt(null)} />
                          ) : null
                        }
                        emoOpen={emoAt === key}
                        onEmotion={handleEmotion}
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
      {editingChar && (
        <CharacterEditor
          char={editingChar}
          others={cast.filter((c) => c.id !== editingChar.id && !isMasked(c))}
          onClose={() => setEditingChar(null)}
          bookId={bookId}
        />
      )}
    </section>
  );
}
