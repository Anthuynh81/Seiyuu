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
  type SuggestInput,
} from "../api/hooks";
import type { CharacterSummary, SuggestCastResponse } from "../api/types";
import { chapterOfBlock } from "../api/types";
import { NextStep } from "../components/NextStep";
import { TalkSelect } from "../components/Select";
import { Tip } from "../components/Tooltip";
import { castingDiffers, castingFromServer, castingVoicesDiffer, type CastingState } from "../lib/casting";
import { buildEmotionMap, emotionKey } from "../lib/emotion";
import { nextStep } from "../lib/nextStep";
import { CharacterEditor } from "./review/CharacterEditor";
import { isReviewable, useFrontier } from "./review/helpers";
import { EmotionPopover, ReassignPopover } from "./review/popovers";
import { ResetCastDialog } from "./review/ResetCastDialog";
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
    // no book has its speakers found yet: point at the first one that could
    const candidate = books.data?.books.find((b) => b.ingested && !b.attributed) ?? books.data?.books[0];
    return (
      <section className="screen">
        <h1>Character Review</h1>
        <p className="sub">
          {candidate
            ? "No book has its speakers found yet. Once Seiyuu has worked out who says each line, you can check it here."
            : "No books yet. Add one in the Library first."}
        </p>
        {/* this screen is about speakers: offer that step even for a book recorded in one voice */}
        <NextStep step={nextStep({ status: candidate, activeJob: candidate?.active_job, want: "attribute" })} />
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
        <div className="refusal">
          <span className="tag">speakers not found yet</span>
          <p>Seiyuu hasn't worked out who says each line in this book yet, so there is nothing to review.</p>
          <NextStep
            step={nextStep({ status: book.data.status, activeJob: book.data.active_job, want: "attribute" })}
          />
        </div>
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
                <div className="refusal mx-3 my-2">
                  <span className="tag">couldn't suggest a cast</span>
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
                    {draftCast.error instanceof ApiError && draftCast.error.code === "payment_confirmation_required"
                      ? "the AI caster is paid — approve it below to continue"
                      : draftCast.error
                        ? "couldn't apply the cast"
                        : "couldn't save casting"}
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
    </section>
  );
}
