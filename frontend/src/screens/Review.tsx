import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ApiError } from "../api/client";
import {
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
  useUndoEdit,
  useVoices,
} from "../api/hooks";
import type { EmotionVerdict, SuggestCastResponse } from "../api/types";
import type { CharacterSummary, SegmentRow, VoiceOut } from "../api/types";
import { chapterOfBlock } from "../api/types";
import { castingDiffers, castingFromServer, type CastingState } from "../lib/casting";
import { buildEmotionMap, emotionKey, intensityDots } from "../lib/emotion";

/* -------------------------------------------------- frontier (localStorage, per book) */

function useFrontier(bookId: string | null): [number, (n: number) => void] {
  const key = `seiyuu.frontier.${bookId}`;
  const [value, setValue] = useState(() => Number(localStorage.getItem(key)) || 1);
  return [
    value,
    (n: number) => {
      setValue(n);
      localStorage.setItem(key, String(n));
    },
  ];
}

/* -------------------------------------------------- roster */

function RosterRow({
  char,
  masked,
  onReveal,
  onEdit,
  selected,
  onSelect,
  voiceCell,
}: {
  char: CharacterSummary;
  masked: boolean;
  onReveal: () => void;
  onEdit: () => void;
  selected: boolean;
  onSelect: () => void;
  voiceCell: React.ReactNode;
}) {
  const debutChapter = chapterOfBlock(char.first_appearance);
  if (masked) {
    return (
      <tr className="masked">
        <td>
          <span className="mask">{"▮".repeat(Math.min(Math.max(char.name.length, 5), 14))}</span>
          <span className="sample">
            enters ch {debutChapter} ·{" "}
            <a
              className="link"
              href="#"
              style={{ fontStyle: "normal" }}
              onClick={(e) => {
                e.preventDefault();
                onReveal();
              }}
            >
              reveal
            </a>
          </span>
        </td>
        <td>{char.line_count.toLocaleString()}</td>
        <td className="vcell">
          {/* auto voices are named after their characters — a visible picker would leak
              the very name the mask hides */}
          <span className="mono" style={{ color: "var(--ink-3)" }}>▮▮</span>
        </td>
      </tr>
    );
  }
  return (
    <tr className={selected ? "sel" : ""} onClick={onSelect}>
      <td>
        {char.name}
        <button
          className="rowedit"
          title="rename / merge"
          onClick={(e) => {
            e.stopPropagation();
            onEdit();
          }}
        >
          ✎
        </button>
        {char.sample_lines[0] && <span className="sample">{char.sample_lines[0]}</span>}
      </td>
      <td>{char.line_count.toLocaleString()}</td>
      <td className="vcell" onClick={(e) => e.stopPropagation()}>{voiceCell}</td>
    </tr>
  );
}

/* -------------------------------------------------- casting */

/** Preview of the smart caster: distinct voice per character. Applying is NOT a silent
    no-op — it spells out how many voices it creates and, if some are already cast, offers an
    explicit re-cast (which re-renders that audio) instead of quietly skipping them. */
function SuggestCastPanel({
  preview,
  applying,
  onApply,
  onDismiss,
}: {
  preview: SuggestCastResponse;
  applying: boolean;
  onApply: (recast: boolean) => void;
  onDismiss: () => void;
}) {
  const [recast, setRecast] = useState(false);
  const create = preview.would_create_voice_ids.length;
  const existing = preview.would_recast_voice_ids.length;
  const noop = create === 0 && !recast; // every character already cast, recast off
  return (
    <div className="caststrip" style={{ flexWrap: "wrap", gap: 8, background: "var(--surface-2)" }}>
      <span className="tag">smart cast</span>
      <span className="mono" style={{ fontSize: 11, color: "var(--ink-2)" }}>
        every character a distinct voice · {create} new
        {existing > 0 ? ` · ${existing} already cast` : ""}
      </span>
      {existing > 0 && (
        <label className="mono" style={{ fontSize: 11, display: "flex", gap: 5, alignItems: "center" }}>
          <input type="checkbox" checked={recast} onChange={(e) => setRecast(e.target.checked)} />
          re-cast the {existing} existing (re-renders their audio)
        </label>
      )}
      <span style={{ flex: 1 }} />
      <button className="key quiet" style={{ padding: "3px 9px" }} onClick={onDismiss}>
        dismiss
      </button>
      <button
        className="key"
        style={{ padding: "3px 12px" }}
        disabled={applying || noop}
        title={noop ? "every character is already cast — enable re-cast to overwrite" : undefined}
        onClick={() => onApply(recast)}
      >
        {applying ? "applying…" : noop ? "nothing to apply" : "apply cast"}
      </button>
    </div>
  );
}

function VoicePicker({
  value,
  onChange,
  voices,
  allowOwn,
}: {
  value: string | null;
  onChange: (v: string | null) => void;
  voices: VoiceOut[];
  allowOwn?: boolean; // the thought-voice "speaker's own" option
}) {
  return (
    <select
      className="vpick"
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
    >
      {allowOwn ? <option value="">speaker's own</option> : value === null && <option value="">— uncast —</option>}
      {voices.map((v) => (
        <option key={v.voice_id} value={v.voice_id}>
          {v.name} · {v.engine}
        </option>
      ))}
    </select>
  );
}

function CharacterEditor({
  char,
  others,
  onClose,
  bookId,
}: {
  char: CharacterSummary;
  others: CharacterSummary[];
  onClose: () => void;
  bookId: string;
}) {
  const record = useRecordEdit(bookId);
  const [name, setName] = useState(char.name);
  const [mergeInto, setMergeInto] = useState("");
  const error = record.error instanceof ApiError ? record.error.message : record.error?.message;
  return (
    <div className="overlay on" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog">
        <div className="dh">
          <b>Edit character — {char.name}</b>
          <button className="key quiet" onClick={onClose}>esc</button>
        </div>
        <div className="db">
          <label>canonical name</label>
          <div style={{ display: "flex", gap: 8 }}>
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
            <button
              className="key"
              disabled={record.isPending || name.trim() === "" || name === char.name}
              onClick={() =>
                record.mutate({ op: "rename", character_id: char.id, new_name: name.trim() }, { onSuccess: onClose })
              }
            >
              rename
            </button>
          </div>
          {char.aliases.length > 0 && (
            <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 6 }}>
              also seen as: {char.aliases.join(", ")}
            </div>
          )}
          <label>merge into another character (this one's lines move there)</label>
          <div style={{ display: "flex", gap: 8 }}>
            <select value={mergeInto} onChange={(e) => setMergeInto(e.target.value)}>
              <option value="">— choose —</option>
              {others.map((o) => (
                <option key={o.id} value={o.id}>{o.name}</option>
              ))}
            </select>
            <button
              className="key"
              disabled={record.isPending || !mergeInto}
              onClick={() =>
                record.mutate({ op: "merge", loser_id: char.id, winner_id: mergeInto }, { onSuccess: onClose })
              }
            >
              merge
            </button>
          </div>
          {error && <div className="errline" style={{ marginTop: 12 }}>{error}</div>}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------- reassign popover */

function ReassignPopover({
  row,
  cast,
  bookId,
  onClose,
}: {
  row: SegmentRow;
  cast: CharacterSummary[];
  bookId: string;
  onClose: () => void;
}) {
  const record = useRecordEdit(bookId);
  const [speaker, setSpeaker] = useState(row.speaker ?? "");
  const anchor = row.text.replace(/\s+/g, " ").trim().slice(0, 58);
  const error = record.error instanceof ApiError ? record.error.message : record.error?.message;
  return (
    <span className="popover" onClick={(e) => e.stopPropagation()}>
      <span className="tag">reassign · {row.block_id} [{row.segment_index}]</span>
      <span className="row">
        <select value={speaker} onChange={(e) => setSpeaker(e.target.value)}>
          <option value="">— narration —</option>
          {cast.map((c) => (
            <option key={c.id} value={c.id}>{c.name}</option>
          ))}
        </select>
      </span>
      <span style={{ display: "block", fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)" }}>
        anchor: {anchor}…
      </span>
      {error && <span style={{ display: "block", fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--clip)", marginTop: 6 }}>{error}</span>}
      <span className="acts">
        <button className="key quiet" onClick={onClose}>cancel</button>
        <button
          className="key"
          disabled={record.isPending}
          onClick={() =>
            record.mutate(
              { op: "reassign", block_id: row.block_id, segment_index: row.segment_index, speaker: speaker || null },
              { onSuccess: onClose },
            )
          }
        >
          {record.isPending ? "recording…" : "record edit"}
        </button>
      </span>
    </span>
  );
}

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
  // F2: the per-segment emotion tags the model captured (read-only here — override needs a
  // backend edit op that doesn't exist yet). Keyed block_id:ordinal to line up with each row.
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

  const [frontier, setFrontier] = useFrontier(bookId);
  const [spoilerSafe, setSpoilerSafe] = useState(true);
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const [selectedChar, setSelectedChar] = useState<string | null>(null);
  const [editingChar, setEditingChar] = useState<CharacterSummary | null>(null);
  const [popoverAt, setPopoverAt] = useState<string | null>(null); // `${block_id}:${idx}`

  const pageRef = useRef<HTMLDivElement>(null);

  // Front matter (title page, copyright) is often chapter 1 and never attributed —
  // skip forward once instead of opening the screen on a 404.
  const autoSkipped = useRef(false);
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
    () => segments.data?.segments.filter((s) => s.speaker !== null && s.confidence < threshold) ?? [],
    [segments.data, threshold],
  );

  const isMasked = (c: CharacterSummary) => {
    if (!spoilerSafe || revealed.has(c.id)) return false;
    const debut = chapterOfBlock(c.first_appearance);
    return debut !== null && debut > frontier;
  };

  const jumpToLowConf = () => {
    const first = lowConfInChapter[0];
    if (!first) return;
    pageRef.current
      ?.querySelector(`[data-seg="${first.block_id}:${first.segment_index}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  if (books.isPending) return <section className="screen"><div className="loadline">reading the shelf…</div></section>;
  if (!bookId) {
    return (
      <section className="screen">
        <h1>Character Review</h1>
        <p className="sub">No attributed books yet — run attribution from Render &amp; Jobs first.</p>
      </section>
    );
  }

  const cast = overview.data?.characters ?? [];
  const warnings = segments.data?.edit_warnings ?? [];
  const flaggedHere = overview.data?.flagged.filter((f) => f.chapter_index === chapter) ?? [];

  return (
    <section className="screen">
      <h1 style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
        Character Review
        <select className="bookpick" value={bookId} onChange={(e) => setParams({ book: e.target.value })} aria-label="book">
          {books.data?.books.filter((b) => b.attributed).map((b) => (
            <option key={b.book_id} value={b.book_id}>{b.title ?? b.book_id}</option>
          ))}
        </select>
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
            <span className="mono" style={{ fontSize: 11.5, color: "var(--ink-2)" }}>
              {overview.data?.low_confidence_segments ?? "…"} low-confidence in the book · {lowConfInChapter.length} in this chapter
            </span>
            {overview.data && (
              <span
                className="mono"
                style={{ fontSize: 10.5, color: "var(--ink-3)" }}
                title="the LLM that produced this attribution"
              >
                read by {overview.data.provider_id} · {overview.data.model_id} · {overview.data.prompt_version}
              </span>
            )}
            <button className="key quiet" onClick={jumpToLowConf} disabled={lowConfInChapter.length === 0}>next ▸</button>
            <span style={{ flex: 1 }} />
            <span className="tag">edits</span>
            <span className="mono" style={{ fontSize: 11.5, color: "var(--ink-2)" }}>{editLog.data?.ops.length ?? 0}</span>
            <button
              className="key quiet"
              disabled={undo.isPending || (editLog.data?.ops.length ?? 0) === 0}
              onClick={() => undo.mutate()}
            >
              undo last
            </button>
            {undo.error && <span className="mono" style={{ color: "var(--clip)", fontSize: 11 }}>{undo.error.message}</span>}
          </div>

          <div className="review">
            <div className="panel roster" style={{ margin: 0 }}>
              <div className="panel-h">
                <b>Cast</b>
                <span className="tag" style={{ marginLeft: 8 }}>{cast.length}</span>
                <button
                  className="key quiet"
                  style={{ marginLeft: "auto", padding: "3px 9px" }}
                  title="hide characters that first appear beyond your reading frontier"
                  onClick={() => setSpoilerSafe(!spoilerSafe)}
                >
                  spoiler-safe {spoilerSafe ? "✓" : "✗"}
                </button>
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
                    <span style={{ color: "var(--ink-2)", fontSize: 12 }}>
                      no casting yet — auto-cast gives every character a distinct voice blend
                    </span>
                    <button
                      className="key quiet"
                      style={{ marginLeft: "auto", padding: "3px 9px" }}
                      title="preview a smart cast: every character a distinct voice, no collisions"
                      disabled={suggestCast.isPending}
                      onClick={() => suggestCast.mutate()}
                    >
                      {suggestCast.isPending ? "thinking…" : "suggest cast"}
                    </button>
                    <button
                      className="key"
                      style={{ padding: "3px 12px" }}
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
                        className={`chap ${casting.stage === s ? "on" : ""}`}
                        style={{ padding: "2px 9px" }}
                        onClick={() => setCasting({ ...casting, stage: s })}
                      >
                        {s}
                      </button>
                    ))}
                    <button
                      className="key quiet"
                      style={{ marginLeft: "auto", padding: "3px 9px" }}
                      title="preview a smart cast: every character a distinct voice, no collisions"
                      disabled={suggestCast.isPending}
                      onClick={() => suggestCast.mutate()}
                    >
                      {suggestCast.isPending ? "thinking…" : "suggest cast"}
                    </button>
                    <button
                      className="key quiet"
                      style={{ padding: "3px 9px" }}
                      title="re-run the deterministic draft — fills newly-discovered characters, keeps existing voices"
                      disabled={draftCast.isPending}
                      onClick={() => draftCast.mutate({})}
                    >
                      re-draft
                    </button>
                    <button
                      className="key"
                      style={{ padding: "3px 12px" }}
                      disabled={!castingDirty || saveCast.isPending}
                      onClick={() =>
                        saveCast.mutate({
                          stage: casting.stage,
                          narrator_voice_id: casting.narrator,
                          assignments: casting.map,
                          thought_voice_id: casting.thought,
                        })
                      }
                    >
                      {saveCast.isPending ? "saving…" : castingDirty ? "save casting" : "saved ✓"}
                    </button>
                  </>
                )}
              </div>
              {suggestCast.data && (
                <SuggestCastPanel
                  preview={suggestCast.data}
                  applying={draftCast.isPending}
                  onApply={(recast) =>
                    draftCast.mutate(
                      { strategy: "smart", recast },
                      { onSuccess: () => suggestCast.reset() },
                    )
                  }
                  onDismiss={() => suggestCast.reset()}
                />
              )}
              {suggestCast.error && (
                <div className="refusal" style={{ margin: "8px 12px" }}>
                  <span className="tag">
                    {suggestCast.error instanceof ApiError ? suggestCast.error.code : "error"}
                  </span>
                  <p>{suggestCast.error.message}</p>
                </div>
              )}
              {draftCast.data && draftCast.data.created_voice_ids.length > 0 && (
                <div className="caststrip" style={{ color: "var(--ok)", fontFamily: "var(--mono)", fontSize: 11 }}>
                  created {draftCast.data.created_voice_ids.length} voice(s) — tune them in Voice Studio
                </div>
              )}
              {(draftCast.error || saveCast.error) && (
                <div className="refusal" style={{ margin: "8px 12px" }}>
                  <span className="tag">
                    {(draftCast.error ?? saveCast.error) instanceof ApiError
                      ? ((draftCast.error ?? saveCast.error) as ApiError).code
                      : "error"}
                  </span>
                  <p>{(draftCast.error ?? saveCast.error)?.message}</p>
                </div>
              )}
              {overview.isPending && <div className="loadline" style={{ padding: 14 }}>reading the registry…</div>}
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
                          voices={voicesQ.data?.voices ?? []}
                        />
                      )}
                    </td>
                  </tr>
                  {casting && (
                    <tr>
                      <td style={{ color: "var(--ink-2)" }}>
                        Thoughts<span className="sample">inner voice for thought segments</span>
                      </td>
                      <td>—</td>
                      <td className="vcell">
                        <VoicePicker
                          value={casting.thought}
                          onChange={(v) => setCasting({ ...casting, thought: v })}
                          voices={voicesQ.data?.voices ?? []}
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
                            voices={voicesQ.data?.voices ?? []}
                          />
                        ) : (
                          <span className="mono" style={{ color: "var(--ink-3)" }}>—</span>
                        )
                      }
                    />
                  ))}
                </tbody>
              </table>
            </div>

            <div className="page-wrap" ref={pageRef}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <button className="chap" disabled={chapter <= 1} onClick={() => setChapter(chapter - 1)}>‹</button>
                <span className="tag">
                  chapter {chapter} of {chapterCount}
                  {segments.data && ` · ${segments.data.title}`}
                </span>
                <button className="chap" disabled={chapter >= chapterCount} onClick={() => setChapter(chapter + 1)}>›</button>
                {emotionMap.size > 0 && (
                  <span
                    className="mono"
                    style={{ fontSize: 10.5, color: "var(--ink-3)" }}
                    title="the emotion the model tagged each line with — voiced at render only when emotion rendering is enabled"
                  >
                    {emotionMap.size} emotion tag(s) in this chapter
                  </span>
                )}
                {warnings.length > 0 && (
                  <span className="mono" style={{ fontSize: 10.5, color: "var(--caution)", marginLeft: "auto" }}>
                    {warnings.length} edit warning(s)
                  </span>
                )}
              </div>
              {warnings.map((w) => (
                <div className="mwarn" key={w} style={{ margin: "0 0 10px" }}>edit overlay — {w}</div>
              ))}
              {flaggedHere.map((f) => (
                <div className="mwarn" key={f.block_id} style={{ margin: "0 0 10px" }}>
                  flagged {f.block_id} — {f.reason}
                </div>
              ))}
              {segments.isPending && <div className="loadline">setting the page…</div>}
              {segments.isError && <div className="errline">{segments.error.message}</div>}
              {segments.data && (
                <div className="paper page">
                  {segments.data.segments.map((s) => {
                    const key = `${s.block_id}:${s.segment_index}`;
                    const low = s.speaker !== null && s.confidence < threshold;
                    const dimmed = spoilerSafe && s.speaker !== null && cast.some((c) => c.id === s.speaker && isMasked(c));
                    return (
                      <SegmentPair
                        key={key}
                        row={s}
                        low={low}
                        maskedName={dimmed}
                        emotion={emotionMap.get(emotionKey(s.block_id, s.segment_index)) ?? null}
                        open={popoverAt === key}
                        onChip={() => setPopoverAt(popoverAt === key ? null : key)}
                        popover={
                          popoverAt === key ? (
                            <ReassignPopover row={s} cast={cast} bookId={bookId} onClose={() => setPopoverAt(null)} />
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

function SegmentPair({
  row,
  low,
  maskedName,
  emotion,
  open,
  onChip,
  popover,
}: {
  row: SegmentRow;
  low: boolean;
  maskedName: boolean;
  emotion: EmotionVerdict | null;
  open: boolean;
  onChip: () => void;
  popover: React.ReactNode;
}) {
  const dialogueish = row.type !== "narration";
  const chipLabel = row.speaker === null ? "narration" : maskedName ? "▮▮▮▮▮" : (row.speaker_name ?? row.speaker);
  return (
    <>
      <p className={`seg serif ${dialogueish ? "dlg" : ""}`} data-seg={`${row.block_id}:${row.segment_index}`} style={{ position: "relative" }}>
        {open ? <span className="hl">{row.text}</span> : row.text}
        {popover}
      </p>
      <div className="margin">
        <span className={`chip ${row.speaker === null ? "narr" : ""}`} onClick={onChip} role="button" tabIndex={0}>
          {row.speaker === null ? chipLabel : chipLabel.toUpperCase()}
        </span>
        {emotion && (
          <span
            className={`emochip ${emotion.label}`}
            title={`emotion tag from attribution · intensity ${emotion.intensity}/3 — applied to the voice only when emotion rendering is enabled`}
          >
            {emotion.label} {intensityDots(emotion.intensity)}
          </span>
        )}
        <span className={`conf ${low ? "low" : ""}`}>
          conf {row.confidence.toFixed(2)}
          {low && " · in review"}
        </span>
      </div>
    </>
  );
}
