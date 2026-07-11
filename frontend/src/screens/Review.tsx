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
import type { EmotionLabel, EmotionVerdict, SuggestCastResponse } from "../api/types";
import type { CharacterSummary, SegmentRow, VoiceOut } from "../api/types";
import { chapterOfBlock } from "../api/types";
import { TalkDialog } from "../components/Dialog";
import { TalkSelect } from "../components/Select";
import { Tip } from "../components/Tooltip";
import { castingDiffers, castingFromServer, type CastingState } from "../lib/casting";
import { buildEmotionMap, EMOTION_LABELS, emotionKey, intensityDots } from "../lib/emotion";

/** TalkSelect keys are strings; this sentinel stands in for "no voice / narration". */
const NONE = "__none__";

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
              className="link not-italic"
              href="#"
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
          <span className="mono text-ink-3">▮▮</span>
        </td>
      </tr>
    );
  }
  return (
    <tr className={selected ? "sel" : ""} onClick={onSelect}>
      <td>
        {char.name}
        <Tip content="rename / merge">
          <button
            className="rowedit"
            onClick={(e) => {
              e.stopPropagation();
              onEdit();
            }}
          >
            ✎
          </button>
        </Tip>
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
  onApply: (opts: { recast: boolean; useLlm: boolean }) => void;
  onDismiss: () => void;
}) {
  const [recast, setRecast] = useState(false);
  const [useLlm, setUseLlm] = useState(false);
  const create = preview.would_create_voice_ids.length;
  const existing = preview.would_recast_voice_ids.length;
  const noop = create === 0 && !recast; // every character already cast, recast off
  return (
    <div className="caststrip flex-wrap gap-2 bg-console-hi">
      <span className="tag">smart cast</span>
      <span className="mono text-[11px] text-ink-2">
        every character a distinct voice · {create} new
        {existing > 0 ? ` · ${existing} already cast` : ""}
      </span>
      {existing > 0 && (
        <label className="mono flex items-center gap-[5px] text-[11px]">
          <input type="checkbox" checked={recast} onChange={(e) => setRecast(e.target.checked)} />
          re-cast the {existing} existing (re-renders their audio)
        </label>
      )}
      <label
        className="mono flex items-center gap-[5px] text-[11px]"
        title="ask the LLM for per-character voice-trait hints. Voices stay distinct — the hint only nudges which one each character gets."
      >
        <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
        ✨ use AI trait hints
      </label>
      <span className="flex-1" />
      <button className="key quiet px-[9px] py-[3px]" onClick={onDismiss}>
        dismiss
      </button>
      <button
        className="key px-3 py-[3px]"
        disabled={applying || noop}
        title={noop ? "every character is already cast — enable re-cast to overwrite" : undefined}
        onClick={() => onApply({ recast, useLlm })}
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
  const options = [
    ...(allowOwn ? [{ value: NONE, label: "speaker's own" }] : value === null ? [{ value: NONE, label: "— uncast —" }] : []),
    ...voices.map((v) => ({ value: v.voice_id, label: `${v.name} · ${v.engine}` })),
  ];
  return (
    <TalkSelect
      className="vpick"
      ariaLabel="voice"
      value={value ?? NONE}
      onChange={(v) => onChange(v === NONE ? null : v)}
      options={options}
    />
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
  const [mergeInto, setMergeInto] = useState(NONE);
  const error = record.error instanceof ApiError ? record.error.message : record.error?.message;
  return (
    <TalkDialog title={`Edit character — ${char.name}`} onClose={onClose}>
      <label>canonical name</label>
      <div className="flex gap-2">
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
        <div className="mono mt-1.5 text-[11px] text-ink-3">
          also seen as: {char.aliases.join(", ")}
        </div>
      )}
      <label>merge into another character (this one's lines move there)</label>
      <div className="flex gap-2">
        <TalkSelect
          className="flex-1"
          ariaLabel="merge into"
          value={mergeInto}
          onChange={setMergeInto}
          options={[{ value: NONE, label: "— choose —" }, ...others.map((o) => ({ value: o.id, label: o.name }))]}
        />
        <button
          className="key"
          disabled={record.isPending || mergeInto === NONE}
          onClick={() =>
            record.mutate({ op: "merge", loser_id: char.id, winner_id: mergeInto }, { onSuccess: onClose })
          }
        >
          merge
        </button>
      </div>
      {error && <div className="errline mt-3">{error}</div>}
    </TalkDialog>
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
  const [speaker, setSpeaker] = useState(row.speaker ?? NONE);
  const anchor = row.text.replace(/\s+/g, " ").trim().slice(0, 58);
  const error = record.error instanceof ApiError ? record.error.message : record.error?.message;
  return (
    <span className="popover" onClick={(e) => e.stopPropagation()}>
      <span className="tag">reassign · {row.block_id} [{row.segment_index}]</span>
      <span className="row">
        <TalkSelect
          className="flex-1"
          ariaLabel="speaker"
          value={speaker}
          onChange={setSpeaker}
          options={[{ value: NONE, label: "— narration —" }, ...cast.map((c) => ({ value: c.id, label: c.name }))]}
        />
      </span>
      <span className="mono block text-[10px] text-ink-3">anchor: {anchor}…</span>
      {error && <span className="mono block mt-1.5 text-[10.5px] text-clip">{error}</span>}
      <span className="acts">
        <button className="key quiet" onClick={onClose}>cancel</button>
        <button
          className="key"
          disabled={record.isPending}
          onClick={() =>
            record.mutate(
              {
                op: "reassign",
                block_id: row.block_id,
                segment_index: row.segment_index,
                speaker: speaker === NONE ? null : speaker,
              },
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

/* -------------------------------------------------- emotion editor popover */

/** F2a: set or clear ONE segment's emotion as a durable edit overlay (same affordance as the
    reassign popover). Choosing a label + intensity records a verdict; "clear" records null. */
function EmotionPopover({
  row,
  current,
  bookId,
  onClose,
}: {
  row: SegmentRow;
  current: EmotionVerdict | null;
  bookId: string;
  onClose: () => void;
}) {
  const record = useRecordEdit(bookId);
  const settable = EMOTION_LABELS.filter((l) => l !== "neutral"); // neutral == no override == clear
  const [label, setLabel] = useState<EmotionLabel>(
    current && current.label !== "neutral" ? current.label : settable[0],
  );
  const [intensity, setIntensity] = useState(current?.intensity ?? 2);
  const error = record.error instanceof ApiError ? record.error.message : record.error?.message;
  const save = (emotion: EmotionVerdict | null) =>
    record.mutate(
      { op: "set_emotion", block_id: row.block_id, segment_index: row.segment_index, emotion },
      { onSuccess: onClose },
    );
  return (
    <span className="popover" onClick={(e) => e.stopPropagation()}>
      <span className="tag">emotion · {row.block_id} [{row.segment_index}]</span>
      <span className="row flex gap-1.5">
        <TalkSelect
          className="flex-1"
          ariaLabel="emotion label"
          value={label}
          onChange={(v) => setLabel(v as EmotionLabel)}
          options={settable.map((l) => ({ value: l, label: l }))}
        />
        <TalkSelect
          className="flex-1"
          ariaLabel="intensity"
          value={String(intensity)}
          onChange={(v) => setIntensity(Number(v))}
          options={[1, 2, 3].map((n) => ({ value: String(n), label: `intensity ${n}` }))}
        />
      </span>
      <span className="mono block text-[10px] text-ink-3">
        voiced at render only when emotion rendering is on
      </span>
      {error && <span className="mono block mt-1.5 text-[10.5px] text-clip">{error}</span>}
      <span className="acts">
        {current && (
          <button className="key quiet" disabled={record.isPending} onClick={() => save(null)}>
            clear
          </button>
        )}
        <button className="key quiet" onClick={onClose}>cancel</button>
        <button className="key" disabled={record.isPending} onClick={() => save({ label, intensity })}>
          {record.isPending ? "recording…" : "set emotion"}
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
              {overview.data?.low_confidence_segments ?? "…"} low-confidence in the book · {lowConfInChapter.length} in this chapter
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
                  onApply={({ recast, useLlm }) =>
                    draftCast.mutate(
                      { strategy: "smart", recast, use_llm: useLlm },
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
              {overview.isPending && <div className="loadline p-3.5">reading the registry…</div>}
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
                      <td className="text-ink-2">
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
                    const low = s.speaker !== null && s.confidence < threshold;
                    const dimmed = spoilerSafe && s.speaker !== null && cast.some((c) => c.id === s.speaker && isMasked(c));
                    const emotion = emotionMap.get(emotionKey(s.block_id, s.segment_index)) ?? null;
                    return (
                      <SegmentPair
                        key={key}
                        row={s}
                        low={low}
                        maskedName={dimmed}
                        emotion={emotion}
                        open={popoverAt === key}
                        onChip={() => {
                          setEmoAt(null);
                          setPopoverAt(popoverAt === key ? null : key);
                        }}
                        popover={
                          popoverAt === key ? (
                            <ReassignPopover row={s} cast={cast} bookId={bookId} onClose={() => setPopoverAt(null)} />
                          ) : null
                        }
                        emoOpen={emoAt === key}
                        onEmotion={() => {
                          setPopoverAt(null);
                          setEmoAt(emoAt === key ? null : key);
                        }}
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

function SegmentPair({
  row,
  low,
  maskedName,
  emotion,
  open,
  onChip,
  popover,
  emoOpen,
  onEmotion,
  emotionPopover,
}: {
  row: SegmentRow;
  low: boolean;
  maskedName: boolean;
  emotion: EmotionVerdict | null;
  open: boolean;
  onChip: () => void;
  popover: React.ReactNode;
  emoOpen: boolean;
  onEmotion: () => void;
  emotionPopover: React.ReactNode;
}) {
  const dialogueish = row.type !== "narration";
  const chipLabel = row.speaker === null ? "narration" : maskedName ? "▮▮▮▮▮" : (row.speaker_name ?? row.speaker);
  return (
    <>
      <p className={`seg serif relative ${dialogueish ? "dlg" : ""}`} data-seg={`${row.block_id}:${row.segment_index}`}>
        {open || emoOpen ? <span className="hl">{row.text}</span> : row.text}
        {popover}
      </p>
      <div className="margin relative">
        <span className={`chip ${row.speaker === null ? "narr" : ""}`} onClick={onChip} role="button" tabIndex={0}>
          {row.speaker === null ? chipLabel : chipLabel.toUpperCase()}
        </span>
        {emotion ? (
          <span
            className={`emochip ${emotion.label} ${emoOpen ? "on" : ""}`}
            onClick={onEmotion}
            role="button"
            tabIndex={0}
            title="click to change or clear this line's emotion — voiced at render only when emotion rendering is enabled"
          >
            {emotion.label} {intensityDots(emotion.intensity)}
          </span>
        ) : (
          dialogueish && (
            <button
              className="emoadd"
              onClick={onEmotion}
              title="tag this line with an emotion (voiced at render when emotion rendering is on)"
            >
              + emotion
            </button>
          )
        )}
        <span className={`conf ${low ? "low" : ""}`}>
          conf {row.confidence.toFixed(2)}
          {low && " · in review"}
        </span>
        {emotionPopover}
      </div>
    </>
  );
}
