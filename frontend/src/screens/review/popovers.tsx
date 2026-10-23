import { useState } from "react";

import { ApiError } from "../../api/client";
import { useRecordEdit } from "../../api/hooks";
import type { EmotionLabel, EmotionVerdict } from "../../api/types";
import type { CharacterSummary, SegmentRow } from "../../api/types";
import { TalkSelect } from "../../components/Select";
import { EMOTION_LABELS } from "../../lib/emotion";
import { NONE } from "./helpers";

/* -------------------------------------------------- reassign popover */

export function ReassignPopover({
  row,
  cast,
  speakerHidden = false,
  bookId,
  onClose,
}: {
  row: SegmentRow;
  /** the characters the speaker list offers (spoiler-safe leaves out the ones not met yet) */
  cast: CharacterSummary[];
  /** spoiler-safe really masks this line's current speaker (not merely: missing from `cast`,
      which is also true while the character list loads or after it failed) */
  speakerHidden?: boolean;
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
          options={[
            { value: NONE, label: "— narration —" },
            // spoiler-safe hands over only the characters met so far; a line already credited
            // to a hidden one keeps that choice without naming it. A speaker missing for any
            // other reason (the character list still loading, or failed) keeps their own name.
            ...(row.speaker !== null && !cast.some((c) => c.id === row.speaker)
              ? [
                  {
                    value: row.speaker,
                    label: speakerHidden ? "a hidden character" : (row.speaker_name ?? row.speaker),
                  },
                ]
              : []),
            ...cast.map((c) => ({ value: c.id, label: c.name })),
          ]}
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
export function EmotionPopover({
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
