import { memo } from "react";
import { Button } from "react-aria-components";

import type { EmotionVerdict } from "../../api/types";
import type { SegmentRow } from "../../api/types";
import { intensityDots } from "../../lib/emotion";
import { term } from "../../lib/glossary";
import { canPlay, isConfirmed } from "../../lib/reviewTools";

export const SegmentPair = memo(function SegmentPair({
  segKey,
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
  target = false,
  onConfirm,
  confirming = false,
  confirmLocked = false,
  note = null,
  onPlay,
  playing = false,
}: {
  segKey: string;
  row: SegmentRow;
  low: boolean;
  maskedName: boolean;
  emotion: EmotionVerdict | null;
  open: boolean;
  onChip: (key: string) => void;
  popover: React.ReactNode;
  emoOpen: boolean;
  onEmotion: (key: string) => void;
  emotionPopover: React.ReactNode;
  /** the line a deep link or "Next to check" landed on: marked, and focusable so the page
      can move keyboard focus to it */
  target?: boolean;
  /** "Looks right ✓": keep this speaker (a same-speaker reassign), offered on lines to check */
  onConfirm?: (row: SegmentRow) => void;
  confirming?: boolean;
  /** another line's "Looks right" is saving: one at a time, so a refusal lands on its own line */
  confirmLocked?: boolean;
  /** a message under this line's margin (a refused "Looks right") */
  note?: React.ReactNode;
  /** ▶: play this line's recording in the shared player (only offered when it has audio) */
  onPlay?: (row: SegmentRow) => void;
  playing?: boolean;
}) {
  const dialogueish = row.type !== "narration";
  const thought = row.type === "thought";
  // An unattributed quote is NOT narration — the chip says so, so the reassign popover
  // (which fixes it) is one click away instead of the quote hiding behind "narration". Once
  // someone confirmed the narrator reads it (confidence 1.0) it is plain narration again.
  const chipLabel =
    row.speaker === null
      ? row.unattributed_quote && !isConfirmed(row)
        ? term("unattributed")
        : "narration"
      : maskedName
        ? "▮▮▮▮▮"
        : (row.speaker_name ?? row.speaker);
  const who = row.speaker === null ? "the narrator" : maskedName ? "this hidden speaker" : chipLabel;
  const showConfirm = low && onConfirm !== undefined;
  const showPlay = onPlay !== undefined && canPlay(row);
  return (
    <>
      <p
        className={`seg serif relative ${dialogueish ? "dlg" : ""} ${thought ? "thought" : ""} ${target ? "target" : ""}`}
        data-seg={`${row.block_id}:${row.segment_index}`}
        tabIndex={target ? -1 : undefined}
      >
        {open || emoOpen ? <span className="hl">{row.text}</span> : row.text}
        {popover}
      </p>
      <div className="margin relative">
        {/* react-aria Buttons: Enter and Space press them like a click */}
        <Button className={`chip ${row.speaker === null ? "narr" : ""}`} onPress={() => onChip(segKey)}>
          {row.speaker === null ? chipLabel : chipLabel.toUpperCase()}
        </Button>
        {thought && (
          <span className="thoughttag" title="an inner thought (display only)">
            thought
          </span>
        )}
        {emotion ? (
          <Button className={`emochip ${emotion.label} ${emoOpen ? "on" : ""}`} onPress={() => onEmotion(segKey)}>
            <span title="change or clear this line's emotion — voiced only when a recording has emotion turned on">
              {emotion.label} {intensityDots(emotion.intensity)}
            </span>
          </Button>
        ) : (
          dialogueish && (
            <button
              className="emoadd"
              onClick={() => onEmotion(segKey)}
              title="tag this line with an emotion (voiced only when a recording has emotion turned on)"
            >
              + emotion
            </button>
          )
        )}
        <span className={`conf ${low ? "low" : ""}`}>
          conf {row.confidence.toFixed(2)}
          {low && " · in review"}
        </span>
        {(showConfirm || showPlay) && (
          <span className="lineacts">
            {showConfirm && (
              <button
                className="linekey"
                // the visible words say what the click records: for a quote with no speaker,
                // that the narrator reads it (the accessible name starts with the same words)
                aria-label={row.speaker === null ? "Narrator reads it: keep the narrator for this quote" : `Looks right: keep ${who}`}
                title={`keep ${who} for this line — it stops showing as one to check`}
                aria-disabled={confirming || confirmLocked || undefined}
                aria-busy={confirming || undefined}
                onClick={() => {
                  if (!confirming && !confirmLocked) onConfirm(row);
                }}
              >
                {confirming ? "saving…" : row.speaker === null ? "Narrator reads it ✓" : "Looks right ✓"}
              </button>
            )}
            {showPlay && (
              <button
                className="linekey"
                aria-label={playing ? "pause this line's recording" : "play this line's recording"}
                title={playing ? "pause" : "play the recording of this line"}
                onClick={() => onPlay(row)}
              >
                {playing ? "❚❚" : "▶"}
              </button>
            )}
          </span>
        )}
        {note}
        {emotionPopover}
      </div>
    </>
  );
});
