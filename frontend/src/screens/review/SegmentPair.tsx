import { memo } from "react";

import type { EmotionVerdict } from "../../api/types";
import type { SegmentRow } from "../../api/types";
import { intensityDots } from "../../lib/emotion";

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
}) {
  const dialogueish = row.type !== "narration";
  // An unattributed quote is NOT narration — the chip says so, so the reassign popover
  // (which fixes it) is one click away instead of the quote hiding behind "narration".
  const chipLabel =
    row.speaker === null
      ? row.unattributed_quote
        ? "unattributed"
        : "narration"
      : maskedName
        ? "▮▮▮▮▮"
        : (row.speaker_name ?? row.speaker);
  return (
    <>
      <p className={`seg serif relative ${dialogueish ? "dlg" : ""}`} data-seg={`${row.block_id}:${row.segment_index}`}>
        {open || emoOpen ? <span className="hl">{row.text}</span> : row.text}
        {popover}
      </p>
      <div className="margin relative">
        <span className={`chip ${row.speaker === null ? "narr" : ""}`} onClick={() => onChip(segKey)} role="button" tabIndex={0}>
          {row.speaker === null ? chipLabel : chipLabel.toUpperCase()}
        </span>
        {emotion ? (
          <span
            className={`emochip ${emotion.label} ${emoOpen ? "on" : ""}`}
            onClick={() => onEmotion(segKey)}
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
              onClick={() => onEmotion(segKey)}
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
});
