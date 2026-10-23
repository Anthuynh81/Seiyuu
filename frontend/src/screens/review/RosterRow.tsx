import type { CharacterSummary, VoiceOut } from "../../api/types";
import { chapterOfBlock } from "../../api/types";
import { Tip } from "../../components/Tooltip";
import { auditionLine } from "../../lib/audition";
import { PlayableRow } from "./PlayableRow";

/* -------------------------------------------------- roster */

export function RosterRow({
  char,
  masked,
  onReveal,
  onEdit,
  selected,
  onSelect,
  voiceCell,
  voice,
}: {
  char: CharacterSummary;
  masked: boolean;
  onReveal: () => void;
  onEdit: () => void;
  /** this character's lines are the only ones on the page */
  selected: boolean;
  /** toggle the page's one-character filter */
  onSelect: () => void;
  voiceCell: React.ReactNode;
  /** the voice cast for this character right now (unsaved picks included) — its ▶ plays it
      speaking the character's first sample line */
  voice?: VoiceOut;
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
    <PlayableRow
      className={selected ? "sel" : ""}
      onClick={onSelect}
      colSpan={3}
      voice={voice}
      text={auditionLine(char.sample_lines[0])}
      who={`${char.name}'s voice`}
      cells={
        <>
          <td>
            {/* the row is clickable too; this is its keyboard-reachable twin */}
            <button
              className="rowname"
              aria-pressed={selected}
              title={selected ? "show every line again" : `show only ${char.name}'s lines`}
              onClick={(e) => {
                e.stopPropagation();
                onSelect();
              }}
            >
              {char.name}
            </button>
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
        </>
      }
      picker={voiceCell}
    />
  );
}
