import type { CharacterSummary } from "../../api/types";
import { chapterOfBlock } from "../../api/types";
import { Tip } from "../../components/Tooltip";

/* -------------------------------------------------- roster */

export function RosterRow({
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
