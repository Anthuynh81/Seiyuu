import { useState } from "react";

import { useRecordEdit } from "../../api/hooks";
import type { CharacterSummary } from "../../api/types";
import { TalkDialog } from "../../components/Dialog";
import { PlainErrorLine } from "../../components/PlainError";
import { TalkSelect } from "../../components/Select";
import { mergePreview } from "../../lib/reviewTools";
import { NONE } from "./helpers";

export function CharacterEditor({
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
  const winner = others.find((o) => o.id === mergeInto);
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
      {winner && <p className="mono mt-1.5 text-[11px] text-ink-2">{mergePreview(char, winner.name)}</p>}
      <PlainErrorLine as="div" className="errline mt-3" error={record.error} fallback="Couldn't save that" />
    </TalkDialog>
  );
}
