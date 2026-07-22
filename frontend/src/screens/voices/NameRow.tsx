import { useState } from "react";

import { useRenameVoice } from "../../api/hooks";
import type { VoiceOut } from "../../api/types";

/* -------------------------------------------------- rename (name is a safe, cache-free label) */

export function NameRow({ voice }: { voice: VoiceOut }) {
  const rename = useRenameVoice();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(voice.name);
  const save = () => {
    const name = draft.trim();
    if (!name || name === voice.name) return setEditing(false);
    rename.mutate({ voiceId: voice.voice_id, name }, { onSuccess: () => setEditing(false) });
  };
  if (!editing) {
    return (
      <h3 className="namerow">
        {voice.name}
        <button
          className="rowedit visible ml-1.5"
          title="rename voice"
          aria-label="rename voice"
          onClick={() => {
            setDraft(voice.name);
            rename.reset();
            setEditing(true);
          }}
        >
          ✎
        </button>
      </h3>
    );
  }
  return (
    <h3 className="namerow">
      <input
        className="taginput flex-1"
        value={draft}
        autoFocus
        aria-label="voice name"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") setEditing(false);
        }}
      />
      <button className="key quiet px-2 py-[2px]" disabled={rename.isPending} onClick={save}>
        save
      </button>
      {rename.error && <span className="mono text-[10px] text-clip">{rename.error.message}</span>}
    </h3>
  );
}
