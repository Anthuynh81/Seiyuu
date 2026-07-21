import { useState } from "react";

import { useSetVoiceTags } from "../../api/hooks";
import type { VoiceOut } from "../../api/types";

/* -------------------------------------------------- tags */

export function TagEditor({ voice, titleFor }: { voice: VoiceOut; titleFor: (tag: string) => string }) {
  const setTags = useSetVoiceTags();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const save = () =>
    setTags.mutate(
      { voiceId: voice.voice_id, tags: draft.split(",").map((s) => s.trim()).filter(Boolean) },
      { onSuccess: () => setEditing(false) },
    );
  if (!editing) {
    return (
      <div className="tagsrow">
        {voice.tags.map((t) => (
          <span key={t} className="tagchip" title={t}>{titleFor(t)}</span>
        ))}
        <button
          className="rowedit visible ml-0"
          title="edit tags"
          onClick={() => {
            setDraft(voice.tags.join(", "));
            setEditing(true);
          }}
        >
          {voice.tags.length ? "✎" : "+ tag"}
        </button>
      </div>
    );
  }
  return (
    <div className="tagsrow">
      <input
        className="taginput"
        value={draft}
        autoFocus
        placeholder="comma, separated, tags"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") setEditing(false);
        }}
      />
      <button className="key quiet px-2 py-[2px]" disabled={setTags.isPending} onClick={save}>
        save
      </button>
      {setTags.error && <span className="mono text-[10px] text-clip">{setTags.error.message}</span>}
    </div>
  );
}
