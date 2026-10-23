import type { VoiceOut } from "../../api/types";
import { TalkSelect } from "../../components/Select";
import { NONE } from "./helpers";

/** Option key for "+ new voice for <Name>": never a voice id, never selected — choosing it
    opens the Add voice dialog instead of changing the pick. */
const NEW_VOICE = "__new_voice__";

export function VoicePicker({
  value,
  onChange,
  voices,
  pool,
  allowOwn,
  newVoiceFor,
  onNewVoice,
}: {
  value: string | null;
  onChange: (v: string | null) => void;
  voices: VoiceOut[];
  /** the unfiltered list — keeps the assigned voice visible when a tag filter hides it */
  pool?: VoiceOut[];
  allowOwn?: boolean; // the thought-voice "speaker's own" option
  /** the character's name: adds "+ new voice for <name>" (with onNewVoice) */
  newVoiceFor?: string;
  onNewVoice?: () => void;
}) {
  const current =
    value !== null && !voices.some((v) => v.voice_id === value)
      ? (pool ?? []).filter((v) => v.voice_id === value)
      : [];
  const options = [
    // first, so it's found without scrolling a long library
    ...(newVoiceFor && onNewVoice ? [{ value: NEW_VOICE, label: `+ new voice for ${newVoiceFor}` }] : []),
    ...(allowOwn ? [{ value: NONE, label: "speaker's own" }] : value === null ? [{ value: NONE, label: "— uncast —" }] : []),
    ...[...current, ...voices].map((v) => ({ value: v.voice_id, label: `${v.name} · ${v.engine}` })),
  ];
  return (
    <TalkSelect
      className="vpick"
      ariaLabel="voice"
      value={value ?? NONE}
      onChange={(v) => {
        if (v === NEW_VOICE) onNewVoice?.();
        else onChange(v === NONE ? null : v);
      }}
      options={options}
    />
  );
}
