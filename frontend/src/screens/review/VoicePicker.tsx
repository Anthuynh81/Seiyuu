import type { VoiceOut } from "../../api/types";
import { TalkSelect } from "../../components/Select";
import { NONE } from "./helpers";

export function VoicePicker({
  value,
  onChange,
  voices,
  pool,
  allowOwn,
}: {
  value: string | null;
  onChange: (v: string | null) => void;
  voices: VoiceOut[];
  /** the unfiltered list — keeps the assigned voice visible when a tag filter hides it */
  pool?: VoiceOut[];
  allowOwn?: boolean; // the thought-voice "speaker's own" option
}) {
  const current =
    value !== null && !voices.some((v) => v.voice_id === value)
      ? (pool ?? []).filter((v) => v.voice_id === value)
      : [];
  const options = [
    ...(allowOwn ? [{ value: NONE, label: "speaker's own" }] : value === null ? [{ value: NONE, label: "— uncast —" }] : []),
    ...[...current, ...voices].map((v) => ({ value: v.voice_id, label: `${v.name} · ${v.engine}` })),
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
