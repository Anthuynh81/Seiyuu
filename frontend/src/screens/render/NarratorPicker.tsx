import type { EngineVoicesOut, VoiceOut } from "../../api/types";
import { TalkSelect, type SelectOption } from "../../components/Select";
import { narratorTakesSpeed, SPEEDS, type NarratorChoice } from "../../lib/render";

/* -------------------------------------------------- one narrator: voice + speed */

/** Which voice reads the whole book in One narrator mode, and how fast. Built-in Kokoro
    presets and the standard narrator take a speed; a saved library voice records with its
    own saved settings, so the speed control steps aside and says why. */
export function NarratorPicker({
  choice,
  setChoice,
  speed,
  setSpeed,
  voices,
  presets,
  defaultName,
}: {
  choice: NarratorChoice;
  setChoice: (c: NarratorChoice) => void;
  speed: number;
  setSpeed: (s: number) => void;
  voices: VoiceOut[] | undefined;
  presets: EngineVoicesOut["voices"] | undefined;
  /** the server's standard narrator preset (from settings), when known */
  defaultName?: string;
}) {
  const options: SelectOption[] = [
    { value: "default", label: defaultName ? `Standard narrator (${defaultName})` : "Standard narrator" },
    ...(presets ?? []).map((p) => ({
      value: `preset:${p.id}`,
      label: `${p.name}${p.description ? ` — ${p.description}` : ""} · built-in`,
    })),
    ...(voices ?? [])
      // the one-narrator path can address ElevenLabs stock voices only (the server refuses
      // a cloned or mixed ElevenLabs voice here — those work in Full cast)
      .filter((v) => !(v.engine === "elevenlabs" && v.kind !== "preset"))
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((v) => ({
        value: `voice:${v.voice_id}`,
        label: `${v.name} · your voice${v.engine === "elevenlabs" ? " · paid" : ""}`,
      })),
  ];
  // a choice the lists don't (yet) contain still shows as itself rather than blank
  if (!options.some((o) => o.value === choice)) {
    options.push({ value: choice, label: choice.slice(choice.indexOf(":") + 1) });
  }
  const takesSpeed = narratorTakesSpeed(choice);
  return (
    <div className="scoperow" role="group" aria-label="narrator">
      <span className="tag">narrator</span>
      <TalkSelect ariaLabel="narrator voice" value={choice} onChange={(v) => setChoice(v as NarratorChoice)} options={options} />
      <span className="tag ml-2">speed</span>
      {takesSpeed ? (
        <TalkSelect
          ariaLabel="narrator speed"
          value={String(speed)}
          onChange={(v) => setSpeed(Number(v))}
          options={SPEEDS.map((s) => ({ value: String(s.value), label: s.label }))}
        />
      ) : (
        <span className="mono scopehint">this voice reads at its own saved speed</span>
      )}
    </div>
  );
}
