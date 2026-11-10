import type { DesignTakeOut, VoiceOut } from "../../api/types";

export const BORROW_RETRY_MAX = 3; // bounded auto-retries while a render lends the GPU between segments

/** A recipe lifted off an existing preset/blend voice to pre-fill the Add dialog. Duplicating
    mints a NEW voice_id with its own (empty) render history — so tweaking the copy can never
    drift the original's cached audio. Cloned voices can't be duplicated this way (their source
    is reference.wav + a hash-bound consent, which the create path can't re-derive). */
export interface DuplicateRecipe {
  kind: "preset" | "blend";
  name: string;
  engine: string;
  presetId?: string;
  layers?: { preset_id: string; weight: number }[];
  seed: number;
}

export function recipeOf(voice: VoiceOut): DuplicateRecipe | null {
  if (voice.kind === "preset" && voice.preset_id) {
    return { kind: "preset", name: `${voice.name} copy`, engine: voice.engine, presetId: voice.preset_id, seed: voice.seed };
  }
  if (voice.kind === "blend" && voice.blend && voice.blend.length >= 2) {
    // weights are stored as normalized ratios; scale to readable 0-100 for the mixer faders
    const sum = voice.blend.reduce((a, b) => a + b.weight, 0) || 1;
    const layers = voice.blend.map((b) => ({ preset_id: b.preset_id, weight: Math.max(1, Math.round((100 * b.weight) / sum)) }));
    return { kind: "blend", name: `${voice.name} copy`, engine: voice.engine, layers, seed: voice.seed };
  }
  return null;
}

/** Plain words for one designed take's pace and speech check. */
export function takeVerdict(t: Pick<DesignTakeOut, "wpm" | "in_pace" | "ok" | "reason">): string {
  const pace = t.in_pace ? "natural pace" : t.wpm > 175 ? "a bit fast" : "a bit slow";
  const check = t.ok ? "every word clear" : `the speech check flagged it${t.reason ? ` (${t.reason})` : ""}`;
  return `${pace}, ${check}`;
}

/** Engines that clone locally from the library's reference clip (the server refuses others). */
const LOCAL_CLONING = new Set(["breeze", "qwen3tts", "chatterbox", "indextts2"]);

/** The engine a cloned voice can move to from its card: Breeze, or back to Qwen from Breeze.
    null for a voice that cannot move (a preset or blend, or a clone kept in the cloud). */
export function switchTarget(voice: VoiceOut): string | null {
  if (voice.kind !== "cloned" || !LOCAL_CLONING.has(voice.engine)) return null;
  return voice.engine === "breeze" ? "qwen3tts" : "breeze";
}
