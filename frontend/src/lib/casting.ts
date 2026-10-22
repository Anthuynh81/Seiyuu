import type { VoiceAssignment } from "../api/types";

/** The editable local copy of a casting. */
export interface CastingState {
  narrator: string;
  thought: string | null;
  stage: "draft" | "final";
  map: Record<string, string>;
}

export function castingFromServer(a: VoiceAssignment): CastingState {
  return { narrator: a.narrator_voice_id, thought: a.thought_voice_id, stage: a.stage, map: { ...a.assignments } };
}

/** Dirty tracking: does the local copy differ from what the server holds? Map
    comparison is key-order independent. */
export function castingDiffers(server: VoiceAssignment, local: CastingState): boolean {
  if (local.stage !== server.stage) return true;
  return castingVoicesDiffer(server, local);
}

/** Do the VOICE picks differ — narrator/thought/assignments, ignoring `stage`? This is what
    makes rendered audio stale; a draft->final promote alone re-renders nothing (mirrors the
    backend's _ASSIGNMENT_IDENTITY / hash_assignment, which also exclude stage). */
export function castingVoicesDiffer(server: VoiceAssignment, local: CastingState): boolean {
  if (local.narrator !== server.narrator_voice_id || local.thought !== server.thought_voice_id) {
    return true;
  }
  const a = local.map;
  const b = server.assignments;
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) return true;
  return keys.some((k) => a[k] !== b[k]);
}

/* -------------------------------------------------- manual picks (keep_existing) */

/** One character whose voice the user chose by hand (anything but its `<id>_auto` voice). */
export interface ManualPick {
  characterId: string;
  voiceId: string;
}

/** Is this a hand pick? The auto-caster only ever writes `<char_id>_auto` for a character
    (services/assignment.py), so any other voice id came from the user or a series override.
    Mirrors the backend's keep_existing rule: re-draft/suggest keep exactly these. */
export function isManualPick(characterId: string, voiceId: string): boolean {
  return voiceId !== `${characterId}_auto`;
}

/** The manual picks in an assignment map, in its key order. */
export function manualPicks(assignments: Record<string, string>): ManualPick[] {
  return Object.entries(assignments)
    .filter(([characterId, voiceId]) => isManualPick(characterId, voiceId))
    .map(([characterId, voiceId]) => ({ characterId, voiceId }));
}

/** Port of the backend's voices/library.py `slugify` (casefold, runs of non-[a-z0-9] -> "_",
    trim "_", empty -> "voice"). Only used to predict the auto narrator's id below. */
export function slugify(name: string): string {
  return (
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "") || "voice"
  );
}

/** The exact narrator id a full reset writes: `narrator_{slugify(kokoro_default_voice)}`
    (services/assignment.py draft_assignment). A `narrator_` PREFIX is not enough — user voice
    ids are `slugify(name)_<hex>`, so a cloned voice named "Narrator" is `narrator_3fa2`, and an
    auto narrator from an older default preset is `narrator_<old preset>`; a reset replaces both. */
export function autoNarratorId(defaultPreset: string): string {
  return `narrator_${slugify(defaultPreset)}`;
}

/** What a full reset (draft with keep_existing:false) throws away from a saved assignment:
    every manual pick, a narrator other than the one the reset writes, and the thought voice
    (auto-cast never sets one). Empty picks + both null means nothing manual is lost. */
export interface ResetLosses {
  picks: ManualPick[];
  narrator: string | null;
  thought: string | null;
}

/** `resetNarratorId` is the narrator the reset will write (see autoNarratorId); pass null when
    it is unknown (settings unavailable) and the saved narrator is reported as lost — the
    confirm gate over-reports rather than hides a loss. */
export function resetLosses(a: VoiceAssignment, resetNarratorId: string | null): ResetLosses {
  return {
    picks: manualPicks(a.assignments),
    narrator: a.narrator_voice_id === resetNarratorId ? null : a.narrator_voice_id,
    thought: a.thought_voice_id,
  };
}
