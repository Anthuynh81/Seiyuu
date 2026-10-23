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

/* -------------------------------------------------- suggest-a-cast preview rows */

/** What applying a suggested cast does to one character, in plain words:
    - "new": they get a voice in THIS book's cast for the first time — either a fresh voice is
      made (would_create), or they have no saved voice here and reuse an auto voice that already
      exists (a series sibling / leftover `<id>_auto`, which the server lists in would_recast).
    - "will change": they end up on a different voice than the saved cast, or (re-cast ticked)
      their saved auto voice is rebuilt, which re-records their lines.
    - "keeps your pick": a hand-chosen voice already saved, untouched.
    - "your pick": a hand-chosen voice that isn't saved yet (chosen in the preview, first cast).
    - "unchanged": the voice they already have in this cast stays as it is. */
export type CastRowStatus = "new" | "will change" | "keeps your pick" | "your pick" | "unchanged";

export function castRowStatus(
  characterId: string,
  proposedVoiceId: string,
  ctx: {
    /** the SAVED assignment (null = never cast) */
    saved: VoiceAssignment | null;
    wouldCreate: readonly string[];
    wouldRecast: readonly string[];
    recast: boolean;
  },
): CastRowStatus {
  const before = ctx.saved?.assignments[characterId];
  if (ctx.wouldCreate.includes(proposedVoiceId)) return "new";
  // not in the saved cast: whatever voice they get, it is new to this book
  if (before === undefined) return isManualPick(characterId, proposedVoiceId) ? "your pick" : "new";
  if (before !== proposedVoiceId) return "will change";
  if (ctx.recast && ctx.wouldRecast.includes(proposedVoiceId)) return "will change";
  return isManualPick(characterId, proposedVoiceId) ? "keeps your pick" : "unchanged";
}

/** Would applying this proposal write exactly what is already saved? Only then is apply a
    no-op. Re-cast rebuilds voices, and a new voice must still be made, so neither is a no-op. */
export function proposalMatchesSaved(
  proposed: VoiceAssignment,
  saved: VoiceAssignment | null,
  opts: { wouldCreate: readonly string[]; recast: boolean },
): boolean {
  if (!saved || opts.recast || opts.wouldCreate.length > 0) return false;
  return !castingVoicesDiffer(saved, castingFromServer(proposed));
}

/** A character's gender hint in words ("f"/"female" → "female"); null when unknown/absent. */
export function genderLabel(g: string | null | undefined): string | null {
  if (!g) return null;
  const k = g.trim().toLowerCase();
  if (k === "f" || k === "female") return "female";
  if (k === "m" || k === "male") return "male";
  if (k === "unknown" || k === "u" || k === "") return null;
  return k;
}
