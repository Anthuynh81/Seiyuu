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
