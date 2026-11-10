import type { CastPlan, PlanRow, VoiceAssignment } from "../../api/types";
import { NARRATOR_ROW } from "../../api/types";

/** Every row in table order: the narrator first, then characters busiest first. */
export function planRows(plan: CastPlan): PlanRow[] {
  return [plan.narrator, ...plan.rows];
}

/** Rows the design-cast job will design now: their own voice, not designed yet. */
export function rowsToDesign(plan: CastPlan): PlanRow[] {
  return planRows(plan).filter((r) => !r.keep_current && r.same_as === null && r.voice_id === null);
}

/** Measured on the 3080 Ti: ~5 s per take once the model is loaded, ~1 min to load it. */
export function designMinutes(rows: PlanRow[]): number {
  const takes = rows.reduce((n, r) => n + r.takes, 0);
  return takes === 0 ? 0 : Math.max(1, Math.round(1 + (takes * 5) / 60));
}

/** The row's name as the table shows it ("Narrator" for the narrator row). */
export function rowName(plan: CastPlan, id: string | null): string {
  if (id === null) return "";
  if (id === NARRATOR_ROW) return "Narrator";
  return plan.rows.find((r) => r.character_id === id)?.name ?? id;
}

/** One plain line for where a row's voice stands. */
export function rowStatus(plan: CastPlan, row: PlanRow): string {
  if (row.keep_current) return "keeps its current voice";
  if (row.same_as !== null) return `shares ${rowName(plan, row.same_as)}'s voice`;
  if (row.voice_id !== null) return "designed";
  if (row.design_id !== null) return "designing was interrupted — design again to finish";
  return "not designed yet";
}

/** Rows another row may share a voice with: characters that design their own (never itself). */
export function shareTargets(plan: CastPlan, row: PlanRow): PlanRow[] {
  return plan.rows.filter((r) => r.character_id !== row.character_id && r.same_as === null);
}

/** "Young adult · male · 456 lines" chips text. */
export function rowFacts(row: PlanRow): string {
  const lines = row.character_id === NARRATOR_ROW ? null : `${row.line_count} line${row.line_count === 1 ? "" : "s"}`;
  return [row.age, row.gender, lines].filter(Boolean).join(" · ");
}

/** The render's first-person test: the narrator's voice is also a character's voice. */
export function narratorSharesVoice(assignment: VoiceAssignment): boolean {
  return Object.values(assignment.assignments).includes(assignment.narrator_voice_id);
}

/** One plain line for which text this book's narration is spoken with. */
export function narrationNote(assignment: VoiceAssignment | null): string {
  if (assignment === null)
    return "Narration uses the inner-voice text when the narrator shares a character's voice.";
  return narratorSharesVoice(assignment)
    ? "The narrator shares a character's voice, so this book's narration uses the inner-voice text."
    : "The narrator has a voice of its own, so this book's narration uses the narration text.";
}

/** An instruction as the server stores it: single spaces, no edges. */
export function oneLine(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}
