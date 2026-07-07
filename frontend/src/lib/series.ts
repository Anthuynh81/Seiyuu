import type { LinkSuggestion } from "../api/types";

/* Pure helpers for the Series screen (F5). Linking is suggestion-then-confirm: the backend only
   ever proposes within-series name matches; the user confirms which to inherit, and NOTHING is
   applied silently. These build the {char_id -> voice_id} overrides from the confirmed set. */

/** Suggestions default to confirmed only when their linked voice STILL EXISTS — the common case
    of a returning character whose voice is intact. A deleted voice starts unconfirmed and can't
    be inherited (precision over recall). */
export function defaultConfirmed(suggestions: LinkSuggestion[]): Set<string> {
  return new Set(suggestions.filter((s) => s.voice_exists).map((s) => s.character_id));
}

/** The overrides dict to feed the draft: only suggestions the user CONFIRMED and whose voice
    still exists. A confirmed-but-deleted link is dropped rather than applied to a missing voice. */
export function confirmedOverrides(
  suggestions: LinkSuggestion[],
  confirmed: ReadonlySet<string>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const s of suggestions) {
    if (confirmed.has(s.character_id) && s.voice_exists) out[s.character_id] = s.voice_id;
  }
  return out;
}

/** How many confirmed links are actually applicable (voice present) — what the Apply button
    inherits. Separate from the raw confirmed count so a confirmed-but-deleted link isn't
    miscounted as inheritable. */
export function applicableCount(
  suggestions: LinkSuggestion[],
  confirmed: ReadonlySet<string>,
): number {
  return Object.keys(confirmedOverrides(suggestions, confirmed)).length;
}
