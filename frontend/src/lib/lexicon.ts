import type { LexiconEntry } from "../api/types";

/** A fresh, empty editor row. */
export function blankEntry(term = ""): LexiconEntry {
  return { term, respelling: "", ipa: null, note: null, case_sensitive: false };
}

/** Trim, blank -> null for optional fields, and drop rows with no term or no respelling —
    exactly the shape the PUT endpoint accepts (which also validates non-empty term/respelling). */
export function cleanForSave(entries: LexiconEntry[]): LexiconEntry[] {
  return entries
    .map((e) => ({
      term: e.term.trim(),
      respelling: e.respelling.trim(),
      ipa: e.ipa?.trim() ? e.ipa.trim() : null,
      note: e.note?.trim() ? e.note.trim() : null,
      case_sensitive: e.case_sensitive,
    }))
    .filter((e) => e.term !== "" && e.respelling !== "");
}

/** Terms that collide (case-insensitively, unless case_sensitive) — the server 422s on these,
    so the editor disables Save and flags them first. Compares the cleaned set. */
export function duplicateTerms(entries: LexiconEntry[]): string[] {
  const seen = new Set<string>();
  const dupes = new Set<string>();
  for (const e of cleanForSave(entries)) {
    const key = e.case_sensitive ? e.term : e.term.toLowerCase();
    if (seen.has(key)) dupes.add(e.term);
    seen.add(key);
  }
  return [...dupes];
}

/** Stable signature of the editable content, for dirty detection against the server state. */
export function entriesSig(entries: LexiconEntry[]): string {
  return JSON.stringify(cleanForSave(entries));
}
