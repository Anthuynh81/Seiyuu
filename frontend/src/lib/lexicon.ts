import type { LexiconEntry, RespellSuggestion } from "../api/types";

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

/** F3 (v1.1): fold ADVISORY AI respellings into the editor rows WITHOUT clobbering user input.
    For a term that already has a row, fill an EMPTY respelling (and empty note) but never
    overwrite text the user typed; for a suggested term with no row yet, append a new row. Purely
    functional — the user still reviews and saves. Suggestions are matched case-insensitively. */
export function applyRespellings(
  rows: LexiconEntry[],
  suggestions: RespellSuggestion[],
): LexiconEntry[] {
  const byTerm = new Map(suggestions.map((s) => [s.term.trim().toLowerCase(), s]));
  const present = new Set(rows.map((r) => r.term.trim().toLowerCase()).filter(Boolean));
  const next = rows.map((row) => {
    const s = byTerm.get(row.term.trim().toLowerCase());
    if (!s) return row;
    return {
      ...row,
      respelling: row.respelling.trim() ? row.respelling : s.respelling,
      note: row.note?.trim() ? row.note : (s.note ?? null),
    };
  });
  for (const s of suggestions) {
    const key = s.term.trim().toLowerCase();
    if (key && !present.has(key)) {
      next.push({ ...blankEntry(s.term.trim()), respelling: s.respelling, note: s.note ?? null });
    }
  }
  return next;
}
