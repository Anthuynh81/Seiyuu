import { describe, expect, it } from "vitest";

import type { LexiconEntry } from "../api/types";
import { blankEntry, cleanForSave, duplicateTerms, entriesSig } from "./lexicon";

const entry = (over: Partial<LexiconEntry>): LexiconEntry => ({ ...blankEntry(), ...over });

describe("cleanForSave", () => {
  it("trims, maps blank optional fields to null, and drops incomplete rows", () => {
    const out = cleanForSave([
      entry({ term: "  Hermione ", respelling: " Her My Oh Nee ", ipa: "  ", note: "" }),
      entry({ term: "onlyterm", respelling: "" }), // dropped: no respelling
      entry({ term: "", respelling: "x" }), // dropped: no term
    ]);
    expect(out).toEqual([
      { term: "Hermione", respelling: "Her My Oh Nee", ipa: null, note: null, case_sensitive: false },
    ]);
  });
});

describe("duplicateTerms", () => {
  it("flags case-insensitive collisions but respects case_sensitive entries", () => {
    expect(
      duplicateTerms([
        entry({ term: "Chapter", respelling: "A" }),
        entry({ term: "chapter", respelling: "B" }),
      ]),
    ).toEqual(["chapter"]);
    expect(
      duplicateTerms([
        entry({ term: "Reed", respelling: "A", case_sensitive: true }),
        entry({ term: "reed", respelling: "B", case_sensitive: true }),
      ]),
    ).toEqual([]);
  });
});

describe("entriesSig", () => {
  it("is stable across cosmetic whitespace and ignores dropped rows", () => {
    const a = [entry({ term: "X", respelling: "Y" })];
    const b = [entry({ term: " X ", respelling: "Y" }), entry({ term: "", respelling: "" })];
    expect(entriesSig(a)).toBe(entriesSig(b));
  });
});
