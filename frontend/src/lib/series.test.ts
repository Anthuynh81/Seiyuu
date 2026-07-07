import { describe, expect, it } from "vitest";

import type { LinkSuggestion } from "../api/types";
import { applicableCount, confirmedOverrides, defaultConfirmed } from "./series";

function sug(character_id: string, voice_id: string, voice_exists = true): LinkSuggestion {
  return { character_id, canonical_name: character_id, identity_key: character_id.toLowerCase(), voice_id, voice_exists };
}

describe("defaultConfirmed", () => {
  it("confirms only suggestions whose linked voice still exists", () => {
    const s = [sug("harry", "v1"), sug("ron", "v2", false), sug("hermione", "v3")];
    expect(defaultConfirmed(s)).toEqual(new Set(["harry", "hermione"]));
  });

  it("empty suggestions -> empty set", () => {
    expect(defaultConfirmed([])).toEqual(new Set());
  });
});

describe("confirmedOverrides", () => {
  it("includes only confirmed AND still-existing links", () => {
    const s = [sug("harry", "v1"), sug("ron", "v2"), sug("hermione", "v3")];
    const confirmed = new Set(["harry", "hermione"]);
    expect(confirmedOverrides(s, confirmed)).toEqual({ harry: "v1", hermione: "v3" });
  });

  it("drops a confirmed link whose voice was deleted — never applies a missing voice", () => {
    const s = [sug("harry", "v1"), sug("ron", "gone", false)];
    const confirmed = new Set(["harry", "ron"]);
    expect(confirmedOverrides(s, confirmed)).toEqual({ harry: "v1" });
  });

  it("nothing confirmed -> no overrides (never silent auto-apply)", () => {
    const s = [sug("harry", "v1"), sug("ron", "v2")];
    expect(confirmedOverrides(s, new Set())).toEqual({});
  });
});

describe("applicableCount", () => {
  it("counts only inheritable (confirmed + existing) links", () => {
    const s = [sug("harry", "v1"), sug("ron", "gone", false), sug("hermione", "v3")];
    expect(applicableCount(s, new Set(["harry", "ron", "hermione"]))).toBe(2);
  });
});
