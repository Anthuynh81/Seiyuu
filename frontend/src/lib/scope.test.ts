import { describe, expect, it } from "vitest";

import { continueRange, scopeChapters } from "./scope";

describe("scopeChapters", () => {
  it("whole book means [] — the API contract for 'no chapter filter'", () => {
    expect(scopeChapters({ kind: "whole" }, 61)).toEqual([]);
  });

  it("range is inclusive", () => {
    expect(scopeChapters({ kind: "range", from: 2, to: 5 }, 61)).toEqual([2, 3, 4, 5]);
  });

  it("clamps to the book on both ends", () => {
    expect(scopeChapters({ kind: "range", from: -3, to: 999 }, 4)).toEqual([1, 2, 3, 4]);
  });

  it("inverted range collapses to the from-chapter, never crashes or goes negative", () => {
    expect(scopeChapters({ kind: "range", from: 9, to: 3 }, 61)).toEqual([9]);
  });

  it("single-chapter range works", () => {
    expect(scopeChapters({ kind: "range", from: 7, to: 7 }, 61)).toEqual([7]);
  });
});

describe("continueRange", () => {
  it("picks up after the last rendered chapter", () => {
    expect(continueRange(new Set([1, 2, 3]), 61, 10)).toEqual({ kind: "range", from: 4, to: 13 });
  });

  it("skips holes: continues from the FIRST unrendered chapter", () => {
    expect(continueRange(new Set([1, 2, 5]), 61, 10)).toEqual({ kind: "range", from: 3, to: 12 });
  });

  it("clamps the tail to the book", () => {
    expect(continueRange(new Set([1, 2, 3]), 5, 10)).toEqual({ kind: "range", from: 4, to: 5 });
  });

  it("nothing rendered yet -> no preset (nothing to continue from)", () => {
    expect(continueRange(new Set(), 61, 10)).toBeNull();
  });

  it("fully rendered -> no preset", () => {
    expect(continueRange(new Set([1, 2, 3]), 3, 10)).toBeNull();
  });
});
