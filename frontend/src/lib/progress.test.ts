import { describe, expect, it } from "vitest";

import { jobChapters, parseChapterProgress, progressLabel } from "./progress";

describe("parseChapterProgress", () => {
  it("reads the render/attribute loop's 'chapter N/M: title' line", () => {
    const p = parseChapterProgress("chapter 12/40: The Storm");
    expect(p).toEqual({
      chapter: 12,
      total: 40,
      title: "The Storm",
      done: 11,
      of: 40,
      fraction: 11 / 40,
      partial: false,
    });
    expect(progressLabel(p!)).toBe("chapter 12 of 40 · 28%");
  });

  it("accepts a line with no title, extra spaces and any case", () => {
    expect(parseChapterProgress("Chapter 1 / 3")).toMatchObject({ chapter: 1, total: 3, title: null, fraction: 0 });
  });

  it("anything that is not a chapter line falls back (null): the UI shows the stripe", () => {
    for (const text of [
      "",
      null,
      undefined,
      "verifying cost approval…",
      "kokoro resident",
      "  ! validation failed (wrong number) block=ch002_b0007 after 1 attempt(s) — flagged for review",
      "chapter 3: ch003.mp3 (4.2 min)", // assemble's line has no total
      "ch 2/5 · seg 40/900",
    ]) {
      expect(parseChapterProgress(text)).toBeNull();
    }
  });

  it("refuses impossible numbers instead of drawing a bar past 100%", () => {
    expect(parseChapterProgress("chapter 0/10")).toBeNull();
    expect(parseChapterProgress("chapter 11/10")).toBeNull();
    expect(parseChapterProgress("chapter 1/0")).toBeNull();
  });

  it("a chapter range counts within the range, not out of the whole book", () => {
    const p = parseChapterProgress("chapter 31/40: Night", [30, 31, 32]);
    expect(p).toMatchObject({ chapter: 31, done: 1, of: 3, partial: true });
    expect(p!.fraction).toBeCloseTo(1 / 3);
    expect(progressLabel(p!)).toBe("chapter 31 (2 of 3) · 33%");
  });

  it("a chapter outside the given range (stale params) falls back to the whole-book count", () => {
    expect(parseChapterProgress("chapter 5/40", [30, 31])).toMatchObject({ done: 4, of: 40, partial: false });
  });
});

describe("jobChapters", () => {
  it("reads the job's chapter list and ignores anything malformed", () => {
    expect(jobChapters({ chapters: [3, 1, 2] })).toEqual([3, 1, 2]);
    expect(jobChapters({ chapters: [] })).toEqual([]);
    expect(jobChapters({ chapters: "1-3" })).toEqual([]);
    expect(jobChapters({ chapters: [1, "2", 0, 2.5] })).toEqual([1]);
    expect(jobChapters(null)).toEqual([]);
    expect(jobChapters(undefined)).toEqual([]);
  });
});
