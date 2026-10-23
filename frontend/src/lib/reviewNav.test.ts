import { describe, expect, it, vi } from "vitest";

import type { SegmentRow } from "../api/types";
import { findLine, matchesSpeaker, nextInChapter, readLocation, walkToNextCheck } from "./reviewNav";

function row(block: string, seg: number, overrides: Partial<SegmentRow> = {}): SegmentRow {
  return {
    block_id: block,
    segment_index: seg,
    type: "dialogue",
    speaker: "alice",
    speaker_name: "Alice",
    text: "x",
    confidence: 0.99,
    unattributed_quote: false,
    has_audio: false,
    audio_segment: null,
    duration_seconds: null,
    voice_id: null,
    audio_key: null,
    ...overrides,
  };
}
const low = (r: SegmentRow) => r.confidence < 0.7;

describe("readLocation", () => {
  it("reads chapter, line and speaker from the URL", () => {
    expect(readLocation(new URLSearchParams("ch=3&block=ch003_b0002&seg=1&speaker=bob"))).toEqual({
      chapter: 3,
      explicitChapter: true,
      target: { block: "ch003_b0002", seg: 1 },
      speaker: "bob",
    });
  });

  it("falls back to the linked paragraph's chapter, then chapter 1; junk is ignored", () => {
    expect(readLocation(new URLSearchParams("block=ch004_b0001")).chapter).toBe(4);
    expect(readLocation(new URLSearchParams("block=ch004_b0001")).explicitChapter).toBe(false);
    expect(readLocation(new URLSearchParams("ch=0&seg=-1")).chapter).toBe(1);
    expect(readLocation(new URLSearchParams("ch=abc")).chapter).toBe(1);
    expect(readLocation(new URLSearchParams("block=x&seg=two")).target).toEqual({ block: "x", seg: null });
    expect(readLocation(new URLSearchParams("")).target).toBeNull();
  });
});

describe("line helpers", () => {
  const rows = [row("b1", 0), row("b1", 1, { confidence: 0.5 }), row("b2", 0, { speaker: null, confidence: 0.2 })];

  it("findLine picks the exact line, or the paragraph's first line without an index", () => {
    expect(findLine(rows, { block: "b1", seg: 1 })).toBe(rows[1]);
    expect(findLine(rows, { block: "b1", seg: null })).toBe(rows[0]);
    expect(findLine(rows, { block: "b9", seg: null })).toBeUndefined();
  });

  it("nextInChapter starts after the given line, or from the top", () => {
    expect(nextInChapter(rows, null, low)).toBe(rows[1]);
    expect(nextInChapter(rows, { block: "b1", seg: 1 }, low)).toBe(rows[2]);
    expect(nextInChapter(rows, { block: "b2", seg: 0 }, low)).toBeNull();
    expect(nextInChapter(rows, { block: "gone", seg: 0 }, low)).toBe(rows[1]);
  });

  it("matchesSpeaker understands 'narration' and no filter", () => {
    expect(matchesSpeaker(rows[0], null)).toBe(true);
    expect(matchesSpeaker(rows[0], "alice")).toBe(true);
    expect(matchesSpeaker(rows[0], "bob")).toBe(false);
    expect(matchesSpeaker(rows[2], "narration")).toBe(true);
    expect(matchesSpeaker(rows[0], "narration")).toBe(false);
  });
});

describe("walkToNextCheck", () => {
  it("stays in the open chapter when it can, fetching nothing", async () => {
    const laterRows = vi.fn();
    const result = await walkToNextCheck({
      chapters: [1, 2],
      from: { chapter: 1, after: null },
      startRows: [row("ch1a", 0), row("ch1b", 0, { confidence: 0.3 })],
      keep: low,
      laterRows,
    });
    expect(result).toMatchObject({ kind: "found", chapter: 1, row: { block_id: "ch1b" } });
    expect(laterRows).not.toHaveBeenCalled();
  });

  it("crosses chapters lazily, skipping ones with nothing to read, in order", async () => {
    const seen: number[] = [];
    const result = await walkToNextCheck({
      chapters: [1, 2, 3, 4],
      from: { chapter: 1, after: { block: "ch1a", seg: 0 } },
      startRows: [row("ch1a", 0, { confidence: 0.3 })],
      keep: low,
      laterRows: async (ch) => {
        seen.push(ch);
        if (ch === 2) return null; // front matter
        if (ch === 3) return [row("ch3a", 0, { confidence: 0.9 }), row("ch3b", 2, { confidence: 0.1 })];
        throw new Error("should have stopped at chapter 3");
      },
    });
    expect(seen).toEqual([2, 3]);
    expect(result).toMatchObject({ kind: "found", chapter: 3, row: { block_id: "ch3b", segment_index: 2 } });
  });

  it("reports the end of the book, an error, or a cancellation", async () => {
    const base = { chapters: [1, 2], from: { chapter: 1, after: null }, startRows: [], keep: low };
    expect(await walkToNextCheck({ ...base, laterRows: async () => [] })).toEqual({ kind: "end" });
    expect(
      await walkToNextCheck({
        ...base,
        laterRows: async () => {
          throw new Error("offline");
        },
      }),
    ).toEqual({ kind: "error", chapter: 2, message: "offline" });
    expect(
      await walkToNextCheck({ ...base, laterRows: async () => [row("x", 0, { confidence: 0 })], cancelled: () => true }),
    ).toEqual({ kind: "cancelled" });
  });

  it("without the open chapter's rows, that chapter is fetched too", async () => {
    const laterRows = vi.fn(async (ch: number) => (ch === 2 ? [row("ch2a", 0, { confidence: 0.2 })] : []));
    const result = await walkToNextCheck({
      chapters: [1, 2],
      from: { chapter: 2, after: null },
      startRows: null,
      keep: low,
      laterRows,
    });
    expect(laterRows).toHaveBeenCalledWith(2);
    expect(result).toMatchObject({ kind: "found", chapter: 2 });
  });
});
