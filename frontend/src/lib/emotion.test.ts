import { describe, expect, it } from "vitest";

import type { AttributedChapterView, EmotionVerdict } from "../api/types";
import { buildEmotionMap, intensityDots } from "./emotion";

const happy: EmotionVerdict = { label: "happy", intensity: 2 };
const angry: EmotionVerdict = { label: "angry", intensity: 3 };
const neutral: EmotionVerdict = { label: "neutral", intensity: 2 };

/** A block with two quoted lines (F1) trading emotions, wrapped in narration. */
function chapter(segments: { block_id: string }[], emotions: (EmotionVerdict | null)[]): AttributedChapterView {
  return { index: 1, segments, segment_emotions: emotions };
}

describe("buildEmotionMap", () => {
  it("keys each verdict by its per-block ordinal (matches segment_index)", () => {
    const ch = chapter(
      // ch1_b1 has narration + two dialogue quotes; ch1_b2 one quote
      [{ block_id: "ch1_b1" }, { block_id: "ch1_b1" }, { block_id: "ch1_b1" }, { block_id: "ch1_b2" }],
      [null, happy, angry, angry],
    );
    const map = buildEmotionMap(ch);
    expect(map.get("ch1_b1:1")).toEqual(happy); // second segment of the block
    expect(map.get("ch1_b1:2")).toEqual(angry); // third
    expect(map.get("ch1_b2:0")).toEqual(angry); // ordinal resets per block
    expect(map.has("ch1_b1:0")).toBe(false); // narration carries no verdict
  });

  it("skips neutral and null (they degrade to no render override)", () => {
    const ch = chapter([{ block_id: "b" }, { block_id: "b" }], [neutral, null]);
    expect(buildEmotionMap(ch).size).toBe(0);
  });

  it("treats a length-mismatched (pre-emotion v3/v4) report as no tags", () => {
    const ch = chapter([{ block_id: "b" }, { block_id: "b" }], []); // segment_emotions empty
    expect(buildEmotionMap(ch).size).toBe(0);
  });

  it("undefined chapter yields an empty map", () => {
    expect(buildEmotionMap(undefined).size).toBe(0);
  });
});

describe("intensityDots", () => {
  it("renders one dot per level, clamped to 1–3", () => {
    expect(intensityDots(1)).toBe("•");
    expect(intensityDots(2)).toBe("••");
    expect(intensityDots(3)).toBe("•••");
    expect(intensityDots(0)).toBe("•"); // clamp low
    expect(intensityDots(9)).toBe("•••"); // clamp high
  });
});
