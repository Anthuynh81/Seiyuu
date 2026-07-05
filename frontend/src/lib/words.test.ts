import { describe, expect, it } from "vitest";

import { alignWordTimings, buildClipWords, foldToken, groupPlayableRows, wordWeight, type WhisperWord } from "./words";

const row = (block: string, segIdx: number, audioSeg: number, dur: number, speaker: string | null, text: string) => ({
  block_id: block,
  segment_index: segIdx,
  audio_segment: audioSeg,
  duration_seconds: dur,
  speaker,
  speaker_name: speaker ? speaker.toUpperCase() : null,
  text,
});

describe("groupPlayableRows — the single-voice/multivoice clip merge", () => {
  it("multivoice: 1 row per wav -> 1 clip per row", () => {
    const rows = [
      row("b1", 0, 0, 2.0, null, "He said,"),
      row("b1", 1, 1, 3.0, "mr", "'hello.'"),
      row("b2", 0, 0, 4.0, null, "Silence."),
    ];
    const groups = groupPlayableRows(rows);
    expect(groups.map((g) => g.key)).toEqual(["b1:0", "b1:1", "b2:0"]);
    expect(groups.map((g) => g.duration)).toEqual([2.0, 3.0, 4.0]);
  });

  it("single-voice: a block's rows share ONE wav -> ONE clip (the field bug: duplicated audio)", () => {
    const rows = [
      row("b1", 0, 0, 10.0, null, "He said,"),
      row("b1", 1, 0, 10.0, "mr", "'hello.'"),
      row("b1", 2, 0, 10.0, null, "and left."),
      row("b2", 0, 0, 5.0, null, "Silence."),
    ];
    const groups = groupPlayableRows(rows);
    expect(groups).toHaveLength(2);
    expect(groups[0].rows).toHaveLength(3);
    // total duration counts the shared wav ONCE — 15s, not 35s
    expect(groups.reduce((a, g) => a + g.duration, 0)).toBe(15.0);
  });

  it("narration rows label the clip 'narration'; spoken rows use the speaker name", () => {
    const groups = groupPlayableRows([row("b1", 0, 0, 1, null, "x"), row("b2", 0, 0, 1, "mrs", "y")]);
    expect(groups[0].speaker).toBe("narration");
    expect(groups[1].speaker).toBe("MRS");
  });
});

describe("wordWeight — punctuation makes TTS linger", () => {
  it("orders sentence end > clause break > comma > bare word (same length)", () => {
    const w = (s: string) => wordWeight(s);
    expect(w("stop.")).toBeGreaterThan(w("stop;"));
    expect(w("stop;")).toBeGreaterThan(w("stop,"));
    expect(w("stop,")).toBeGreaterThan(w("stops"));
  });

  it("sees through closing quotes — 'hello.”' still counts as a sentence end", () => {
    expect(wordWeight('end.”')).toBe(wordWeight("ends.") ); // same length + same sentence bonus
  });
});

describe("buildClipWords — interpolated offsets", () => {
  const el = () => document.createElement("p");

  it("offsets start at 0, strictly increase, and stay inside the clip", () => {
    const words = buildClipWords([{ text: "It is a truth universally acknowledged.", el: el() }], 8.0);
    expect(words[0].offset).toBe(0);
    for (let i = 1; i < words.length; i++) expect(words[i].offset).toBeGreaterThan(words[i - 1].offset);
    expect(words[words.length - 1].offset).toBeLessThan(8.0);
  });

  it("a multi-row clip runs its rows sequentially — row 2's words start after row 1's", () => {
    const e1 = el();
    const e2 = el();
    const words = buildClipWords(
      [
        { text: "First segment here.", el: e1 },
        { text: "Second one.", el: e2 },
      ],
      10.0,
    );
    const row1Count = 3;
    expect(words).toHaveLength(5);
    expect(words[row1Count].offset).toBeGreaterThan(words[row1Count - 1].offset);
    // the spans landed in their own containers
    expect(e1.querySelectorAll(".w")).toHaveLength(3);
    expect(e2.querySelectorAll(".w")).toHaveLength(2);
  });

  it("rebuilding a container replaces its spans instead of appending duplicates", () => {
    const e1 = el();
    buildClipWords([{ text: "one two three", el: e1 }], 3);
    buildClipWords([{ text: "one two three", el: e1 }], 3);
    expect(e1.querySelectorAll(".w")).toHaveLength(3);
  });

  it("whitespace-only text yields no words and no crash", () => {
    expect(buildClipWords([{ text: "   ", el: el() }], 3)).toEqual([]);
  });

  it("each word's end meets the next word's offset; the last ends at the clip duration", () => {
    const words = buildClipWords([{ text: "one two three", el: el() }], 6.0);
    expect(words).toHaveLength(3);
    for (let i = 0; i < words.length - 1; i++) expect(words[i].end).toBe(words[i + 1].offset);
    expect(words[words.length - 1].end).toBe(6.0);
    for (const w of words) expect(w.end).toBeGreaterThan(w.offset);
  });
});

describe("alignWordTimings — whisper words drive real timing over source tokens", () => {
  const ww = (word: string, start: number, end: number): WhisperWord => ({ word, start, end });

  it("1:1 — every display token takes its whisper (start,end) verbatim", () => {
    const spans = alignWordTimings(
      ["The", "cat", "sat"],
      [ww("the", 0, 0.5), ww("cat", 0.5, 1.0), ww("sat", 1.0, 1.8)],
      1.8,
    );
    expect(spans).toEqual([
      { offset: 0, end: 0.5 },
      { offset: 0.5, end: 1.0 },
      { offset: 1.0, end: 1.8 },
    ]);
  });

  it("whisper-shorter — an unmatched display token is interpolated between its anchors", () => {
    // "dear" has no spoken counterpart; it fills the gap between hello.end and world.start
    const spans = alignWordTimings(
      ["hello", "dear", "world"],
      [ww("hello", 0, 1), ww("world", 2, 3)],
      3,
    );
    expect(spans[0]).toEqual({ offset: 0, end: 1 });
    expect(spans[1]).toEqual({ offset: 1, end: 2 }); // interpolated across the [1,2] gap
    expect(spans[2]).toEqual({ offset: 2, end: 3 });
  });

  it("whisper-longer — an extra spoken token is ignored, anchors still land exactly", () => {
    const spans = alignWordTimings(
      ["hello", "world"],
      [ww("hello", 0, 1), ww("cruel", 1, 1.5), ww("world", 1.5, 2.5)],
      2.5,
    );
    expect(spans[0]).toEqual({ offset: 0, end: 1 });
    expect(spans[1]).toEqual({ offset: 1.5, end: 2.5 });
  });

  it("folds punctuation, digits, and curly quotes so display and spoken forms still anchor", () => {
    expect(foldToken("“Hello,”")).toBe("hello");
    expect(foldToken("1,000")).toBe("1000");
    expect(foldToken("don’t")).toBe("dont");
    const spans = alignWordTimings(
      ["“Hello,”", "world.", "1,000"],
      [ww("hello", 0, 1), ww("world", 1, 2), ww("1000", 2, 3)],
      3,
    );
    expect(spans).toEqual([
      { offset: 0, end: 1 },
      { offset: 1, end: 2 },
      { offset: 2, end: 3 },
    ]);
  });

  it("no whisper words -> even spread fallback of the right length", () => {
    const spans = alignWordTimings(["a", "b", "c", "d"], [], 4);
    expect(spans).toHaveLength(4);
    expect(spans[0].offset).toBe(0);
    expect(spans[3].end).toBe(4);
  });

  it("times are monotonic non-decreasing and clamped inside the clip", () => {
    // leading + trailing unmatched tokens, one interior anchor
    const spans = alignWordTimings(
      ["intro", "here", "cat", "then", "outro"],
      [ww("cat", 2, 2.5)],
      5,
    );
    expect(spans).toHaveLength(5);
    for (let i = 1; i < spans.length; i++) expect(spans[i].offset).toBeGreaterThanOrEqual(spans[i - 1].offset);
    for (const s of spans) {
      expect(s.end).toBeGreaterThanOrEqual(s.offset);
      expect(s.offset).toBeGreaterThanOrEqual(0);
      expect(s.end).toBeLessThanOrEqual(5);
    }
    // the anchored token keeps its exact whisper time
    expect(spans[2]).toEqual({ offset: 2, end: 2.5 });
  });
});
