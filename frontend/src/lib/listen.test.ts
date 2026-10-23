import { describe, expect, it } from "vitest";

import {
  castVoiceFor,
  chapterSeconds,
  clock,
  fileSize,
  lexiconHref,
  parseResume,
  pronunciationTerm,
  resumeTarget,
  reviewChapterHref,
  reviewLineHref,
  voiceChangedSince,
} from "./listen";

const cast = { narrator_voice_id: "narr", assignments: { mara: "v-mara" }, thought_voice_id: null };

describe("links back to the fixes", () => {
  it("land on the exact line, the chapter, or the dictionary with the word waiting", () => {
    expect(reviewLineHref("b 1", 3, { block_id: "ch3_b7", segment_index: 2 })).toBe(
      "/review?book=b+1&ch=3&block=ch3_b7&seg=2",
    );
    expect(reviewChapterHref("b1", 3)).toBe("/review?book=b1&ch=3");
    expect(lexiconHref("b1", "Kael'thas")).toBe("/lexicon?book=b1&term=Kael%27thas");
    expect(lexiconHref("b1", null)).toBe("/lexicon?book=b1");
  });

  it("pronunciationTerm trims outer punctuation but keeps names whole, and refuses phrases", () => {
    expect(pronunciationTerm("“Kael'thas,”")).toBe("Kael'thas");
    expect(pronunciationTerm("  Jean-Luc. ")).toBe("Jean-Luc");
    expect(pronunciationTerm("Ser  Duncan\nthe")).toBe("Ser Duncan the");
    expect(pronunciationTerm("—")).toBeNull();
    expect(pronunciationTerm("")).toBeNull();
    expect(pronunciationTerm(null)).toBeNull();
    expect(pronunciationTerm("one two three four five")).toBeNull();
  });
});

describe("voices in the margin", () => {
  it("castVoiceFor follows the renderer's rule", () => {
    expect(castVoiceFor({ type: "narration", speaker: null }, cast)).toBe("narr");
    expect(castVoiceFor({ type: "dialogue", speaker: null }, cast)).toBe("narr");
    expect(castVoiceFor({ type: "dialogue", speaker: "mara" }, cast)).toBe("v-mara");
    expect(castVoiceFor({ type: "dialogue", speaker: "uncast" }, cast)).toBe("narr");
    expect(castVoiceFor({ type: "thought", speaker: "mara" }, cast)).toBe("v-mara");
    expect(castVoiceFor({ type: "thought", speaker: "mara" }, { ...cast, thought_voice_id: "inner" })).toBe("inner");
  });

  it("voiceChangedSince names today's voice only when it differs on a full-cast recording", () => {
    const row = { type: "dialogue" as const, speaker: "mara", voice_id: "narr" };
    expect(voiceChangedSince(row, cast, true)).toBe("v-mara");
    expect(voiceChangedSince({ ...row, voice_id: "v-mara" }, cast, true)).toBeNull();
    expect(voiceChangedSince(row, cast, false)).toBeNull(); // one-narrator audio
    expect(voiceChangedSince(row, null, true)).toBeNull(); // not cast
    expect(voiceChangedSince({ ...row, voice_id: null }, cast, true)).toBeNull(); // no recording
  });
});

describe("resume where you left off", () => {
  it("parseResume accepts a well-formed point and rejects anything else without throwing", () => {
    const ok = { chapter: 2, clip: "b:0", offset: 1.5, seconds: 40 };
    expect(parseResume(JSON.stringify(ok))).toEqual(ok);
    expect(parseResume(null)).toBeNull();
    expect(parseResume("{not json")).toBeNull();
    expect(parseResume("null")).toBeNull();
    expect(parseResume(JSON.stringify({ ...ok, chapter: 0 }))).toBeNull();
    expect(parseResume(JSON.stringify({ ...ok, offset: -1 }))).toBeNull();
    expect(parseResume(JSON.stringify({ ...ok, clip: 4 }))).toBeNull();
  });

  it("resumeTarget prefers the same clip, else the same seconds into the chapter", () => {
    const clips = [
      { key: "a:0", duration: 10 },
      { key: "b:0", duration: 20 },
    ];
    expect(resumeTarget(clips, { clip: "b:0", offset: 4, seconds: 14 })).toEqual({ index: 1, offset: 4 });
    expect(resumeTarget(clips, { clip: "gone:0", offset: 4, seconds: 14 })).toEqual({ index: 1, offset: 4 });
    expect(resumeTarget(clips, { clip: "gone:0", offset: 0, seconds: 5 })).toEqual({ index: 0, offset: 5 });
    // past the end: the last clip (the player clamps inside it)
    expect(resumeTarget(clips, { clip: "gone:0", offset: 0, seconds: 99 })).toEqual({ index: 1, offset: 89 });
    expect(resumeTarget([], { clip: "a:0", offset: 0, seconds: 0 })).toBeNull();
    expect(chapterSeconds([10, 20, 30], 2, 5)).toBe(35);
  });

  it("clock and fileSize read like a person wrote them", () => {
    expect(clock(0)).toBe("0:00");
    expect(clock(754.9)).toBe("12:34");
    expect(clock(3723)).toBe("1:02:03");
    expect(fileSize(412 * 1024 * 1024)).toBe("412 MB");
    expect(fileSize(1.5 * 1024 ** 3)).toBe("1.5 GB");
    expect(fileSize(10)).toBe("1 KB");
  });
});
