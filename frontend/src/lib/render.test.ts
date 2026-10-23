import { describe, expect, it } from "vitest";

import {
  choiceForVoice,
  choiceVoiceId,
  MODE_LABEL,
  misheardWord,
  respelledWords,
  narratorSpec,
  narratorTakesSpeed,
  singleParams,
  wholeBookConfirm,
} from "./render";

const voices = [
  { voice_id: "v-kira", engine: "chatterbox" },
  { voice_id: "v-bella", engine: "kokoro" },
];

describe("modes", () => {
  it("are named in plain words", () => {
    expect(MODE_LABEL).toEqual({ multivoice: "Full cast", single: "One narrator" });
  });
});

describe("narratorSpec", () => {
  it("the standard narrator at normal speed is exactly the server default {}", () => {
    expect(narratorSpec("default", 1, voices)).toEqual({});
    expect(narratorSpec("default", 1.1, voices)).toEqual({ speed: 1.1 });
  });

  it("a built-in preset names kokoro and its id, with speed when not normal", () => {
    expect(narratorSpec("preset:am_adam", 1, voices)).toEqual({ engine: "kokoro", voice: "am_adam" });
    expect(narratorSpec("preset:am_adam", 0.9, voices)).toEqual({ engine: "kokoro", voice: "am_adam", speed: 0.9 });
  });

  it("a library voice carries its own engine and never a speed (it uses its saved settings)", () => {
    expect(narratorSpec("voice:v-kira", 1.25, voices)).toEqual({ engine: "chatterbox", voice: "v-kira" });
    expect(narratorTakesSpeed("voice:v-kira")).toBe(false);
    expect(narratorTakesSpeed("preset:af_heart")).toBe(true);
    expect(narratorTakesSpeed("default")).toBe(true);
  });

  it("estimate query params mirror the spec", () => {
    expect(singleParams(undefined)).toBe("");
    expect(singleParams({})).toBe("");
    expect(singleParams({ engine: "kokoro", voice: "am_adam", speed: 1.1 })).toBe("&engine=kokoro&voice=am_adam&speed=1.1");
  });

  it("finds the picker value for a recorded voice, and the voice id for a choice", () => {
    expect(choiceForVoice("v-kira", voices, ["af_heart"])).toBe("voice:v-kira");
    expect(choiceForVoice("af_heart", voices, ["af_heart"])).toBe("preset:af_heart");
    expect(choiceForVoice("gone", voices, ["af_heart"])).toBeNull();
    expect(choiceForVoice(null, voices, [])).toBeNull();
    expect(choiceVoiceId("default", "af_heart")).toBe("af_heart");
    expect(choiceVoiceId("default", undefined)).toBeNull();
    expect(choiceVoiceId("voice:v-kira", undefined)).toBe("v-kira");
    expect(choiceVoiceId("preset:am_adam", "af_heart")).toBe("am_adam");
  });
});

describe("wholeBookConfirm", () => {
  const chapters = [{ speakable_blocks: 60 }, { speakable_blocks: 50 }];

  it("asks for a whole-book recording over the limit, with the paragraphs and runtime", () => {
    expect(wholeBookConfirm([], chapters, 100, 3600)).toEqual({ speakable_blocks: 110, runtime_estimate_seconds: 3600 });
  });

  it("never asks for a chapter range, at or under the limit, or when the limit is unknown", () => {
    expect(wholeBookConfirm([1], chapters, 100, 3600)).toBeNull();
    expect(wholeBookConfirm([], chapters, 110, 3600)).toBeNull();
    expect(wholeBookConfirm([], chapters, undefined, 3600)).toBeNull();
    expect(wholeBookConfirm([], null, 100, 3600)).toBeNull();
  });

  it("a missing runtime estimate reads as 0 (the sheet then leaves the hours out)", () => {
    expect(wholeBookConfirm([], chapters, 100, null)?.runtime_estimate_seconds).toBe(0);
  });
});

describe("misheardWord", () => {
  it("is the first book word the check did not hear, outer punctuation trimmed", () => {
    expect(misheardWord("Kael'thas raised a hand.", "Kale has raised a hand.")).toBe("Kael'thas");
    expect(misheardWord("He drew the blade.", "He threw the blade.")).toBe("drew");
  });

  it("suggests nothing for failures a pronunciation entry can't fix", () => {
    // the scorer's own reasons: a number, negation, loop or cut-off is a misread, not a word to teach
    for (const reason of [
      "wrong number: '12' heard as '13'",
      "number missing: 'twelve' not heard",
      "negation missing: 'not' not heard",
      "meaning flipped: 'can' heard as 'cannot'",
      "repeated words: 'the door the door'",
      "the end was cut off: 'raised a hand' not heard",
    ]) {
      expect(misheardWord("Kael'thas raised a hand.", "Kale has raised.", reason)).toBeNull();
    }
    expect(misheardWord("Kael'thas raised a hand.", "Kale has raised a hand.", "2 of 4 words differ from the text")).toBe(
      "Kael'thas",
    );
  });

  it("skips number words, digits and short common words", () => {
    expect(misheardWord('"Twelve," she said.', "13, she said.")).toBeNull();
    expect(misheardWord("He was not at the gate of Ostrava.", "He was at gate of Ostrova.")).toBe("Ostrava");
    expect(misheardWord("Room 101 is in Zhytomyr.", "Room is in Jitomir.")).toBe("Zhytomyr");
  });

  it("never offers a word that is already a lexicon respelling", () => {
    const taught = respelledWords([{ respelling: "Kale-thass" }, { respelling: "Hermy Oh-nee" }]);
    expect(misheardWord("Kale-thass raised Ostrava's sword.", "Kales raised Ostrova sword.", null, taught)).toBe(
      "Ostrava's",
    );
    expect(misheardWord("Kale-thass raised a hand.", "Kales raised a hand.", null, taught)).toBeNull();
  });

  it("ignores case and punctuation, and is null when every word was heard", () => {
    expect(misheardWord("It was late.", "it was LATE")).toBeNull();
    expect(misheardWord("", "anything")).toBeNull();
  });
});
