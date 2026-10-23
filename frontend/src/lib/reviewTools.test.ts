import { describe, expect, it } from "vitest";

import type { CharacterSummary, SegmentRow } from "../api/types";
import {
  canPlay,
  cannotBeSame,
  castFamily,
  characterHints,
  describeEdit,
  describeEditWarning,
  genderWarning,
  isConfirmed,
  mergePreview,
  mergeSuggestions,
  reprStrings,
} from "./reviewTools";

function char(id: string, name: string, line_count: number, extra: Partial<CharacterSummary> = {}): CharacterSummary {
  return {
    id,
    name,
    aliases: [],
    gender: null,
    age_hint: null,
    line_count,
    sample_lines: [],
    first_appearance: "ch001_b0001",
    ...extra,
  };
}

const row = (o: Partial<SegmentRow> = {}): SegmentRow => ({
  block_id: "ch001_b0001",
  segment_index: 0,
  type: "dialogue",
  speaker: "a",
  speaker_name: "A",
  text: "x",
  confidence: 0.5,
  unattributed_quote: false,
  has_audio: false,
  audio_segment: null,
  duration_seconds: null,
  voice_id: null,
  audio_key: null,
  ...o,
});

describe("line state", () => {
  it("a line at confidence 1.0 is confirmed; below is not", () => {
    expect(isConfirmed(row({ confidence: 1 }))).toBe(true);
    expect(isConfirmed(row({ confidence: 0.99 }))).toBe(false);
  });

  it("a line is playable only with audio, a segment index and a duration", () => {
    expect(canPlay(row({ has_audio: true, audio_segment: 0, duration_seconds: 1.5 }))).toBe(true);
    expect(canPlay(row({ has_audio: true, audio_segment: null, duration_seconds: 1.5 }))).toBe(false);
    expect(canPlay(row({ has_audio: false, audio_segment: 0, duration_seconds: 1.5 }))).toBe(false);
    expect(canPlay(row({ has_audio: true, audio_segment: 0, duration_seconds: null }))).toBe(false);
  });
});

describe("gender and age hints", () => {
  it("mirrors the caster: only male-ish words get the male family, everything else female", () => {
    for (const g of ["male", "M", "man", " boy "]) expect(castFamily(g)).toBe("male");
    for (const g of ["female", "f", null, undefined, "", "unknown", "nonbinary"]) expect(castFamily(g)).toBe("female");
  });

  it("warns when the gender is unknown, or a word the caster doesn't know; never for male or female", () => {
    expect(genderWarning(null)).toBe(
      "Gender unknown, so automatic casting gives them a female voice. If that's wrong, pick a voice for them by hand.",
    );
    expect(genderWarning("unknown")).toMatch(/Gender unknown/);
    expect(genderWarning("nonbinary")).toMatch(/"nonbinary" gets a female voice\. If that's wrong/);
    // words that mean female already get the right family: nothing to warn about
    for (const g of ["f", "Woman", "girl", "lady"]) expect(genderWarning(g)).toBeNull();
    expect(genderWarning("male")).toBeNull();
    expect(genderWarning("boy")).toBeNull();
  });

  it("chips say the gender in words, 'gender unknown' when absent, and the age hint when there is one", () => {
    expect(characterHints({ gender: "f", age_hint: "child" })).toEqual({
      gender: "female",
      genderKnown: true,
      age: "child",
    });
    expect(characterHints({ gender: null, age_hint: "  " })).toEqual({
      gender: "gender unknown",
      genderKnown: false,
      age: null,
    });
  });
});

describe("reprStrings", () => {
  it("reads single- and double-quoted Python reprs, unescaping them", () => {
    expect(reprStrings(`['Mr. Smith', "O'Neil", 'it\\'s']`)).toEqual(["Mr. Smith", "O'Neil", "it's"]);
    expect(reprStrings("no quotes")).toEqual([]);
  });
});

describe("mergeSuggestions", () => {
  const cast = [
    char("smith", "Mr. Smith", 30),
    char("smith-2", "Mrs. Smith", 4),
    char("tom", "Tom", 12, { aliases: ["Tommy"] }),
    char("thomas", "Thomas", 2),
    char("anna", "Anna", 9),
  ];

  it("an ambiguous alias note is NOT a card: the server writes it only for a pair it judged to be different people", () => {
    expect(
      mergeSuggestions(["alias: ambiguous 'smith' shared by ['Mr. Smith', 'Mrs. Smith'] — not merged, review"], cast),
    ).toEqual([]);
  });

  it("a pair the server's gender/title guard would veto never becomes a card, whatever the note", () => {
    const people = [
      char("smith", "Mr. Smith", 30),
      char("smith-2", "Mrs. Smith", 4),
      char("jo", "Jo", 5, { gender: "girl" }),
      char("joe", "Joe", 8, { gender: "male" }),
      char("lucas", "Lady Lucas", 3),
      char("lucas-2", "Miss Lucas", 3),
    ];
    expect(
      mergeSuggestions(
        [
          "not merging 'Mrs. Smith' with existing ['Mr. Smith'] (flagged for review, not auto-applied)",
          "alias: adjudicator approved 'Jo' <-> 'Joe' at confidence 0.6 < 0.8 — flagged, not merged",
          "not merging 'Miss Lucas' with existing ['Lady Lucas'] (flagged for review, not auto-applied)",
        ],
        people,
      ),
    ).toEqual([]);
    expect(cannotBeSame({ name: "Mr. Bennet", gender: null }, { name: "Elizabeth Bennet", gender: "f" })).toBe(true);
    expect(cannotBeSame({ name: "Tom", gender: null }, { name: "Thomas", gender: "m" })).toBe(false);
  });

  it("a low-confidence approval and a registry 'not merging' note are cards too", () => {
    const out = mergeSuggestions(
      [
        "alias: adjudicator approved 'Thomas' <-> 'Tom' at confidence 0.62 < 0.80 — flagged, not merged",
        "not merging 'Anna' with existing ['Tom', 'Mr. Smith'] (flagged for review, not auto-applied)",
      ],
      cast,
    );
    expect(out.map((s) => `${s.loser.id}>${s.winner.id}`)).toEqual(["thomas>tom", "anna>tom", "anna>smith"]);
    expect(out[0].why).toMatch(/AI thought/);
    expect(out[1].why).toMatch(/didn't merge them on its own/);
  });

  it("rejected, vetoed, merged and non-alias notes are not cards", () => {
    expect(
      mergeSuggestions(
        [
          "alias: adjudicator rejected 'Thomas' <-> 'Tom' (nickname) — not merged",
          "alias: vetoed 'Thomas' <-> 'Anna' despite approval — gender/generation clash",
          "alias: merged 'Thomas' -> 'Tom' (LLM nickname, conf 0.91)",
          "alias: low-evidence character 'anna' ('Anna') — no metadata, no attributed lines, possible hallucination, review",
          "dialogue convention: single curly quotes (‘…’) detected",
        ],
        cast,
      ),
    ).toEqual([]);
  });

  it("a pair already merged (one name is now the other's alias) or gone drops out; duplicates collapse", () => {
    const out = mergeSuggestions(
      [
        "alias: adjudicator approved 'Tommy' <-> 'Tom' at confidence 0.6 < 0.8 — flagged, not merged",
        "alias: adjudicator approved 'Ghost' <-> 'Tom' at confidence 0.6 < 0.8 — flagged, not merged",
        "alias: adjudicator approved 'Tom' <-> 'Thomas' at confidence 0.6 < 0.8 — flagged, not merged",
        "alias: adjudicator approved 'Thomas' <-> 'Tom' at confidence 0.7 < 0.8 — flagged, not merged",
      ],
      cast,
    );
    expect(out).toHaveLength(1);
    expect(out[0].key).toBe("thomas|tom");
  });

  it("the preview says how many lines move, and handles a character with none", () => {
    expect(mergePreview(char("x", "Bob", 17), "Alice")).toBe(
      'Moves 17 lines from Bob to Alice. "Bob" stays as another name for Alice.',
    );
    expect(mergePreview(char("x", "Bob", 1), "Alice")).toMatch(/^Moves 1 line from Bob/);
    expect(mergePreview(char("x", "Bob", 0), "Alice")).toBe('Bob has no lines; "Bob" becomes another name for Alice.');
  });
});

describe("describeEdit", () => {
  const names: Record<string, string> = { alice: "Alice", bob: "Bob" };
  const nameOf = (id: string) => names[id] ?? null;

  it("names each kind of change in words, with the line it touched", () => {
    expect(describeEdit({ op: "rename", character_id: "alice", new_name: "Alicia", expected_name: "Alice" }, nameOf)).toEqual({
      text: 'Renamed "Alice" to "Alicia"',
      target: null,
    });
    expect(
      describeEdit(
        { op: "merge", loser_id: "ali", winner_id: "alice", expected_loser_name: "Ali", expected_winner_name: "Alice" },
        nameOf,
      ).text,
    ).toBe("Merged Ali into Alice");
    expect(
      describeEdit(
        { op: "reassign", block_id: "ch003_b0012", segment_index: 1, speaker: "bob", text_anchor: '"Hi," he said.' },
        nameOf,
      ),
    ).toEqual({ text: 'Bob says ""Hi," he said."', target: { chapter: 3, block: "ch003_b0012", seg: 1 } });
    expect(
      describeEdit({ op: "reassign", block_id: "ch003_b0012", segment_index: 0, speaker: null, text_anchor: "Rain." }, nameOf)
        .text,
    ).toBe('Narrator reads "Rain."');
    expect(
      describeEdit(
        {
          op: "set_emotion",
          block_id: "ch001_b0001",
          segment_index: 1,
          emotion: { label: "angry", intensity: 3 },
          text_anchor: "No!",
        },
        nameOf,
      ).text,
    ).toBe('Emotion set to angry (strongly) on "No!"');
    expect(
      describeEdit({ op: "set_emotion", block_id: "ch001_b0001", segment_index: 1, emotion: null, text_anchor: "No!" }, nameOf)
        .text,
    ).toBe('Emotion cleared on "No!"');
  });

  it("with spoiler-safe hiding a character, a rename or merge of it leaves out the stored names", () => {
    const hidden = (id: string) => id === "alice";
    const masked = (id: string) => (hidden(id) ? "a hidden character" : nameOf(id));
    expect(
      describeEdit({ op: "rename", character_id: "alice", new_name: "Alicia", expected_name: "Alice" }, masked, hidden).text,
    ).toBe("Renamed a hidden character");
    expect(
      describeEdit(
        { op: "merge", loser_id: "ali", winner_id: "alice", expected_loser_name: "Ali", expected_winner_name: "Alice" },
        masked,
        hidden,
      ).text,
    ).toBe("Merged a name into a hidden character");
    // a character that isn't hidden still reads in full
    expect(
      describeEdit({ op: "rename", character_id: "bob", new_name: "Robert", expected_name: "Bob" }, masked, hidden).text,
    ).toBe('Renamed "Bob" to "Robert"');
  });

  it("a long anchor is marked as cut off; a character gone from the cast is said plainly", () => {
    const long = "x".repeat(60);
    expect(
      describeEdit({ op: "reassign", block_id: "ch001_b0001", segment_index: 0, speaker: "zed", text_anchor: long }, nameOf)
        .text,
    ).toBe(`a character no longer in the cast says "${long}…"`);
    expect(describeEdit({ op: "mystery" }, nameOf).text).toMatch(/doesn't recognise/);
  });
});

describe("describeEditWarning", () => {
  it("a reassign whose line changed links to that line", () => {
    const d = describeEditWarning(
      "reassign skipped: 'ch002_b0003'[1] is now different text than when the edit was recorded — re-record it if still wanted",
    );
    expect(d.text).toMatch(/^A speaker fix no longer applies: the line changed/);
    expect(d.target).toEqual({ chapter: 2, block: "ch002_b0003", seg: 1 });
  });

  it("a line index that no longer exists links to its paragraph", () => {
    const d = describeEditWarning(
      "set_emotion skipped: block 'ch004_b0001' has 1 segment(s), index 2 no longer exists",
    );
    expect(d.text).toMatch(/^An emotion tag no longer applies/);
    expect(d.target).toEqual({ chapter: 4, block: "ch004_b0001", seg: null });
  });

  it("warnings with nothing to open have no target but still read plainly, keeping the raw text", () => {
    const gone = describeEditWarning("reassign skipped: block 'ch009_b0001' not in this attribution");
    expect(gone.target).toBeNull();
    expect(gone.text).toMatch(/paragraph is no longer in the book/);
    expect(describeEditWarning("merge skipped: character 'x' not distinct/present in this attribution").text).toMatch(
      /^A merge no longer applies: a character/,
    );
    const rename = describeEditWarning("rename skipped: 'a' is now 'Bea', not 'Al' — re-record the edit if still wanted");
    expect(rename.text).toMatch(/^A rename no longer applies: the characters changed/);
    expect(rename.raw).toMatch(/^rename skipped/);
    expect(describeEditWarning("something new").text).toBe("An earlier fix no longer applies.");
  });
});
