import { describe, expect, it } from "vitest";

import { makeBook } from "../test/fixtures";
import { engineSubject, finishedHeadline, liveJobLine, stepName, subjectName } from "./jobCopy";

describe("jobCopy", () => {
  const books = [makeBook({ book_id: "b1", title: "The Hollow Crown" }), makeBook({ book_id: "b2", title: null })];

  it("names a book by its title, an engine by its name, never by id", () => {
    expect(subjectName({ book_id: "b1" }, books)).toBe("The Hollow Crown");
    expect(subjectName({ book_id: "engine:chatterbox" }, books)).toBe("Chatterbox");
    expect(subjectName({ book_id: "b2" }, books)).toBe("your book");
    expect(subjectName({ book_id: "gone" }, undefined)).toBe("your book");
    expect(engineSubject("engine:kokoro")).toBe("kokoro");
    expect(engineSubject("b1")).toBeNull();
  });

  it.each([
    ["attribute", "succeeded", "Found the speakers in T"],
    ["render", "succeeded", "Recorded T"],
    ["assemble", "succeeded", "Built the chapter MP3s for T"],
    ["master", "succeeded", "Built the .m4b audiobook for T"],
    ["warmup", "succeeded", "T is downloaded and ready"],
    ["attribute", "failed", "Couldn't find the speakers in T"],
    ["render", "canceled", "Stopped recording T"],
  ] as const)("%s %s -> %s", (kind, state, text) => {
    expect(finishedHeadline({ kind, state }, "T")).toBe(text);
  });

  it("describes a live job in plain words", () => {
    expect(liveJobLine({ kind: "render", state: "running" })).toBe("Recording…");
    expect(liveJobLine({ kind: "attribute", state: "queued" })).toBe("Finding the speakers — waiting its turn");
    expect(stepName("master")).toBe("Building the .m4b audiobook");
    expect(stepName("render")).toBe("Recording");
  });
});
