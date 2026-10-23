import { describe, expect, it } from "vitest";

import type { ActiveJobSummary, JobKind } from "../api/types";
import { makeBook, makeJob } from "../test/fixtures";
import { liveJobLine } from "./jobCopy";
import { ADD_A_BOOK, bookHref, nextStep, type BookFlags } from "./nextStep";

const flags = (o: Partial<BookFlags> = {}): BookFlags => makeBook({ book_id: "b 1", ...o });

describe("nextStep", () => {
  it("no book at all: add one in the Library", () => {
    expect(nextStep({ status: undefined })).toEqual(ADD_A_BOOK);
    expect(nextStep({ status: null })).toEqual(ADD_A_BOOK);
  });

  it.each([
    ["a book with no text", { ingested: false }, "Add the book again", "/#add"],
    ["just imported", {}, "Find the speakers", "/render?book=b%201#find-speakers"],
    ["speakers found", { attributed: true }, "Cast the voices", "/review?book=b%201"],
    ["cast", { attributed: true, assigned: true }, "Record the audiobook", "/render?book=b%201#record"],
    ["recorded", { attributed: true, assigned: true, rendered: true }, "Build the .m4b", "/render?book=b%201#build"],
    // single-voice recordings skip speakers and casting: recorded wins over the earlier gaps
    ["recorded in one voice", { rendered: true }, "Build the .m4b", "/render?book=b%201#build"],
    ["chapter MP3s only", { rendered: true, assembled: true }, "Build the .m4b", "/render?book=b%201#build"],
    ["finished", { rendered: true, assembled: true, mastered: true }, "Listen", "/listen?book=b%201"],
  ] as const)("%s → %s", (_name, o, label, to) => {
    const step = nextStep({ status: flags(o) });
    expect(step.label).toBe(label);
    expect(step.to).toBe(to);
    expect(step.disabled).toBeFalsy();
  });

  it.each([
    ["attribute", "running", "Finding the speakers…", /bar at the bottom/],
    ["attribute", "queued", "Finding the speakers…", /waiting for the job ahead/],
    ["render", "running", "Recording…", /bar at the bottom/],
    ["assemble", "running", "Building the chapter MP3s…", /bar at the bottom/],
    ["master", "running", "Building the .m4b audiobook…", /bar at the bottom/],
  ] as const)("a live %s job (%s) disables the step: %s", (kind, state, label, reason) => {
    const activeJob: ActiveJobSummary = { job_id: "j9", kind: kind as JobKind, state };
    const step = nextStep({ status: flags({ attributed: true }), activeJob });
    expect(step).toMatchObject({ label, disabled: true });
    expect(step.reason).toMatch(reason);
    expect(step.label).not.toMatch(/j9/); // never the raw job id
  });

  it("the running label uses the same words as the card's live job line", () => {
    for (const kind of ["ingest", "attribute", "render", "assemble", "master"] as JobKind[]) {
      const step = nextStep({ status: flags(), activeJob: { kind, state: "running" } });
      expect(step.label).toBe(liveJobLine({ kind, state: "running" }));
    }
  });

  it("a live job outranks a failure", () => {
    const failed = makeJob({ kind: "attribute", state: "failed", error: "OllamaUnreachableError: x" });
    const step = nextStep({ status: flags(), activeJob: { kind: "attribute", state: "running" }, failed });
    expect(step.disabled).toBe(true);
  });

  it.each([
    ["attribute", {}, "Find the speakers again", "OllamaUnreachableError: Ollama isn't reachable", /Start Ollama/],
    [
      "render",
      { attributed: true, assigned: true },
      "Record again",
      "GpuBusyError: another seiyuu process holds the GPU",
      /Close it or wait/,
    ],
    ["master", { rendered: true }, "Build the .m4b again", "RuntimeError: ffmpeg not found on PATH", /Install ffmpeg/],
  ] as const)("a failed %s job turns the step into a retry with the plain fix", (kind, o, label, error, reason) => {
    const failed = makeJob({ kind: kind as JobKind, state: "failed", is_terminal: true, error });
    const step = nextStep({ status: flags(o), failed });
    expect(step.label).toBe(label);
    expect(step.reason).toMatch(reason);
    expect(step.disabled).toBeFalsy();
    expect(step.retryOf).toBe(kind);
  });

  it("a failed chapter-MP3 run (optional) leaves Build the .m4b alone — the .m4b doesn't need them", () => {
    const failed = makeJob({ kind: "assemble", state: "failed", is_terminal: true, error: "ValueError: nobody knows" });
    const step = nextStep({ status: flags({ rendered: true }), failed });
    expect(step.label).toBe("Build the .m4b");
    expect(step.retryOf).toBeUndefined();
  });

  it("a failure of another kind leaves the step alone", () => {
    // a render failed on a book that is now finished: the next step is still Listen
    const failed = makeJob({ kind: "render", state: "failed", error: "boom" });
    const step = nextStep({ status: flags({ rendered: true, mastered: true }), failed });
    expect(step.label).toBe("Listen");
    expect(step.retryOf).toBeUndefined();
  });

  describe("want: a screen about one step forces it", () => {
    it("a book recorded in one voice can still be sent to find its speakers", () => {
      for (const o of [{ rendered: true }, { rendered: true, assembled: true, mastered: true }]) {
        const step = nextStep({ status: flags(o), want: "attribute" });
        expect(step.label).toBe("Find the speakers");
        expect(step.to).toBe("/render?book=b%201#find-speakers");
      }
    });

    it("a failed speaker search still turns it into the retry", () => {
      const failed = makeJob({ kind: "attribute", state: "failed", error: "OllamaUnreachableError: x" });
      const step = nextStep({ status: flags({ rendered: true }), failed, want: "attribute" });
      expect(step.label).toBe("Find the speakers again");
      expect(step.retryOf).toBe("attribute");
    });

    it("a live job and a book with no text still win", () => {
      const live = nextStep({ status: flags(), activeJob: { kind: "render", state: "running" }, want: "attribute" });
      expect(live).toMatchObject({ label: "Recording…", disabled: true });
      expect(nextStep({ status: flags({ ingested: false }), want: "attribute" }).label).toBe("Add the book again");
    });
  });

  it("bookHref encodes the id and appends the anchor", () => {
    expect(bookHref("render", "a&b", "record")).toBe("/render?book=a%26b#record");
    expect(bookHref("review", "x")).toBe("/review?book=x");
  });
});
