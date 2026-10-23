import { describe, expect, it } from "vitest";

import { makeSystemProbe } from "../test/fixtures";
import { attributeBlocker, chapterSpan, parseChapterList, promptChanged, sameReaderReason } from "./attribute";

const local = { provider: "local" as const, model: "qwen2.5:7b" };

describe("attributeBlocker", () => {
  it("blocks a local run with ONE reason and the fix command when Ollama is down", () => {
    const probe = makeSystemProbe({ ollama: { base_url: "x", reachable: false, model_installed: null } });
    expect(attributeBlocker({ probe, reader: local })).toEqual({
      reason: expect.stringMatching(/Ollama isn't running/),
      command: "ollama serve",
    });
  });

  it("blocks a local run with the pull command when the configured model isn't downloaded", () => {
    const probe = makeSystemProbe({ ollama: { base_url: "x", reachable: true, model_installed: false } });
    expect(attributeBlocker({ probe, reader: local })).toEqual({
      reason: expect.stringMatching(/qwen2.5:7b\) isn't downloaded/),
      command: "ollama pull qwen2.5:7b",
    });
    // the probe only knows the configured model: another one is left to the server's pre-flight
    expect(attributeBlocker({ probe, reader: { provider: "local", model: "llama3.1:8b" } })).toBeNull();
  });

  it("with Claude as the server default, checks the LOCAL model the probe actually probed", () => {
    // provider=anthropic: probe.attribution.model is the Claude id, but model_installed is
    // about settings.attribution_model (the local model)
    const probe = makeSystemProbe({
      attribution: { provider: "anthropic", model: "claude-test", anthropic_model: "claude-test", prompt_version: "v6", hybrid: false },
      ollama: { base_url: "x", reachable: true, model_installed: false },
    });
    // never "ollama pull claude-…"
    expect(attributeBlocker({ probe, reader: { provider: "local", model: "claude-test" } })).toBeNull();
    // the local model is unknown without /api/settings: nothing to claim
    expect(attributeBlocker({ probe, reader: local })).toBeNull();
    expect(attributeBlocker({ probe, reader: local, localModel: "qwen2.5:7b" })).toEqual({
      reason: expect.stringMatching(/qwen2.5:7b\) isn't downloaded/),
      command: "ollama pull qwen2.5:7b",
    });
  });

  it("never blocks Claude, a healthy probe, an unknown probe, or a probe that errored", () => {
    const down = makeSystemProbe({ ollama: { base_url: "x", reachable: false, model_installed: null } });
    expect(attributeBlocker({ probe: down, reader: { provider: "anthropic", model: "claude" } })).toBeNull();
    expect(attributeBlocker({ probe: makeSystemProbe(), reader: local })).toBeNull();
    expect(attributeBlocker({ probe: undefined, reader: local })).toBeNull();
    expect(attributeBlocker({ probe: down, probeError: new Error("boom"), reader: local })).toBeNull();
    expect(
      attributeBlocker({ probe: makeSystemProbe({ ollama: { base_url: "x", reachable: null } }), reader: local }),
    ).toBeNull();
  });
});

describe("parseChapterList", () => {
  it("reads numbers and ranges, sorted without repeats", () => {
    expect(parseChapterList("7, 3 5-6,3", 10)).toEqual({ chapters: [3, 5, 6, 7] });
    expect(parseChapterList("2–4", 10)).toEqual({ chapters: [2, 3, 4] });
    // spaces around the dash still make a range
    expect(parseChapterList("5 - 7", 10)).toEqual({ chapters: [5, 6, 7] });
    expect(parseChapterList("5 -7, 2– 3", 10)).toEqual({ chapters: [2, 3, 5, 6, 7] });
  });

  it("explains empty, malformed, backwards and out-of-range input in plain words", () => {
    expect(parseChapterList("  ", 10).error).toMatch(/List the chapters/);
    expect(parseChapterList("three", 10).error).toMatch(/"three" isn't a chapter number/);
    expect(parseChapterList("6-2", 10).error).toMatch(/runs backwards/);
    expect(parseChapterList("9-12", 10).error).toMatch(/chapters 1 to 10/);
    expect(parseChapterList("0", 10).error).toMatch(/out of range/);
  });
});

describe("sameReaderReason", () => {
  const current = { provider_id: "local", model_id: "qwen2.5:7b", prompt_version: "v6" };

  it("refuses the reader that produced the current speakers (the cache would replay it)", () => {
    expect(sameReaderReason(current, local, "v6")).toMatch(/same model gives the same answer/);
    expect(sameReaderReason(current, { provider: "local", model: " Qwen2.5:7B " }, "v6")).not.toBeNull();
  });

  it("accepts a different model or a different provider", () => {
    expect(sameReaderReason(current, { provider: "local", model: "llama3.1:8b" }, "v6")).toBeNull();
    expect(sameReaderReason(current, { provider: "anthropic", model: "qwen2.5:7b" }, "v6")).toBeNull();
  });

  it("allows the same model once Seiyuu's prompt has changed (the cache key includes it)", () => {
    expect(sameReaderReason(current, local, "v7")).toBeNull();
    expect(promptChanged(current, "v7")).toBe(true);
    expect(promptChanged(current, "v6")).toBe(false);
    expect(promptChanged(current, undefined)).toBe(false);
  });

  it("asks for a model name when it is blank", () => {
    expect(sameReaderReason(current, { provider: "local", model: "  " }, "v6")).toMatch(/Type the model/);
  });
});

describe("chapterSpan", () => {
  it("names the span in plain words", () => {
    expect(chapterSpan([])).toBe("the whole book");
    expect(chapterSpan([4])).toBe("chapter 4");
    expect(chapterSpan([4, 5])).toBe("2 chapters");
  });
});
