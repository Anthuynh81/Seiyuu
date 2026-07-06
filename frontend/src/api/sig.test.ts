import { describe, expect, it } from "vitest";

import { resolvedSig } from "./hooks";

/** resolvedSig backs the read-along's apply effect: its output is a dependency, so it MUST
    change whenever a resolved clip's audio identity changes — otherwise a re-render/reassignment
    that swaps a wav (same clip key, new audio_key) leaves stale whisper timings in place and the
    read-along drifts permanently. */
describe("resolvedSig — the apply-effect signature folds audio_key in", () => {
  it("changes when a clip's audioKey changes even though the SET of keys is identical", () => {
    const before = resolvedSig([
      { key: "b1:0", audioKey: "aaa" },
      { key: "b1:1", audioKey: "bbb" },
    ]);
    const afterRerender = resolvedSig([
      { key: "b1:0", audioKey: "aaa" },
      { key: "b1:1", audioKey: "ZZZ" }, // same clip key, wav re-rendered
    ]);
    expect(afterRerender).not.toBe(before);
  });

  it("is order-independent — clip iteration order must not perturb the signature", () => {
    const a = resolvedSig([
      { key: "b1:0", audioKey: "aaa" },
      { key: "b2:0", audioKey: "bbb" },
    ]);
    const b = resolvedSig([
      { key: "b2:0", audioKey: "bbb" },
      { key: "b1:0", audioKey: "aaa" },
    ]);
    expect(a).toBe(b);
  });

  it("is stable when nothing changed, and reflects a clip dropping out (404 / still loading)", () => {
    const full = [
      { key: "b1:0", audioKey: "aaa" },
      { key: "b1:1", audioKey: "bbb" },
    ];
    expect(resolvedSig(full)).toBe(resolvedSig([...full]));
    expect(resolvedSig([{ key: "b1:0", audioKey: "aaa" }])).not.toBe(resolvedSig(full));
  });

  it("distinguishes a null audioKey from an empty-string key without colliding", () => {
    expect(resolvedSig([{ key: "b1:0", audioKey: null }])).toBe("b1:0@");
    expect(resolvedSig([])).toBe("");
  });
});
