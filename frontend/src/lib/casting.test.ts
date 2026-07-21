import { describe, expect, it } from "vitest";

import type { VoiceAssignment } from "../api/types";
import { castingDiffers, castingFromServer, castingVoicesDiffer } from "./casting";

const server: VoiceAssignment = {
  schema_version: 1,
  book_id: "bk",
  stage: "draft",
  narrator_voice_id: "narrator_af_heart",
  assignments: { mrs_bennet: "mrs_bennet_auto", mr_bennet: "mr_bennet_auto" },
  thought_voice_id: null,
  created_at: "2026-07-04T00:00:00Z",
};

describe("casting dirty-tracking", () => {
  it("a fresh local copy is clean", () => {
    expect(castingDiffers(server, castingFromServer(server))).toBe(false);
  });

  it("key order does not fake dirtiness (JSON.stringify would have)", () => {
    const local = castingFromServer(server);
    local.map = { mr_bennet: "mr_bennet_auto", mrs_bennet: "mrs_bennet_auto" }; // reversed order
    expect(castingDiffers(server, local)).toBe(false);
  });

  it.each([
    ["narrator", (c: ReturnType<typeof castingFromServer>) => (c.narrator = "other_voice")],
    ["thought", (c: ReturnType<typeof castingFromServer>) => (c.thought = "inner_voice")],
    ["stage", (c: ReturnType<typeof castingFromServer>) => (c.stage = "final")],
    ["a character's voice", (c: ReturnType<typeof castingFromServer>) => (c.map.mrs_bennet = "af_nicole")],
  ])("changing %s arms the save key", (_what, mutate) => {
    const local = castingFromServer(server);
    mutate(local);
    expect(castingDiffers(server, local)).toBe(true);
  });

  it("adding or removing a character differs (re-draft after re-attribution)", () => {
    const added = castingFromServer(server);
    added.map.elizabeth = "elizabeth_auto";
    expect(castingDiffers(server, added)).toBe(true);

    const removed = castingFromServer(server);
    delete removed.map.mr_bennet;
    expect(castingDiffers(server, removed)).toBe(true);
  });

  it("castingFromServer copies the map — mutating local never aliases server state", () => {
    const local = castingFromServer(server);
    local.map.mrs_bennet = "changed";
    expect(server.assignments.mrs_bennet).toBe("mrs_bennet_auto");
  });
});

describe("castingVoicesDiffer (audio-staleness, stage-agnostic)", () => {
  it("a stage-only draft->final change is NOT a voice change (audio stays valid)", () => {
    const local = castingFromServer(server);
    local.stage = "final";
    expect(castingDiffers(server, local)).toBe(true); // still dirty -> save is armed
    expect(castingVoicesDiffer(server, local)).toBe(false); // ...but no audio is stale
  });

  it.each([
    ["narrator", (c: ReturnType<typeof castingFromServer>) => (c.narrator = "other_voice")],
    ["thought", (c: ReturnType<typeof castingFromServer>) => (c.thought = "inner_voice")],
    ["a character's voice", (c: ReturnType<typeof castingFromServer>) => (c.map.mrs_bennet = "af_nicole")],
    ["adding a character", (c: ReturnType<typeof castingFromServer>) => (c.map.elizabeth = "elizabeth_auto")],
  ])("a real voice change (%s) IS a voice change", (_what, mutate) => {
    const local = castingFromServer(server);
    mutate(local);
    expect(castingVoicesDiffer(server, local)).toBe(true);
  });
});
