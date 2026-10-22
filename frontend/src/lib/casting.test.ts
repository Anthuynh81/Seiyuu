import { describe, expect, it } from "vitest";

import type { VoiceAssignment } from "../api/types";
import {
  castingDiffers,
  castingFromServer,
  castingVoicesDiffer,
  autoNarratorId,
  isManualPick,
  manualPicks,
  resetLosses,
  slugify,
} from "./casting";

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

describe("manual picks (what keep_existing protects / a reset loses)", () => {
  it("only the character's own <id>_auto voice counts as auto", () => {
    expect(isManualPick("alice", "alice_auto")).toBe(false);
    expect(isManualPick("alice", "af_nicole")).toBe(true);
    // another character's auto voice is still a hand pick for THIS character
    expect(isManualPick("alice", "bob_auto")).toBe(true);
    // prefix lookalikes are not auto
    expect(isManualPick("alice", "alice_auto2")).toBe(true);
  });

  it("manualPicks lists only the hand picks, in map order", () => {
    expect(
      manualPicks({ mrs_bennet: "mrs_bennet_auto", lizzy: "af_nicole", mr_bennet: "lizzy_auto" }),
    ).toEqual([
      { characterId: "lizzy", voiceId: "af_nicole" },
      { characterId: "mr_bennet", voiceId: "lizzy_auto" },
    ]);
    expect(manualPicks(server.assignments)).toEqual([]);
    expect(manualPicks({})).toEqual([]);
  });

  it("autoNarratorId predicts the backend's narrator_{slugify(default preset)} exactly", () => {
    expect(slugify("af_heart")).toBe("af_heart");
    expect(slugify("  Mr. Darcy's Voice! ")).toBe("mr_darcy_s_voice");
    expect(slugify("???")).toBe("voice");
    expect(autoNarratorId("af_heart")).toBe("narrator_af_heart");
    expect(autoNarratorId("AF-Bella")).toBe("narrator_af_bella");
  });

  it("resetLosses: an all-auto casting loses nothing", () => {
    expect(resetLosses(server, "narrator_af_heart")).toEqual({ picks: [], narrator: null, thought: null });
  });

  it("resetLosses: a user voice named 'Narrator' (narrator_<hex>) is a loss, not auto", () => {
    // user voice ids are slugify(name)_<hex> (voices/library.py new_voice_id)
    const cloned: VoiceAssignment = { ...server, narrator_voice_id: "narrator_ab12" };
    expect(resetLosses(cloned, "narrator_af_heart").narrator).toBe("narrator_ab12");
  });

  it("resetLosses: an auto narrator from an older default preset is a loss", () => {
    // saved under the old default af_bella; a reset now writes narrator_af_heart
    const stale: VoiceAssignment = { ...server, narrator_voice_id: "narrator_af_bella" };
    expect(resetLosses(stale, "narrator_af_heart").narrator).toBe("narrator_af_bella");
  });

  it("resetLosses: an unknown reset narrator over-reports rather than hides the narrator", () => {
    expect(resetLosses(server, null).narrator).toBe("narrator_af_heart");
  });

  it("resetLosses reports picks, a hand-chosen narrator, and a thought voice", () => {
    const custom: VoiceAssignment = {
      ...server,
      narrator_voice_id: "my_narrator",
      thought_voice_id: "inner_voice",
      assignments: { ...server.assignments, mr_bennet: "am_adam" },
    };
    expect(resetLosses(custom, "narrator_af_heart")).toEqual({
      picks: [{ characterId: "mr_bennet", voiceId: "am_adam" }],
      narrator: "my_narrator",
      thought: "inner_voice",
    });
  });
});
