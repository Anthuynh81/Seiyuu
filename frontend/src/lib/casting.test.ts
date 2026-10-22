import { describe, expect, it } from "vitest";

import type { VoiceAssignment } from "../api/types";
import {
  castingDiffers,
  castingFromServer,
  castingVoicesDiffer,
  autoNarratorId,
  castRowStatus,
  proposalMatchesSaved,
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

describe("suggest-a-cast row status", () => {
  const saved: VoiceAssignment = {
    ...server,
    assignments: { mrs_bennet: "af_nicole", mr_bennet: "mr_bennet_auto" },
  };
  const ctx = { saved, wouldCreate: ["lydia_auto"], wouldRecast: ["mr_bennet_auto"], recast: false };

  it("a voice that must be made is new", () => {
    expect(castRowStatus("lydia", "lydia_auto", ctx)).toBe("new");
  });

  it("a saved hand pick that stays is 'keeps your pick'", () => {
    expect(castRowStatus("mrs_bennet", "af_nicole", ctx)).toBe("keeps your pick");
  });

  it("an existing auto voice is unchanged — until re-cast rebuilds it", () => {
    expect(castRowStatus("mr_bennet", "mr_bennet_auto", ctx)).toBe("unchanged");
    expect(castRowStatus("mr_bennet", "mr_bennet_auto", { ...ctx, recast: true })).toBe("will change");
  });

  it("a different voice than the saved one will change", () => {
    expect(castRowStatus("mr_bennet", "am_adam", ctx)).toBe("will change");
  });

  it("no saved voice in this book is 'new', even when its auto voice already exists elsewhere", () => {
    // a series sibling / leftover kitty_auto: the server lists it in would_recast, not would_create
    const reuse = { saved: null, wouldCreate: [], wouldRecast: ["kitty_auto"], recast: false };
    expect(castRowStatus("kitty", "kitty_auto", reuse)).toBe("new");
    expect(castRowStatus("kitty", "kitty_auto", { ...reuse, recast: true })).toBe("new");
    // a newly found character in a book that already has a cast
    expect(castRowStatus("kitty", "kitty_auto", { ...ctx, wouldRecast: ["kitty_auto"] })).toBe("new");
  });

  it("a hand pick with nothing saved for that character is 'your pick'", () => {
    expect(castRowStatus("kitty", "am_adam", ctx)).toBe("your pick");
    expect(castRowStatus("kitty", "am_adam", { ...ctx, saved: null })).toBe("your pick");
  });
});

describe("proposalMatchesSaved", () => {
  const opts = { wouldCreate: [] as string[], recast: false };

  it("is true only when the proposal writes exactly the saved voices", () => {
    expect(proposalMatchesSaved({ ...server, stage: "final" }, server, opts)).toBe(true);
    const moved = { ...server, assignments: { ...server.assignments, mr_bennet: "am_adam" } };
    expect(proposalMatchesSaved(moved, server, opts)).toBe(false);
  });

  it("never a no-op with nothing saved, a voice to make, or re-cast on", () => {
    expect(proposalMatchesSaved(server, null, opts)).toBe(false);
    expect(proposalMatchesSaved(server, server, { ...opts, wouldCreate: ["x_auto"] })).toBe(false);
    expect(proposalMatchesSaved(server, server, { ...opts, recast: true })).toBe(false);
  });
});
