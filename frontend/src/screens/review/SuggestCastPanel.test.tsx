import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, type Mock } from "vitest";

import type { SuggestInput } from "../../api/hooks";
import type { CharacterSummary, SuggestCastResponse, VoiceAssignment, VoiceOut } from "../../api/types";
import { makeQueryClient } from "../../test/utils";
import { SuggestCastPanel } from "./SuggestCastPanel";

function assignmentOf(assignments: Record<string, string>, narrator = "narrator_af_heart"): VoiceAssignment {
  return {
    schema_version: 1,
    book_id: "b1",
    stage: "draft",
    narrator_voice_id: narrator,
    assignments,
    thought_voice_id: null,
    created_at: "2026-07-11T00:00:00Z",
  };
}

function preview(
  assignments: Record<string, string>,
  { create = ["carol_auto"], recast = [] as string[] } = {},
): SuggestCastResponse {
  return {
    assignment: assignmentOf(assignments),
    would_create_voice_ids: create,
    would_recast_voice_ids: recast,
    edit_warnings: [],
  };
}

function character(id: string, name: string, gender: string | null = null, age: string | null = null): CharacterSummary {
  return {
    id,
    name,
    aliases: [],
    gender,
    age_hint: age,
    line_count: 3,
    sample_lines: [],
    first_appearance: "ch001_b0001",
  };
}

function voice(id: string, name: string): VoiceOut {
  return {
    voice_id: id,
    name,
    kind: "preset",
    engine: "kokoro",
    preset_id: "af_heart",
    blend: null,
    reference_audio: null,
    seed: 7,
    consent_attested: false,
    consent: null,
    tags: [],
    created_at: "2026-07-11T00:00:00Z",
    has_audition: false,
  };
}

const CHARACTERS = [
  character("alice", "Alice", "f", "adult"),
  character("bob", "Bob", "m"),
  character("carol", "Carol", "female", "child"),
];
const VOICES = [
  voice("af_nicole", "Nicole"),
  voice("am_adam", "Adam"),
  voice("alice_auto", "Alice"),
  voice("bob_auto", "Bob"),
  voice("carol_auto", "Carol"), // auto voices are named after their character
  voice("narrator_af_heart", "Narrator"),
];

type Done = (error: Error | null) => void;
type PreviewFn = Mock<(input: SuggestInput, done: Done) => void>;
/** The server accepts every re-request: the table "catches up" at once. */
const accepting = (): PreviewFn => vi.fn((_input: SuggestInput, done: Done) => done(null));
const lastInput = (fn: PreviewFn) => fn.mock.lastCall?.[0];

function renderPanel(
  p: SuggestCastResponse,
  {
    saved = null as VoiceAssignment | null,
    onPreview = accepting(),
    onApply = vi.fn(),
    masked = [] as string[],
    refreshing = false,
  } = {},
) {
  // each row's ▶ is a real audition mutation, so the panel needs a query client
  render(
    <QueryClientProvider client={makeQueryClient()}>
      <SuggestCastPanel
      preview={p}
      saved={saved}
      characters={CHARACTERS}
      isMasked={(id) => masked.includes(id)}
      voices={VOICES}
      refreshing={refreshing}
      unsavedEdits={false}
      applying={false}
      applyError={null}
      onPreview={onPreview}
      onApply={onApply}
      onDismiss={() => {}}
      />
    </QueryClientProvider>,
  );
  return { onPreview, onApply };
}

const rowOf = (name: RegExp) => screen.getByRole("row", { name });

describe("SuggestCastPanel kept-picks line", () => {
  it("counts the hand picks the apply will keep (plural)", () => {
    renderPanel(preview({ alice: "af_nicole", bob: "am_adam", carol: "carol_auto" }));
    expect(screen.getByText("Your 2 picks are kept")).toBeInTheDocument();
    // the create summary still renders alongside it
    expect(screen.getByText(/1 new/)).toBeInTheDocument();
  });

  it("singular for one pick", () => {
    renderPanel(preview({ alice: "af_nicole", carol: "carol_auto" }));
    expect(screen.getByText("Your 1 pick is kept")).toBeInTheDocument();
  });

  it("no line when every character is on its auto voice", () => {
    renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }));
    expect(screen.queryByText(/kept$/)).not.toBeInTheDocument();
  });
});

describe("SuggestCastPanel preview table", () => {
  it("one row per character: name, gender/age chips, voice by DISPLAY name, plain status", () => {
    const saved = assignmentOf({ alice: "af_nicole", bob: "bob_auto" });
    renderPanel(
      preview({ alice: "af_nicole", bob: "bob_auto", carol: "carol_new" }, { create: ["carol_new"], recast: ["bob_auto"] }),
      { saved },
    );
    const table = screen.getByRole("table", { name: "proposed voices" });
    // header + three characters
    expect(within(table).getAllByRole("row")).toHaveLength(4);

    const alice = rowOf(/Alice/);
    expect(within(alice).getByText("female")).toBeInTheDocument();
    expect(within(alice).getByText("adult")).toBeInTheDocument();
    expect(within(alice).getByRole("button", { name: /voice for Alice/ })).toHaveTextContent("Nicole · kokoro");
    expect(within(alice).getByText("keeps your pick")).toBeInTheDocument();

    const bob = rowOf(/Bob/);
    expect(within(bob).getByText("male")).toBeInTheDocument();
    expect(within(bob).getByText("unchanged")).toBeInTheDocument();

    // a voice that doesn't exist yet reads as a new voice for that character — never its raw id
    const carol = rowOf(/Carol/);
    expect(within(carol).getByText("child")).toBeInTheDocument();
    expect(within(carol).getByRole("button", { name: /voice for Carol/ })).toHaveTextContent("new voice for Carol");
    expect(within(carol).getByText("new")).toBeInTheDocument();
    expect(screen.queryByText(/carol_new/)).not.toBeInTheDocument();
    // "already cast" counts the saved cast, not the voices that happen to exist
    expect(screen.getByText(/1 new · 2 already cast/)).toBeInTheDocument();
  });

  it("ticking re-cast flips the existing auto voices to 'will change'", async () => {
    const user = userEvent.setup();
    const saved = assignmentOf({ alice: "af_nicole", bob: "bob_auto" });
    renderPanel(preview({ alice: "af_nicole", bob: "bob_auto" }, { create: [], recast: ["bob_auto"] }), { saved });
    expect(within(rowOf(/Bob/)).getByText("unchanged")).toBeInTheDocument();
    // nothing differs from the saved cast: apply is a no-op until something changes
    expect(screen.getByRole("button", { name: "nothing to apply" })).toBeDisabled();

    await user.click(screen.getByRole("checkbox", { name: /re-cast the 1 existing \(their lines are recorded again\)/ }));
    expect(within(rowOf(/Bob/)).getByText("will change")).toBeInTheDocument();
    expect(within(rowOf(/Alice/)).getByText("keeps your pick")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "apply cast" })).toBeEnabled();
  });

  it("a row whose saved voice differs from the proposal says 'will change'", () => {
    const saved = assignmentOf({ alice: "alice_auto" });
    renderPanel(preview({ alice: "af_nicole" }, { create: [] }), { saved });
    expect(within(rowOf(/Alice/)).getByText("will change")).toBeInTheDocument();
    expect(screen.getByText(/1 will change/)).toBeInTheDocument();
  });

  it("changing a row re-requests the preview with that pick as an override", async () => {
    const user = userEvent.setup();
    const { onPreview } = renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }));

    await user.click(within(rowOf(/Carol/)).getByRole("button", { name: /voice for Carol/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Adam · kokoro" }));

    expect(onPreview).toHaveBeenCalledTimes(1);
    expect(lastInput(onPreview)).toEqual({ accent: "a", overrides: { carol: "am_adam" } });
  });

  it("rows follow the roster order, not the reply's key order", () => {
    // the server appends overridden characters: alice (just changed) comes back last
    renderPanel(preview({ bob: "bob_auto", carol: "carol_auto", alice: "af_nicole" }));
    const rows = within(screen.getByRole("table", { name: "proposed voices" })).getAllByRole("row").slice(1);
    // each row has its voice select (and a ▶); the select names the character
    expect(rows.map((r) => within(r).getByRole("button", { name: /voice for/ }).getAttribute("aria-label"))).toEqual([
      expect.stringMatching(/voice for Alice/),
      expect.stringMatching(/voice for Bob/),
      expect.stringMatching(/voice for Carol/),
    ]);
  });

  it("narrator and accent feed the request", async () => {
    const user = userEvent.setup();
    const { onPreview } = renderPanel(preview({ alice: "alice_auto" }));

    await user.click(screen.getByRole("button", { name: /narrator for the suggested cast/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Nicole · kokoro" }));
    expect(lastInput(onPreview)).toEqual({ narrator_voice_id: "af_nicole", accent: "a", overrides: {} });

    await user.click(screen.getByRole("button", { name: /accent for new voices/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "British English" }));
    expect(lastInput(onPreview)).toEqual({ narrator_voice_id: "af_nicole", accent: "b", overrides: {} });
  });

  it("apply sends the same input the table was computed from, recast off unless ticked", async () => {
    const user = userEvent.setup();
    const { onApply } = renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }));

    await user.click(within(rowOf(/Carol/)).getByRole("button", { name: /voice for Carol/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Nicole · kokoro" }));
    await user.click(screen.getByRole("button", { name: "apply cast" }));

    expect(onApply).toHaveBeenCalledWith({
      input: { accent: "a", overrides: { carol: "af_nicole" } },
      recast: false,
      useLlm: false,
    });
  });

  it("apply waits while a re-request is in flight, and says so in a live region", () => {
    renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }), { refreshing: true });
    expect(screen.getByRole("button", { name: "apply cast" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("updating the suggestion…");
  });

  it("a change waits for the server's answer before apply can carry it", async () => {
    const user = userEvent.setup();
    let answer: Done = () => {};
    const onPreview: PreviewFn = vi.fn((_input: SuggestInput, done: Done) => {
      answer = done;
    });
    const { onApply } = renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }), { onPreview });

    await user.click(within(rowOf(/Carol/)).getByRole("button", { name: /voice for Carol/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Adam · kokoro" }));
    expect(screen.getByRole("button", { name: "apply cast" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("updating the suggestion…");

    act(() => answer(null));
    await user.click(screen.getByRole("button", { name: "apply cast" }));
    expect(onApply).toHaveBeenCalledWith({
      input: { accent: "a", overrides: { carol: "am_adam" } },
      recast: false,
      useLlm: false,
    });
  });

  it("a refused change is rolled back and announced; apply sends the last choices that worked", async () => {
    const user = userEvent.setup();
    const onPreview: PreviewFn = vi.fn((input: SuggestInput, done: Done) =>
      done(input.overrides?.carol === "am_adam" ? new Error("voice am_adam is not in the library") : null),
    );
    const { onApply } = renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }), { onPreview });

    // a good change first (accent), then one the server refuses (Carol -> Adam)
    await user.click(screen.getByRole("button", { name: /accent for new voices/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "British English" }));
    await user.click(within(rowOf(/Carol/)).getByRole("button", { name: /voice for Carol/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Adam · kokoro" }));

    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(/couldn't use that change, so it was undone/);
    expect(status).toHaveTextContent(/voice am_adam is not in the library/);

    // the refused pick is not kept: the next change re-sends only what worked
    await user.click(within(rowOf(/Alice/)).getByRole("button", { name: /voice for Alice/ }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: "Nicole · kokoro" }));
    expect(lastInput(onPreview)).toEqual({ accent: "b", overrides: { alice: "af_nicole" } });
    expect(status).toBeEmptyDOMElement();

    const apply = screen.getByRole("button", { name: "apply cast" });
    expect(apply).toBeEnabled();
    await user.click(apply);
    expect(onApply).toHaveBeenCalledWith({
      input: { accent: "b", overrides: { alice: "af_nicole" } },
      recast: false,
      useLlm: false,
    });
  });

  it("only claims every voice is distinct when no existing voice is kept as it is", async () => {
    const user = userEvent.setup();
    const saved = assignmentOf({ alice: "alice_auto" });
    renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }, { recast: ["alice_auto"] }), { saved });
    expect(screen.queryByText(/every character a distinct voice/)).not.toBeInTheDocument();
    expect(screen.getByText(/new characters get their own voices/)).toBeInTheDocument();
    expect(screen.getByText(/tick re-cast to make every voice distinct/)).toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: /re-cast the 1 existing/ }));
    expect(screen.getByText(/every character a distinct voice/)).toBeInTheDocument();
    expect(screen.queryByText(/tick re-cast to make every voice distinct/)).not.toBeInTheDocument();
  });

  it("a hidden (spoiler-safe) character shows neither its name nor its auto-named voice — in any list", async () => {
    const user = userEvent.setup();
    renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }), { masked: ["carol"] });
    expect(screen.queryByText("Carol")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /voice for Carol/ })).not.toBeInTheDocument();
    expect(screen.getByText("a character you haven't met yet")).toBeInTheDocument();

    // carol_auto ("Carol · kokoro") is in the voice library, but no dropdown may offer it
    await user.click(within(rowOf(/Alice/)).getByRole("button", { name: /voice for Alice/ }));
    const rowList = await screen.findByRole("listbox");
    expect(within(rowList).getByRole("option", { name: "Bob · kokoro" })).toBeInTheDocument();
    expect(within(rowList).queryByText(/Carol/)).not.toBeInTheDocument();
    await user.keyboard("{Escape}");

    await user.click(screen.getByRole("button", { name: /narrator for the suggested cast/ }));
    const narratorList = await screen.findByRole("listbox");
    expect(within(narratorList).getByRole("option", { name: "Adam · kokoro" })).toBeInTheDocument();
    expect(within(narratorList).queryByText(/Carol/)).not.toBeInTheDocument();
  });
});

describe("SuggestCastPanel ▶ per row", () => {
  it("an existing proposed voice gets a ▶; one made only on apply is greyed and says why, reachably", async () => {
    const user = userEvent.setup();
    renderPanel(preview({ alice: "alice_auto", bob: "bob_new" }, { create: ["bob_new"] }));
    const aliceKey = within(rowOf(/Alice/)).getByRole("button", { name: "hear Alice's proposed voice" });
    expect(aliceKey).toBeEnabled();
    expect(aliceKey).not.toHaveAttribute("aria-disabled");
    const bobKey = within(rowOf(/Bob/)).getByRole("button", { name: /hear Bob's proposed voice/ });
    // aria-disabled, not disabled: still focusable, so the keyboard reaches the reason too
    expect(bobKey).toHaveAttribute("aria-disabled", "true");
    expect(bobKey).toHaveAccessibleName(/made when you apply the cast/);
    for (let i = 0; i < 40 && document.activeElement !== bobKey; i++) await user.tab();
    expect(bobKey).toHaveFocus();
    expect(await screen.findByRole("tooltip")).toHaveTextContent(/made when you apply the cast — hear it then/);
  });

  it("a hidden (spoiler-safe) character's row has no ▶", () => {
    renderPanel(preview({ alice: "alice_auto", carol: "carol_auto" }), { masked: ["carol"] });
    expect(screen.getAllByRole("button", { name: /proposed voice/ })).toHaveLength(1);
  });
});
