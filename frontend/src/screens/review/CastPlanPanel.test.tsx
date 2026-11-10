import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { CastPlan, Delivery, DesignOut, PlanRow, VoiceAssignment } from "../../api/types";
import { NARRATOR_ROW } from "../../api/types";
import { makeJob } from "../../test/fixtures";
import { errorResponse, mockApi, renderWithProviders } from "../../test/utils";
import {
  designMinutes,
  narrationNote,
  narratorSharesVoice,
  oneLine,
  rowStatus,
  rowsToDesign,
  shareTargets,
} from "./castPlan";
import { CastPlanPanel } from "./CastPlanPanel";

const B = "ghost";
const PLAN_URL = `/api/books/${B}/cast-plan`;

function row(over: Partial<PlanRow>): PlanRow {
  return {
    character_id: "kim",
    name: "Kim Soleum",
    line_count: 91,
    gender: "male",
    age: "young adult",
    description: "A clear, composed young male voice.",
    sample_lines: ["Could you tell me more?", "Based on what you've told me, I have a plan."],
    sample_index: 0,
    same_as: null,
    keep_current: false,
    takes: 2,
    edited: false,
    drafted_by: "llm",
    design_id: null,
    voice_id: null,
    ...over,
  };
}

function plan(over: Partial<CastPlan> = {}): CastPlan {
  return {
    book_id: B,
    provider_id: "local",
    model_id: "gemma4:12b-it-qat",
    prompt_version: "v1",
    first_person: true,
    protagonist_suggestion: "kim",
    narrator: row({
      character_id: NARRATOR_ROW,
      name: "Narrator",
      line_count: 0,
      gender: null,
      age: null,
      description: "A warm, clear audiobook narrator's voice.",
      sample_lines: ["As soon as I did, the shape of the mask began to change."],
    }),
    rows: [
      row({ character_id: "new", name: "New Employee", line_count: 456, gender: null, same_as: "kim" }),
      row({}),
      row({ character_id: "ghost", name: "Ghost", line_count: 1, takes: 1, gender: null, age: "adult" }),
    ],
    notes: [],
    delivery: delivery(),
    ...over,
  };
}

function delivery(over: Partial<Delivery> = {}): Delivery {
  return {
    enabled: true,
    inner_voice: "The narrator's inner voice: calm and intimate.",
    narration: "",
    dialogue: "Natural, expressive spoken delivery.",
    ...over,
  };
}

function cast(narrator: string): VoiceAssignment {
  return {
    schema_version: 1,
    book_id: B,
    stage: "draft",
    narrator_voice_id: narrator,
    assignments: { kim: "kim_v", ghost: "ghost_v" },
    thought_voice_id: null,
    created_at: "2026-11-09",
  };
}

const describeJob = (over = {}) =>
  makeJob({ job_id: "j1", book_id: B, kind: "describe_cast", progress_text: "describing Kim Soleum (5 of 21)", ...over });

describe("CastPlanPanel", () => {
  it("edits how lines are spoken: the switch and each text, saved on blur", async () => {
    const user = userEvent.setup();
    const server = mockApi().get(PLAN_URL, plan());
    server.on("PUT", PLAN_URL, () => plan());
    renderWithProviders(<CastPlanPanel bookId={B} assignment={cast("kim_v")} onClose={() => {}} />);

    const section = await screen.findByRole("group", { name: "how lines are spoken" });
    expect(within(section).getByText(/Breeze voices only/)).toBeInTheDocument();
    // the narrator shares Kim's voice in the saved cast: narration is the inner voice
    expect(within(section).getByText(/narration uses the inner-voice text/)).toBeInTheDocument();

    const dialogue = within(section).getByLabelText(/^Dialogue/);
    expect(dialogue).toHaveValue("Natural, expressive spoken delivery.");
    fireEvent.blur(dialogue); // unchanged: nothing is saved
    expect(server.calls.filter((c) => c.method === "PUT")).toHaveLength(0);
    await user.clear(dialogue);
    await user.type(dialogue, "  Quick   and clipped. ");
    fireEvent.blur(dialogue);
    await waitFor(() =>
      expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ delivery: { dialogue: "Quick and clipped." } }),
    );

    // an emptied text is saved as empty: the plain voice for that kind of line
    const inner = within(section).getByLabelText(/^Inner voice/);
    await user.clear(inner);
    fireEvent.blur(inner);
    await waitFor(() => expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ delivery: { inner_voice: "" } }));

    await user.click(within(section).getByRole("checkbox", { name: "direct the delivery" }));
    await waitFor(() => expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ delivery: { enabled: false } }));
  });

  it("with direction off, says what off means and keeps the texts editable", async () => {
    mockApi().get(PLAN_URL, plan({ delivery: delivery({ enabled: false }) }));
    renderWithProviders(<CastPlanPanel bookId={B} onClose={() => {}} />);
    const section = await screen.findByRole("group", { name: "how lines are spoken" });
    expect(within(section).getByRole("checkbox", { name: "direct the delivery" })).not.toBeChecked();
    expect(within(section).getByText(/every line is spoken in the voice as it is/)).toBeInTheDocument();
    expect(within(section).getAllByRole("textbox")).toHaveLength(3); // kept for when it is on again
  });

  it("keeps a refused text marked not saved until it is saved, whatever else saves", async () => {
    const user = userEvent.setup();
    const server = mockApi().get(PLAN_URL, plan());
    // a server that keeps what it is sent, and refuses everything while a render runs
    let stored = delivery();
    let rendering = true;
    server.on("PUT", PLAN_URL, (_url, init) => {
      if (rendering) return errorResponse(409, "render_active", "a render is running for this book");
      stored = { ...stored, ...JSON.parse(String(init?.body)).delivery };
      return plan({ delivery: stored });
    });
    renderWithProviders(<CastPlanPanel bookId={B} onClose={() => {}} />);
    const section = await screen.findByRole("group", { name: "how lines are spoken" });

    const dialogue = within(section).getByLabelText(/^Dialogue/);
    await user.clear(dialogue);
    await user.type(dialogue, "Quick.");
    await user.click(document.body); // leave the field: the save is refused
    const flag = await within(section).findByRole("group", { name: "Dialogue text not saved" });

    // the render ends and another text saves fine: the refused one stays flagged, with its text
    rendering = false;
    const narration = within(section).getByLabelText(/^Narration/);
    await user.type(narration, "Steady.");
    await user.click(document.body);
    await waitFor(() => expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ delivery: { narration: "Steady." } }));
    await waitFor(() => expect(stored.narration).toBe("Steady."));
    expect(flag).toBeInTheDocument();
    expect(dialogue).toHaveValue("Quick.");
    expect(stored.dialogue).not.toBe("Quick.");

    // saving it again sends that text; once the server has it, the flag goes
    await user.click(within(flag).getByRole("button", { name: "save the dialogue text again" }));
    await waitFor(() => expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ delivery: { dialogue: "Quick." } }));
    await waitFor(() =>
      expect(within(section).queryByRole("group", { name: "Dialogue text not saved" })).not.toBeInTheDocument(),
    );
  });

  it("does not replace what is typed while a save of the same text is still on its way", async () => {
    const user = userEvent.setup();
    const server = mockApi().get(PLAN_URL, plan());
    let release: (p: CastPlan) => void = () => {};
    server.on("PUT", PLAN_URL, () => new Promise<CastPlan>((resolve) => (release = resolve)));
    renderWithProviders(<CastPlanPanel bookId={B} onClose={() => {}} />);
    const section = await screen.findByRole("group", { name: "how lines are spoken" });

    const dialogue = within(section).getByLabelText(/^Dialogue/);
    await user.clear(dialogue);
    await user.type(dialogue, "Quick.");
    fireEvent.blur(dialogue); // the save is on its way
    await waitFor(() => expect(server.calls.filter((c) => c.method === "PUT")).toHaveLength(1));
    await user.type(dialogue, " And dry."); // the user keeps typing
    release(plan({ delivery: delivery({ dialogue: "Quick." }) })); // the reply lands
    await waitFor(() => expect(server.calls.filter((c) => c.method === "PUT")).toHaveLength(1));
    await new Promise((r) => setTimeout(r, 50));
    expect(dialogue).toHaveValue("Quick. And dry.");

    server.on("PUT", PLAN_URL, () => plan({ delivery: delivery({ dialogue: "Quick. And dry." }) }));
    fireEvent.blur(dialogue); // and the rest is saved on the next blur
    await waitFor(() =>
      expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ delivery: { dialogue: "Quick. And dry." } }),
    );
  });

  it("with no plan yet, offers to describe the cast and shows the job while it runs", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .error("GET", PLAN_URL, 404, "not_found", "no cast plan yet")
      .post(`${PLAN_URL}/describe`, describeJob());
    renderWithProviders(<CastPlanPanel bookId={B} onClose={() => {}} />);

    await user.click(await screen.findByRole("button", { name: "Describe the cast" }));
    expect(server.lastCall("POST", "/cast-plan/describe")).toBeDefined();

    server.get("/api/jobs", { jobs: [describeJob()] });
    expect(await screen.findByRole("status", {}, { timeout: 4000 })).toHaveTextContent(
      "Describing the cast: describing Kim Soleum (5 of 21)",
    );
  });

  it("shows the plan narrator first and saves each kind of edit", async () => {
    const user = userEvent.setup();
    const server = mockApi().get(PLAN_URL, plan());
    server.on("PUT", PLAN_URL, () => plan());
    renderWithProviders(<CastPlanPanel bookId={B} onClose={() => {}} />);

    const list = await screen.findByRole("list", { name: "cast plan" });
    const names = within(list)
      .getAllByRole("listitem")
      .map((li) => li.getAttribute("aria-label"));
    expect(names).toEqual(["Narrator", "New Employee", "Kim Soleum", "Ghost"]);
    const employee = screen.getByRole("listitem", { name: "New Employee" });
    expect(within(employee).getByText("shares Kim Soleum's voice")).toBeInTheDocument();
    expect(within(employee).queryByRole("textbox")).not.toBeInTheDocument(); // shares: nothing to describe

    // first person: offer the protagonist's voice for the narration
    await user.click(screen.getByRole("button", { name: "Narrate in Kim Soleum's voice" }));
    expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({
      updates: [{ character_id: NARRATOR_ROW, same_as: "kim" }],
    });

    // a description saves on blur, only when it changed
    const kim = screen.getByRole("listitem", { name: "Kim Soleum" });
    const box = within(kim).getByLabelText("how Kim Soleum sounds");
    fireEvent.blur(box);
    const puts = () => server.calls.filter((c) => c.method === "PUT").length;
    expect(puts()).toBe(1);
    await user.clear(box);
    await user.type(box, "A tired, dry young man.");
    fireEvent.blur(box);
    await waitFor(() =>
      expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({
        updates: [{ character_id: "kim", description: "A tired, dry young man." }],
      }),
    );

    await user.selectOptions(within(kim).getByLabelText("Kim Soleum's sample line"), "1");
    expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ updates: [{ character_id: "kim", sample_index: 1 }] });

    await user.selectOptions(within(employee).getByLabelText("New Employee shares a voice with"), "__own__");
    expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ updates: [{ character_id: "new", clear_same_as: true }] });

    await user.click(within(screen.getByRole("listitem", { name: "Ghost" })).getByRole("checkbox"));
    expect(server.jsonBodyOf("PUT", PLAN_URL)).toEqual({ updates: [{ character_id: "ghost", keep_current: true }] });
  });

  it("designs the voices that still need one, with a time estimate", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get(PLAN_URL, plan())
      .post(`${PLAN_URL}/design`, makeJob({ job_id: "j2", book_id: B, kind: "design_cast" }));
    renderWithProviders(<CastPlanPanel bookId={B} onClose={() => {}} />);
    // narrator 2 + Kim 2 + Ghost 1 takes; New Employee shares Kim's voice
    await user.click(await screen.findByRole("button", { name: "Design 3 voices (about 1 min)" }));
    expect(server.lastCall("POST", "/cast-plan/design")).toBeDefined();
  });

  it("plays a designed character's takes and switches to another", async () => {
    const user = userEvent.setup();
    const designed = plan({
      rows: [row({ design_id: "0123456789ab", voice_id: "kim_v1" })],
      narrator: plan().narrator,
    });
    const takes: DesignOut = {
      design_id: "0123456789ab",
      name: "Kim Soleum",
      description: "x",
      text: "Could you tell me more?",
      takes_requested: 2,
      recommended: 1,
      takes: [1, 2].map((k) => ({
        take: k,
        seconds: 7,
        wpm: 175,
        in_pace: true,
        ok: true,
        score: 0.98,
        reason: null,
        audio_url: `/api/designs/0123456789ab/takes/${k}.wav`,
        kept_voice_id: k === 1 ? "kim_v1" : null,
      })),
    };
    const server = mockApi()
      .get(PLAN_URL, designed)
      .get("/api/designs/0123456789ab", takes)
      .post(`${PLAN_URL}/rows/kim/take`, {
        plan: designed,
        assignment: { book_id: B, narrator_voice_id: "n", assignments: {}, stage: "draft", thought_voice_id: null },
      });
    renderWithProviders(<CastPlanPanel bookId={B} onClose={() => {}} />);

    const kim = await screen.findByRole("listitem", { name: "Kim Soleum" });
    expect(within(kim).getByText("designed")).toBeInTheDocument();
    await user.click(within(kim).getByRole("button", { name: "hear takes" }));
    const first = await screen.findByRole("listitem", { name: "take 1" });
    expect(within(first).getByText("in use")).toBeInTheDocument();
    const second = screen.getByRole("listitem", { name: "take 2" });
    expect(second.querySelector("audio")?.getAttribute("src")).toBe("/api/designs/0123456789ab/takes/2.wav");
    await user.click(within(second).getByRole("button", { name: "use this take" }));
    expect(server.jsonBodyOf("POST", "/rows/kim/take")).toEqual({ take: 2 });
  });
});

describe("castPlan helpers", () => {
  it("knows a first-person book by the narrator sharing a character's voice", () => {
    expect(narratorSharesVoice(cast("kim_v"))).toBe(true);
    expect(narratorSharesVoice(cast("narrator_v"))).toBe(false);
    expect(narrationNote(cast("kim_v"))).toMatch(/uses the inner-voice text/);
    expect(narrationNote(cast("narrator_v"))).toMatch(/uses the narration text/);
    expect(narrationNote(null)).toMatch(/when the narrator shares a character's voice/);
    expect(oneLine("  calm,\n  dry  ")).toBe("calm, dry");
  });

  it("count what will be designed and say where each row stands", () => {
    const p = plan();
    expect(rowsToDesign(p).map((r) => r.character_id)).toEqual([NARRATOR_ROW, "kim", "ghost"]);
    expect(designMinutes(rowsToDesign(p))).toBe(1);
    expect(designMinutes([])).toBe(0);
    expect(designMinutes(Array.from({ length: 20 }, () => row({ takes: 2 })))).toBe(4); // 40 takes
    expect(rowStatus(p, p.rows[0])).toBe("shares Kim Soleum's voice");
    expect(rowStatus(p, row({ keep_current: true }))).toBe("keeps its current voice");
    expect(rowStatus(p, row({ design_id: "x" }))).toMatch(/interrupted/);
    expect(shareTargets(p, p.rows[1]).map((r) => r.character_id)).toEqual(["ghost"]); // not itself, not a sharer
  });
});
