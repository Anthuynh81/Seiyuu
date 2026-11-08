import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { CastPlan, DesignOut, PlanRow } from "../../api/types";
import { NARRATOR_ROW } from "../../api/types";
import { makeJob } from "../../test/fixtures";
import { mockApi, renderWithProviders } from "../../test/utils";
import { designMinutes, rowStatus, rowsToDesign, shareTargets } from "./castPlan";
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
    ...over,
  };
}

const describeJob = (over = {}) =>
  makeJob({ job_id: "j1", book_id: B, kind: "describe_cast", progress_text: "describing Kim Soleum (5 of 21)", ...over });

describe("CastPlanPanel", () => {
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
