import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { DesignOut, DesignTakeOut } from "../../api/types";
import { makeJob } from "../../test/fixtures";
import { mockApi, renderWithProviders } from "../../test/utils";
import { DesignDialog } from "./DesignDialog";
import { takeVerdict } from "./helpers";

const ID = "0123456789ab";

function take(k: number, over: Partial<DesignTakeOut> = {}): DesignTakeOut {
  return {
    take: k,
    seconds: 8,
    wpm: 175,
    in_pace: true,
    ok: true,
    score: 0.98,
    reason: null,
    audio_url: `/api/designs/${ID}/takes/${k}.wav`,
    kept_voice_id: null,
    ...over,
  };
}

function design(over: Partial<DesignOut> = {}): DesignOut {
  return {
    design_id: ID,
    name: "Old janitor",
    description: "A gruff man in his sixties",
    text: "I didn't expect to find anyone here.",
    takes_requested: 2,
    takes: [],
    recommended: null,
    ...over,
  };
}

const designJob = (over = {}) =>
  makeJob({ job_id: "job-d", book_id: `design:${ID}`, kind: "design", progress_text: "designing take 1 of 2", ...over });

async function fillAndStart(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("voice name"), "Old janitor");
  await user.type(screen.getByLabelText("describe the voice"), "A gruff man in his sixties");
  await user.click(screen.getByRole("button", { name: "design it" }));
}

describe("DesignDialog", () => {
  it("starts a design, shows takes as they land, and keeps the chosen one", async () => {
    const user = userEvent.setup();
    let jobState: "running" | "succeeded" = "running";
    const server = mockApi()
      .post("/api/designs", { design: design(), job: designJob() })
      .on("GET", "/api/jobs/job-d", () =>
        designJob({ state: jobState, is_terminal: jobState === "succeeded", progress_text: "checking take 2 of 2" }),
      )
      .get(`/api/designs/${ID}`, design({ takes: [take(1)] }));
    renderWithProviders(<DesignDialog onClose={() => {}} />);

    expect(screen.getByRole("button", { name: "design it" })).toBeDisabled();
    await fillAndStart(user);
    expect(server.jsonBodyOf("POST", "/api/designs")).toEqual({
      name: "Old janitor",
      description: "A gruff man in his sixties",
      takes: 3,
    });

    // live: progress + the first take, already playable
    expect(await screen.findByText(/checking take 2 of 2/)).toBeInTheDocument();
    expect(await screen.findByRole("listitem", { name: "take 1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "close (keeps designing)" })).toBeInTheDocument();

    // the job finishes: both takes, the recommended one marked
    jobState = "succeeded";
    server.get(
      `/api/designs/${ID}`,
      design({ takes: [take(1, { wpm: 240, in_pace: false }), take(2)], recommended: 2 }),
    );
    const second = await screen.findByRole("listitem", { name: "take 2" }, { timeout: 5000 });
    expect(within(second).getByText("recommended")).toBeInTheDocument();
    expect(within(second).getByText("natural pace, every word clear")).toBeInTheDocument();
    expect(second.querySelector("audio")?.getAttribute("src")).toBe(`/api/designs/${ID}/takes/2.wav`);
    const first = screen.getByRole("listitem", { name: "take 1" });
    expect(within(first).queryByText("recommended")).not.toBeInTheDocument();
    expect(within(first).getByText(/a bit fast/)).toBeInTheDocument();

    server
      .post(`/api/designs/${ID}/keep`, { voice_id: "old_janitor_x", name: "Old janitor", kind: "cloned", engine: "qwen3tts" })
      .get("/api/voices", { voices: [], unreadable: [] });
    await user.click(within(second).getByRole("button", { name: "keep this one" }));
    expect(await screen.findByText(/to your voices/)).toHaveTextContent("Saved Old janitor to your voices.");
    expect(server.jsonBodyOf("POST", "/keep")).toEqual({ take: 2, name: "Old janitor" });
    expect(screen.getByRole("button", { name: "done" })).toBeInTheDocument();
  }, 15000);

  it("a failed design explains itself and keeps the takes it finished", async () => {
    const user = userEvent.setup();
    mockApi()
      .post("/api/designs", { design: design(), job: designJob() })
      .get(
        "/api/jobs/job-d",
        designJob({ state: "failed", is_terminal: true, error: "qwen3tts worker failed to load: CUDA out of memory" }),
      )
      .get(`/api/designs/${ID}`, design({ takes: [take(1)] }));
    renderWithProviders(<DesignDialog onClose={() => {}} />);
    await fillAndStart(user);

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("design stopped")).toBeInTheDocument();
    expect(within(alert).getByText("The takes it finished are still here.")).toBeInTheDocument();
    expect(screen.getByRole("listitem", { name: "take 1" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "done" })).toBeInTheDocument());
  });

  it("an already-kept take shows as kept instead of offering keep", async () => {
    const user = userEvent.setup();
    mockApi()
      .post("/api/designs", { design: design(), job: designJob() })
      .get("/api/jobs/job-d", designJob({ state: "succeeded", is_terminal: true }))
      .get(`/api/designs/${ID}`, design({ takes: [take(1, { kept_voice_id: "v1" })], recommended: 1 }));
    renderWithProviders(<DesignDialog onClose={() => {}} />);
    await fillAndStart(user);
    const row = await screen.findByRole("listitem", { name: "take 1" });
    expect(within(row).getByText("kept")).toBeInTheDocument();
    expect(within(row).queryByRole("button", { name: "keep this one" })).not.toBeInTheDocument();
  });
});

describe("takeVerdict", () => {
  it("says pace and speech check in plain words", () => {
    expect(takeVerdict({ wpm: 175, in_pace: true, ok: true, reason: null })).toBe("natural pace, every word clear");
    expect(takeVerdict({ wpm: 120, in_pace: false, ok: false, reason: "dropped words" })).toBe(
      "a bit slow, the speech check flagged it (dropped words)",
    );
  });
});
