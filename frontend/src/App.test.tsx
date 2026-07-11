import { act, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { JobOut } from "./api/types";
import App from "./App";
import { makeJob } from "./test/fixtures";
import { type MockApi, makeQueryClient, mockApi, renderWithProviders } from "./test/utils";

/** Every mount-time GET the seven screens make, answered with settled empty payloads so each
    screen reaches its heading instead of hanging in a load state. */
function mockEmptyBackend(): MockApi {
  return mockApi()
    .get("/api/books", { books: [] })
    .get("/api/voices", { voices: [], unreadable: [] })
    .get("/api/cloud-slots", { max_slots: 10, count: 0, slots: [] })
    .get("/api/series", { series: [] })
    .get("/api/engines/kokoro/voices", { engine_id: "kokoro", voices: [] })
    .get("/api/system", {
      attribution: {
        provider: "local",
        model: "qwen2.5:7b",
        anthropic_model: "claude-test",
        prompt_version: "v6",
      },
      keys: { anthropic_configured: false, elevenlabs_configured: false },
      apply_emotion: false,
    });
}

describe("App", () => {
  it.each([
    ["/", "Library"],
    ["/listen", "Listen"],
    ["/review", "Character Review"],
    ["/lexicon", "Pronunciation"],
    ["/voices", "Voice Studio"],
    ["/series", "Series"],
    ["/render", "Render & Jobs"],
  ] as [string, string][])("route %s renders its screen heading %s", async (route, heading) => {
    mockEmptyBackend();
    renderWithProviders(<App />, { route });
    expect(await screen.findByRole("heading", { level: 1, name: heading })).toBeInTheDocument();
  });

  it("invalidates stage-artifact query families when the live job count drops", async () => {
    const server = mockEmptyBackend();
    let jobs: JobOut[] = [makeJob()];
    server.on("GET", "/api/jobs", () => ({ jobs }));

    const queryClient = makeQueryClient();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    renderWithProviders(<App />, { queryClient });

    // the first poll lands: the transport bar shows the running job
    expect(await screen.findByText("render · demo")).toBeInTheDocument();
    const families = ["segments", "render-summary", "validation", "estimate", "assignment", "voices"];
    for (const key of families) {
      expect(invalidate).not.toHaveBeenCalledWith({ queryKey: [key] });
    }

    // the job finishes: drive the next poll deterministically instead of waiting out the
    // refetch interval — the test must not care what cadence useLiveJobs polls at
    jobs = [];
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "live"] });
    });
    await waitFor(() => {
      for (const key of families) {
        expect(invalidate).toHaveBeenCalledWith({ queryKey: [key] });
      }
    });
  });
});
