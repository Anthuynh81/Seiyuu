import { act, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { JobOut } from "./api/types";
import App from "./App";
import { makeBook, makeEngine, makeJob, makeSystemProbe } from "./test/fixtures";
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
    .get("/api/engines", { engines: [makeEngine()] })
    .get("/api/system", {
      attribution: {
        provider: "local",
        model: "qwen2.5:7b",
        anthropic_model: "claude-test",
        prompt_version: "v6",
      },
      keys: { anthropic_configured: false, elevenlabs_configured: false },
      apply_emotion: false,
    })
    // registered AFTER the pathname-only /api/system route: the latest match wins
    .get("/api/system?probe=true", makeSystemProbe());
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
      // a finished warm-up flips weights_cached: the NEW health keys refresh too
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["engines"] });
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["system", "probe"] });
    });
  });

  it("a finished job fetches the recent-jobs window and raises one plain notice; book keys keep their shape", async () => {
    const server = mockEmptyBackend().get("/api/books", { books: [makeBook({ book_id: "b1", title: "Night Train" })] });
    const running = makeJob({ job_id: "j9", book_id: "b1", kind: "render", state: "running" });
    let live: JobOut[] = [running];
    let recent: JobOut[] = [running];
    server.on("GET", "/api/jobs", (url) => (url.searchParams.has("state") ? { jobs: live } : { jobs: recent }));
    server.get("/api/jobs/j9", { ...running, state: "succeeded", is_terminal: true, finished_at: "2026-10-22T10:00:00Z" });

    const queryClient = makeQueryClient();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    renderWithProviders(<App />, { queryClient });
    expect(await screen.findByText("render · b1")).toBeInTheDocument();
    await waitFor(() => expect(queryClient.getQueryData(["jobs", "recent"])).toBeDefined());
    const recentCalls = () => server.calls.filter((c) => c.url === "/api/jobs?limit=10").length;
    const before = recentCalls();

    live = [];
    recent = [{ ...running, state: "succeeded", is_terminal: true, finished_at: "2026-10-22T10:00:00Z" }];
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "live"] });
    });
    const notices = screen.getByRole("status", { name: "Job notices" });
    expect(await within(notices).findByText("Recorded Night Train")).toBeInTheDocument();
    expect(recentCalls()).toBeGreaterThan(before);
    expect(server.calls.some((c) => c.url === "/api/jobs/j9")).toBe(true); // the departed job, by id
    expect(within(notices).getAllByText("Recorded Night Train")).toHaveLength(1); // both sources, one notice
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["jobs", "shelf"] });
    // the stable book keys are invalidated exactly as before, never re-keyed
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["books"] });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["book"] });
  });

  it("a finish hidden by a new job queuing in the same tick still raises its notice", async () => {
    const server = mockEmptyBackend().get("/api/books", { books: [makeBook({ book_id: "b1", title: "Night Train" })] });
    const j1 = makeJob({ job_id: "j1", book_id: "b1", kind: "attribute", state: "running" });
    const j2 = makeJob({ job_id: "j2", book_id: "b1", kind: "render", state: "queued" });
    let live: JobOut[] = [j1];
    // the window never shows j1 finished: only the by-id fetch can tell
    server.on("GET", "/api/jobs", (url) => ({ jobs: url.searchParams.has("state") ? live : [j2] }));
    server.get("/api/jobs/j1", { ...j1, state: "failed", is_terminal: true, error: "interrupted: server stopped mid-job" });

    const queryClient = makeQueryClient();
    renderWithProviders(<App />, { queryClient });
    expect(await screen.findByText("attribute · b1")).toBeInTheDocument();
    await waitFor(() => expect(queryClient.getQueryData(["jobs", "recent"])).toBeDefined());
    const recentCalls = () => server.calls.filter((c) => c.url === "/api/jobs?limit=10").length;
    const before = recentCalls();

    live = [j2]; // same count, different id
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "live"] });
    });
    const notices = screen.getByRole("status", { name: "Job notices" });
    expect(await within(notices).findByText("Couldn't find the speakers in Night Train")).toBeInTheDocument();
    expect(within(notices).getByText("Seiyuu was closed while this was running.")).toBeInTheDocument();
    expect(recentCalls()).toBeGreaterThan(before);
  });
});
