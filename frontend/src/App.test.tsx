import { act, screen, waitFor, within } from "@testing-library/react";
import { useLocation } from "react-router-dom";
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
    ["/review", "Speakers & cast"],
    ["/lexicon", "Proof & names"],
    ["/voices", "Voices"],
    ["/series", "Series"],
    ["/render", "Produce"],
  ] as [string, string][])("route %s renders its screen heading %s", async (route, heading) => {
    mockEmptyBackend();
    renderWithProviders(<App />, { route });
    expect(await screen.findByRole("heading", { level: 1, name: heading })).toBeInTheDocument();
  });

  it.each(["/", "/review", "/listen", "/render", "/lexicon"])(
    "blocked storage doesn't crash the app on open (%s)",
    async (route) => {
      const deny = () => {
        throw new DOMException("The operation is insecure.", "SecurityError");
      };
      mockEmptyBackend().get("/api/books", { books: [makeBook({ book_id: "b1", attributed: true })] });
      vi.stubGlobal("localStorage", { getItem: deny, setItem: deny, removeItem: deny, clear: deny, key: deny });
      renderWithProviders(<App />, { route });
      expect(await screen.findByRole("heading", { level: 1 })).toBeInTheDocument();
      expect(screen.getByRole("navigation")).toBeInTheDocument();
    },
  );

  it.each([
    ["/render", "/api/books/ghost"],
    ["/review", "/api/books/ghost"],
    ["/lexicon", "/api/books/ghost/lexicon"],
  ] as [string, string][])(
    "%s with no ?book= opens the remembered book, not the first on the shelf",
    async (route, wanted) => {
      const server = mockEmptyBackend().get("/api/books", {
        books: [
          makeBook({ book_id: "heart", attributed: true }),
          makeBook({ book_id: "ghost", attributed: true }),
        ],
      });
      localStorage.setItem("seiyuu.currentBook", "ghost");
      renderWithProviders(<App />, { route });
      await waitFor(() => expect(server.calls.some((c) => c.url.startsWith(wanted))).toBe(true));
      expect(server.calls.some((c) => c.url.startsWith(wanted.replace("ghost", "heart")))).toBe(false);
    },
  );

  it("/listen with no ?book= opens the remembered book", async () => {
    const server = mockEmptyBackend().get("/api/books", {
      books: [makeBook({ book_id: "heart", rendered: true }), makeBook({ book_id: "ghost", rendered: true })],
    });
    localStorage.setItem("seiyuu.currentBook", "ghost");
    renderWithProviders(<App />, { route: "/listen" });
    await waitFor(() => expect(server.calls.some((c) => c.url.startsWith("/api/books/ghost"))).toBe(true));
    expect(server.calls.some((c) => c.url.startsWith("/api/books/heart"))).toBe(false);
  });

  it("a screen's own default book is pinned into the URL, so the nav names it and every step keeps it", async () => {
    mockEmptyBackend().get("/api/books", {
      books: [
        makeBook({ book_id: "heart", title: "The Expired Heart", attributed: true }),
        makeBook({ book_id: "ghost", title: "Ghost Story", attributed: true }),
      ],
    });
    renderWithProviders(<App />, { route: "/render" }); // fresh browser: nothing remembered
    const nav = screen.getByRole("navigation");
    await waitFor(() =>
      expect(within(nav).getByRole("link", { name: "4 Listen" })).toHaveAttribute("href", "/listen?book=heart"),
    );
    expect(within(nav).getByRole("link", { name: "2 Proof & names" })).toHaveAttribute("href", "/lexicon?book=heart");
    expect(await within(nav).findByText("The Expired Heart")).toBeInTheDocument();
    await waitFor(() => expect(localStorage.getItem("seiyuu.currentBook")).toBe("heart"));
  });

  it("a book picked in another tab doesn't move a screen that is already open", async () => {
    const server = mockEmptyBackend().get("/api/books", {
      books: [makeBook({ book_id: "heart", attributed: true }), makeBook({ book_id: "ghost", attributed: true })],
    });
    localStorage.setItem("seiyuu.currentBook", "ghost");
    function Where() {
      const location = useLocation();
      return <output data-testid="where">{location.pathname + location.search}</output>;
    }
    renderWithProviders(
      <>
        <App />
        <Where />
      </>,
      { route: "/render" },
    );
    // the screen has settled on the remembered book and pinned it into its URL
    await waitFor(() => expect(screen.getByTestId("where").textContent).toBe("/render?book=ghost"));
    const nav = screen.getByRole("navigation");

    act(() => {
      localStorage.setItem("seiyuu.currentBook", "heart");
      window.dispatchEvent(new StorageEvent("storage", { key: "seiyuu.currentBook", newValue: "heart" }));
    });
    expect(screen.getByTestId("where").textContent).toBe("/render?book=ghost");
    expect(within(nav).getByRole("link", { name: "4 Listen" })).toHaveAttribute("href", "/listen?book=ghost");
    expect(server.calls.some((c) => c.url.startsWith("/api/books/heart"))).toBe(false);
  });

  it("a remembered book that was deleted falls back to the screen's usual default", async () => {
    const server = mockEmptyBackend().get("/api/books", { books: [makeBook({ book_id: "heart" })] });
    localStorage.setItem("seiyuu.currentBook", "gone");
    renderWithProviders(<App />, { route: "/render" });
    await waitFor(() => expect(server.calls.some((c) => c.url.startsWith("/api/books/heart"))).toBe(true));
    expect(server.calls.some((c) => c.url.includes("/gone"))).toBe(false);
  });

  it("invalidates stage-artifact query families when the live job count drops", async () => {
    const server = mockEmptyBackend();
    let jobs: JobOut[] = [makeJob()];
    server.on("GET", "/api/jobs", () => ({ jobs }));

    const queryClient = makeQueryClient();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    renderWithProviders(<App />, { queryClient });

    // the first poll lands: the transport bar shows the running job (book title + plain step;
    // this shelf is empty, so the book reads as "your book", never its id)
    expect(await screen.findByText("your book · Recording")).toBeInTheDocument();
    // "characters": a finished speaker re-run changes the roster and its provenance
    const families = ["segments", "render-summary", "validation", "estimate", "assignment", "voices", "characters"];
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
    expect(await screen.findByText("Night Train · Recording")).toBeInTheDocument();
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
    expect(await screen.findByText("Night Train · Finding the speakers")).toBeInTheDocument();
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
