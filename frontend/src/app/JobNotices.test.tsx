import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { fetchDepartedJobs } from "../api/hooks";
import type { JobOut } from "../api/types";
import { makeBook, makeJob } from "../test/fixtures";
import { type MockApi, makeQueryClient, mockApi, renderWithProviders } from "../test/utils";
import { JobNotices, MAX_NOTICES, NOTICE_AUTO_DISMISS_MS } from "./JobNotices";

const done = (o: Partial<JobOut>) =>
  makeJob({ state: "succeeded", is_terminal: true, finished_at: "2026-10-22T10:05:00Z", progress_text: "", ...o });

function setup(initial: JobOut[]): { server: MockApi; setJobs: (j: JobOut[]) => void } {
  let jobs = initial;
  const server = mockApi()
    .get("/api/books", { books: [makeBook({ book_id: "b1", title: "The Hollow Crown" })] })
    .on("GET", "/api/jobs?limit=10", () => ({ jobs }));
  return { server, setJobs: (j) => (jobs = j) };
}

const region = () => screen.getByRole("status", { name: "Job notices" });

describe("JobNotices", () => {
  it("never toasts jobs that were already finished at load, then toasts each new finish once", async () => {
    const queryClient = makeQueryClient();
    const old = done({ job_id: "old", book_id: "b1", kind: "render" });
    const running = makeJob({ job_id: "j1", book_id: "b1", kind: "attribute", state: "running" });
    const { setJobs } = setup([running, old]);
    renderWithProviders(<JobNotices />, { queryClient });

    await waitFor(() => expect(queryClient.getQueryData(["jobs", "recent"])).toBeDefined());
    await waitFor(() => expect(queryClient.getQueryData(["books"])).toBeDefined());
    expect(region()).toBeEmptyDOMElement();

    // the attribute job finishes: one plain-words notice naming the book by title
    setJobs([done({ job_id: "j1", book_id: "b1", kind: "attribute" }), old]);
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
    });
    expect(await within(region()).findByText("Found the speakers in The Hollow Crown")).toBeInTheDocument();
    expect(within(region()).queryByText(/Recorded/)).not.toBeInTheDocument(); // `old` stays quiet

    // a refetch of the same window never toasts twice
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
    });
    expect(within(region()).getAllByText("Found the speakers in The Hollow Crown")).toHaveLength(1);
  });

  it("names the engine for a warm-up job", async () => {
    const queryClient = makeQueryClient();
    const { setJobs } = setup([]);
    renderWithProviders(<JobNotices />, { queryClient });
    await waitFor(() => expect(queryClient.getQueryData(["jobs", "recent"])).toBeDefined());

    setJobs([done({ job_id: "w1", book_id: "engine:chatterbox", kind: "warmup" })]);
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
    });
    expect(await within(region()).findByText("Chatterbox is downloaded and ready")).toBeInTheDocument();
    expect(within(region()).queryByText(/engine:/)).not.toBeInTheDocument();
  });

  // fake timers + fireEvent (userEvent deadlocks under vitest fake timers; see Voices.test)
  it("successes clear themselves after a few seconds; failures stay until dismissed", async () => {
    vi.useFakeTimers();
    try {
      const queryClient = makeQueryClient();
      const { setJobs } = setup([]);
      renderWithProviders(<JobNotices />, { queryClient });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });

      setJobs([
        done({ job_id: "r1", book_id: "b1", kind: "render" }),
        done({
          job_id: "a1",
          book_id: "b1",
          kind: "attribute",
          state: "failed",
          error: "OllamaUnreachableError: Ollama isn't reachable at http://localhost:11434 - start it with 'ollama serve'",
        }),
      ]);
      await act(async () => {
        await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
        await vi.advanceTimersByTimeAsync(50);
      });
      expect(screen.getByText("Recorded The Hollow Crown")).toBeInTheDocument();
      expect(screen.getByText("Couldn't find the speakers in The Hollow Crown")).toBeInTheDocument();
      // the failure explains itself in plain words, raw message under Details
      expect(screen.getByText(/couldn't reach Ollama/)).toBeInTheDocument();
      expect(screen.getByText("ollama serve")).toBeInTheDocument();
      expect(screen.getByText("Details")).toBeInTheDocument();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(NOTICE_AUTO_DISMISS_MS + 100);
      });
      expect(screen.queryByText("Recorded The Hollow Crown")).not.toBeInTheDocument();
      expect(screen.getByText("Couldn't find the speakers in The Hollow Crown")).toBeInTheDocument();

      // still there long after; only the user clears it
      await act(async () => {
        await vi.advanceTimersByTimeAsync(60_000);
      });
      fireEvent.click(screen.getByRole("button", { name: "Dismiss: Couldn't find the speakers in The Hollow Crown" }));
      expect(screen.queryByText("Couldn't find the speakers in The Hollow Crown")).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("a success notice can be dismissed by hand and links to the next step", async () => {
    const user = userEvent.setup();
    const queryClient = makeQueryClient();
    const { setJobs } = setup([]);
    renderWithProviders(<JobNotices />, { queryClient });
    await waitFor(() => expect(queryClient.getQueryData(["jobs", "recent"])).toBeDefined());

    setJobs([done({ job_id: "r1", book_id: "b1", kind: "render" })]);
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
    });
    expect(await screen.findByRole("button", { name: "Listen" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Dismiss: Recorded The Hollow Crown" }));
    expect(screen.queryByText("Recorded The Hollow Crown")).not.toBeInTheDocument();
  });

  it("toasts a finishing job that is older than the 10 newest (fetched by id when it left the live poll)", async () => {
    const queryClient = makeQueryClient();
    // ten newer queued jobs fill the window; the finishing render is below it
    const queued = Array.from({ length: 10 }, (_, i) =>
      makeJob({ job_id: `q${i}`, book_id: "b1", kind: "attribute", state: "queued" }),
    );
    const { server } = setup(queued);
    server.get("/api/jobs/old-render", {
      ...done({ job_id: "old-render", book_id: "b1", kind: "render", state: "failed" }),
      error: "AssembleError: ffmpeg not found on PATH; install it and retry",
    });
    renderWithProviders(<JobNotices />, { queryClient });
    await waitFor(() => expect(queryClient.getQueryData(["jobs", "recent"])).toBeDefined());

    await act(async () => {
      await fetchDepartedJobs(queryClient, ["old-render"]);
      await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] }); // still only q0..q9
    });
    expect(await within(region()).findByText("Recording The Hollow Crown stopped with a problem")).toBeInTheDocument();
    expect(within(region()).getByText(/ffmpeg isn't installed/)).toBeInTheDocument();
  });

  it("the cap trims old successes, never a failure", async () => {
    const queryClient = makeQueryClient();
    const { setJobs } = setup([]);
    renderWithProviders(<JobNotices />, { queryClient });
    await waitFor(() => expect(queryClient.getQueryData(["jobs", "recent"])).toBeDefined());

    const failure = done({ job_id: "f1", book_id: "b1", kind: "attribute", state: "failed", error: "boom", finished_at: "2026-10-22T10:00:00Z" });
    setJobs([failure]);
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
    });
    expect(await within(region()).findByText("Couldn't find the speakers in The Hollow Crown")).toBeInTheDocument();
    const successes = Array.from({ length: MAX_NOTICES }, (_, i) =>
      done({ job_id: `s${i}`, book_id: "engine:kokoro", kind: "warmup", finished_at: `2026-10-22T10:0${i + 1}:00Z` }),
    );
    setJobs([...successes, failure]);
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
    });
    await waitFor(() => expect(region().querySelectorAll(".toast")).toHaveLength(MAX_NOTICES));
    expect(within(region()).getByText("Couldn't find the speakers in The Hollow Crown")).toBeInTheDocument();
  });

  it("a success holds while hovered or focused, and leaving with the pointer keeps it while focus is inside", async () => {
    vi.useFakeTimers();
    try {
      const queryClient = makeQueryClient();
      const { setJobs } = setup([]);
      renderWithProviders(<JobNotices />, { queryClient });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });
      setJobs([done({ job_id: "r1", book_id: "b1", kind: "render" })]);
      await act(async () => {
        await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
        await vi.advanceTimersByTimeAsync(50);
      });
      const toast = screen.getByText("Recorded The Hollow Crown").closest(".toast") as HTMLElement;

      // focus inside, pointer passes over and leaves: still held
      const listen = within(toast).getByRole("button", { name: "Listen" });
      act(() => listen.focus());
      fireEvent.mouseEnter(toast);
      fireEvent.mouseLeave(toast);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(NOTICE_AUTO_DISMISS_MS * 2);
      });
      expect(screen.getByText("Recorded The Hollow Crown")).toBeInTheDocument();

      // moving focus between the notice's own buttons is not leaving it
      const x = within(toast).getByRole("button", { name: "Dismiss: Recorded The Hollow Crown" });
      act(() => x.focus());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(NOTICE_AUTO_DISMISS_MS * 2);
      });
      expect(screen.getByText("Recorded The Hollow Crown")).toBeInTheDocument();

      // hovered with focus gone: still held
      fireEvent.mouseEnter(toast);
      act(() => x.blur());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(NOTICE_AUTO_DISMISS_MS * 2);
      });
      expect(screen.getByText("Recorded The Hollow Crown")).toBeInTheDocument();

      // neither: it clears itself
      fireEvent.mouseLeave(toast);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(NOTICE_AUTO_DISMISS_MS + 100);
      });
      expect(screen.queryByText("Recorded The Hollow Crown")).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("dismissing a focused notice by keyboard hands focus to the next one", async () => {
    const user = userEvent.setup();
    const queryClient = makeQueryClient();
    const { setJobs } = setup([]);
    renderWithProviders(<JobNotices />, { queryClient });
    await waitFor(() => expect(queryClient.getQueryData(["jobs", "recent"])).toBeDefined());
    setJobs([
      done({ job_id: "f1", book_id: "b1", kind: "render", state: "failed", error: "x", finished_at: "2026-10-22T10:00:00Z" }),
      done({ job_id: "f2", book_id: "b1", kind: "attribute", state: "failed", error: "y", finished_at: "2026-10-22T10:01:00Z" }),
    ]);
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ["jobs", "recent"] });
    });
    const first = await screen.findByRole("button", { name: "Dismiss: Recording The Hollow Crown stopped with a problem" });
    first.focus();
    await user.keyboard("{Enter}");
    expect(screen.queryByText("Recording The Hollow Crown stopped with a problem")).not.toBeInTheDocument();
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Dismiss: Couldn't find the speakers in The Hollow Crown" }),
    );
  });
});
