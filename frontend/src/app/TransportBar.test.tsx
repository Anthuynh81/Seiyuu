import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useEffect } from "react";
import { describe, expect, it } from "vitest";

import { makeJob } from "../test/fixtures";
import { mockApi, renderWithProviders } from "../test/utils";
import { TransportBar } from "./TransportBar";
import { usePlayer } from "./usePlayer";

/** Loads one clip into the player so the /listen route mounts the audio transport. */
function LoadOneClip() {
  const player = usePlayer();
  useEffect(() => {
    player?.load("b1", "Chapter 1", [{ src: "/audio/a.wav", duration: 10, key: "a", speaker: "N", words: [] }]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

describe("TransportBar", () => {
  it("reads idle when no jobs are queued or running", async () => {
    mockApi(); // pre-registered GET /api/jobs -> {jobs: []}
    renderWithProviders(<TransportBar />);
    expect(await screen.findByText("console idle — no jobs queued or running")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "cancel" })).not.toBeInTheDocument();
  });

  it("shows the running job's kind, book, and progress with an enabled cancel", async () => {
    mockApi().get("/api/jobs", { jobs: [makeJob({ progress_text: "chapter 2/9" })] });
    renderWithProviders(<TransportBar />);
    expect(await screen.findByText("render · demo")).toBeInTheDocument();
    expect(screen.getByText("chapter 2/9")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "cancel" })).toBeEnabled();
  });

  it("falls back to the queued job and the waiting placeholder when nothing runs yet", async () => {
    mockApi().get("/api/jobs", {
      jobs: [makeJob({ state: "queued", progress_text: "", started_at: null })],
    });
    renderWithProviders(<TransportBar />);
    expect(await screen.findByText("waiting for the worker…")).toBeInTheDocument();
    expect(screen.getByText("queued")).toBeInTheDocument();
  });

  it("prefers the running job over an earlier queued one — display and cancel both target it", async () => {
    const user = userEvent.setup();
    const server = mockApi().get("/api/jobs", {
      jobs: [
        makeJob({ job_id: "j-q", kind: "attribute", state: "queued", progress_text: "", started_at: null }),
        makeJob({ job_id: "j-r", kind: "render", state: "running", progress_text: "chapter 3/9" }),
      ],
    });
    server.post("/api/jobs/j-r/cancel", makeJob({ job_id: "j-r", kind: "render", cancel_requested: true }));
    renderWithProviders(<TransportBar />);

    // the bar shows the RUNNING render, not the queued attribute listed before it
    expect(await screen.findByText("render · demo")).toBeInTheDocument();
    expect(screen.getByText("chapter 3/9")).toBeInTheDocument();
    expect(screen.queryByText("attribute · demo")).not.toBeInTheDocument();

    // ...and cancel targets the running job's id — never the queued one's
    await user.click(screen.getByRole("button", { name: "cancel" }));
    await waitFor(() => expect(server.lastCall("POST", "/cancel")?.url).toBe("/api/jobs/j-r/cancel"));
  });

  it("cancel POSTs /api/jobs/{id}/cancel for the shown job", async () => {
    const user = userEvent.setup();
    const server = mockApi().get("/api/jobs", { jobs: [makeJob({ job_id: "job-42" })] });
    server.post("/api/jobs/job-42/cancel", makeJob({ job_id: "job-42", cancel_requested: true }));
    renderWithProviders(<TransportBar />);

    await user.click(await screen.findByRole("button", { name: "cancel" }));
    await waitFor(() => expect(server.lastCall("POST", "/api/jobs/job-42/cancel")).toBeDefined());
  });

  it("a cancel-requested running job reads canceling and disables the cancel button", async () => {
    mockApi().get("/api/jobs", { jobs: [makeJob({ cancel_requested: true })] });
    renderWithProviders(<TransportBar />);
    expect(await screen.findByRole("button", { name: "canceling…" })).toBeDisabled();
    expect(screen.getByText("canceling")).toBeInTheDocument();
  });

  it("the speed key cycles the playback rate and persists it", async () => {
    const user = userEvent.setup();
    mockApi();
    renderWithProviders(
      <>
        <LoadOneClip />
        <TransportBar />
      </>,
      { route: "/listen" },
    );

    const key = await screen.findByRole("button", { name: "playback speed" });
    expect(key).toHaveTextContent("1×");

    await user.click(key);
    expect(key).toHaveTextContent("1.25×");
    expect(localStorage.getItem("seiyuu.rate")).toBe("1.25");

    // a full lap comes back around to 1×
    for (const expected of ["1.5×", "1.75×", "2×", "0.75×", "1×"]) {
      await user.click(key);
      expect(key).toHaveTextContent(expected);
    }
    expect(localStorage.getItem("seiyuu.rate")).toBe("1");
  });
});
