import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { makeJob } from "../test/fixtures";
import { mockApi, renderWithProviders } from "../test/utils";
import { TransportBar } from "./TransportBar";

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
});
