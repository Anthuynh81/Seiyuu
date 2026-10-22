import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { JobOut } from "../api/types";
import { makeEngine, makeJob, makeSystemProbe } from "../test/fixtures";
import { errorResponse, mockApi, renderWithProviders } from "../test/utils";
import { SetupChecklist } from "./SetupChecklist";

const row = (name: string) => screen.getByRole("listitem", { name });

describe("SetupChecklist", () => {
  it("lists backend, Ollama, the attribution model, ffmpeg and each local engine with plain fixes", async () => {
    mockApi()
      .get(
        "/api/system?probe=true",
        makeSystemProbe({
          ffmpeg_available: false,
          ollama: { base_url: "http://localhost:11434", reachable: false, model_installed: null },
        }),
      )
      .get("/api/engines", {
        engines: [
          makeEngine({ engine_id: "kokoro", weights_cached: true, resident: true }),
          makeEngine({ engine_id: "chatterbox", weights_cached: false }),
          makeEngine({ engine_id: "elevenlabs", paid: true, uses_gpu: false, weights_cached: null }),
        ],
      });
    renderWithProviders(<SetupChecklist />);

    expect(await screen.findByRole("listitem", { name: "Kokoro voice model" })).toBeInTheDocument();
    expect(within(row("Seiyuu backend")).getByText("Running")).toBeInTheDocument();
    expect(within(row("Ollama")).getByText("Not running")).toBeInTheDocument();
    expect(within(row("Ollama")).getByText("ollama serve")).toBeInTheDocument();
    expect(within(row("Speaker-detection model")).getByText("Waiting for Ollama")).toBeInTheDocument();
    expect(within(row("ffmpeg")).getByText("Not found")).toBeInTheDocument();
    expect(within(row("Kokoro voice model")).getByText("Downloaded · loaded")).toBeInTheDocument();
    expect(within(row("Chatterbox voice model")).getByText("optional")).toBeInTheDocument();
    expect(within(row("Chatterbox voice model")).getByRole("button", { name: "Download now" })).toBeInTheDocument();
    // a cloud engine has no local weights: no row
    expect(screen.queryByRole("listitem", { name: /ElevenLabs/ })).not.toBeInTheDocument();
    // a downloaded engine offers no download
    expect(within(row("Kokoro voice model")).queryByRole("button")).not.toBeInTheDocument();
  });

  it("model not pulled shows the exact ollama pull command for the configured model", async () => {
    mockApi()
      .get(
        "/api/system?probe=true",
        makeSystemProbe({ ollama: { base_url: "x", reachable: true, model_installed: false } }),
      )
      .get("/api/engines", { engines: [] });
    renderWithProviders(<SetupChecklist />);
    expect(await screen.findByText("ollama pull qwen2.5:7b")).toBeInTheDocument();
  });

  it("Download now queues the existing warm-up job, then shows Queued from the live-jobs poll", async () => {
    const user = userEvent.setup();
    let live: JobOut[] = [];
    const queued = makeJob({ job_id: "w1", kind: "warmup", book_id: "engine:chatterbox", state: "queued", started_at: null });
    const server = mockApi()
      .get("/api/system?probe=true", makeSystemProbe())
      .get("/api/engines", { engines: [makeEngine({ engine_id: "chatterbox", weights_cached: false })] });
    server.on("GET", "/api/jobs", () => ({ jobs: live }));
    server.on("POST", "/api/engines/chatterbox/warmup", () => {
      live = [queued];
      return queued;
    });
    renderWithProviders(<SetupChecklist />);

    await user.click(await screen.findByRole("button", { name: "Download now" }));

    await waitFor(() => expect(within(row("Chatterbox voice model")).getByText("Queued")).toBeInTheDocument());
    expect(server.calls.filter((c) => c.method === "POST")).toEqual([
      { method: "POST", url: "/api/engines/chatterbox/warmup", body: null },
    ]);
    expect(within(row("Chatterbox voice model")).getByText(/starts on its own/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download now" })).not.toBeInTheDocument();
  });

  it("a refused warm-up is shown plainly and never retried", async () => {
    const user = userEvent.setup();
    const server = mockApi()
      .get("/api/system?probe=true", makeSystemProbe())
      .get("/api/engines", { engines: [makeEngine({ engine_id: "chatterbox", weights_cached: false })] })
      .error("POST", "/api/engines/chatterbox/warmup", 503, "not_ready", "runner not started");
    renderWithProviders(<SetupChecklist />);

    await user.click(await screen.findByRole("button", { name: "Download now" }));
    expect(await screen.findByText("The backend is still starting, try again in a moment.")).toBeInTheDocument();
    expect(screen.queryByText("runner not started")).not.toBeInTheDocument();
    expect(server.calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });

  it("stays on queuing… until the live-jobs refetch has the job (no re-offered button)", async () => {
    const user = userEvent.setup();
    let live: JobOut[] = [];
    let releaseJobs: (() => void) | null = null;
    const queued = makeJob({ job_id: "w1", kind: "warmup", book_id: "engine:chatterbox", state: "queued", started_at: null });
    const server = mockApi()
      .get("/api/system?probe=true", makeSystemProbe())
      .get("/api/engines", { engines: [makeEngine({ engine_id: "chatterbox", weights_cached: false })] });
    server.on("GET", "/api/jobs", () => ({ jobs: live }));
    renderWithProviders(<SetupChecklist />);
    const button = await screen.findByRole("button", { name: "Download now" });

    server.on("POST", "/api/engines/chatterbox/warmup", () => {
      live = [queued];
      // hold the follow-up live-jobs refetch open: the gap the button must not reopen in
      server.on("GET", "/api/jobs", () => new Promise((r) => (releaseJobs = () => r({ jobs: live }))));
      return queued;
    });
    await user.click(button);

    await waitFor(() => expect(releaseJobs).not.toBeNull());
    expect(within(row("Chatterbox voice model")).getByRole("button", { name: "queuing…" })).toBeDisabled();
    releaseJobs!();
    await waitFor(() => expect(within(row("Chatterbox voice model")).getByText("Queued")).toBeInTheDocument());
  });

  it("duplicate_job (queued elsewhere) refetches live jobs and shows Queued, not the raw message", async () => {
    const user = userEvent.setup();
    let live: JobOut[] = [];
    const queued = makeJob({ job_id: "w9", kind: "warmup", book_id: "engine:chatterbox", state: "queued", started_at: null });
    const server = mockApi()
      .get("/api/system?probe=true", makeSystemProbe())
      .get("/api/engines", { engines: [makeEngine({ engine_id: "chatterbox", weights_cached: false })] });
    server.on("GET", "/api/jobs", () => ({ jobs: live }));
    server.on("POST", "/api/engines/chatterbox/warmup", () => {
      live = [queued]; // another tab / the CLI got there first
      return errorResponse(409, "duplicate_job", "a warmup job for 'engine:chatterbox' is already queued", queued);
    });
    renderWithProviders(<SetupChecklist />);

    await user.click(await screen.findByRole("button", { name: "Download now" }));
    await waitFor(() => expect(within(row("Chatterbox voice model")).getByText("Queued")).toBeInTheDocument());
    expect(screen.queryByText(/engine:chatterbox/)).not.toBeInTheDocument();
    expect(screen.queryByText("Already queued, it will start on its own.")).not.toBeInTheDocument();
  });

  it("a failed engine catalog shows a Couldn't check row instead of nothing", async () => {
    mockApi()
      .get("/api/system?probe=true", makeSystemProbe())
      .error("GET", "/api/engines", 503, "lock_timeout", "database is locked");
    renderWithProviders(<SetupChecklist />);
    expect(await within(await screen.findByRole("listitem", { name: "Voice models" })).findByText("Couldn't check")).toBeInTheDocument();
  });

  it("IndexTTS-2 explains its manual install instead of offering Download now", async () => {
    mockApi()
      .get("/api/system?probe=true", makeSystemProbe())
      .get("/api/engines", { engines: [makeEngine({ engine_id: "indextts2", weights_cached: false })] });
    renderWithProviders(<SetupChecklist />);
    const r = await screen.findByRole("listitem", { name: "IndexTTS-2 voice model" });
    expect(within(r).getByText(/INDEXTTS2_CHECKPOINTS_DIR/)).toBeInTheDocument();
    expect(within(r).queryByRole("button")).not.toBeInTheDocument();
  });

  it("backend down: one row with the start command, nothing else guessed", async () => {
    mockApi().on("GET", "/api/system", () => new Response("", { status: 500 }));
    renderWithProviders(<SetupChecklist />);
    expect(await screen.findByText("uv run uvicorn seiyuu.api.main:app")).toBeInTheDocument();
    expect(within(row("Seiyuu backend")).getByText("Not running")).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });
});
