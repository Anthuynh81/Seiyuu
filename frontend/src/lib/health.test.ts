import { describe, expect, it } from "vitest";

import { ApiError, BACKEND_START_COMMAND } from "../api/client";
import { makeEngine, makeJob, makeSystemProbe } from "../test/fixtures";
import { buildChecklist, overallHealth, warmupErrorText, warmupJobFor } from "./health";

const down = new ApiError(0, "backend_down", "The Seiyuu backend isn't running.");

function rowsFor(over: Partial<Parameters<typeof buildChecklist>[0]> = {}) {
  return buildChecklist({
    probe: makeSystemProbe(),
    probeError: null,
    engines: [makeEngine()],
    liveJobs: [],
    ...over,
  });
}

const byId = (rows: ReturnType<typeof rowsFor>, id: string) => rows.find((r) => r.id === id)!;

describe("buildChecklist", () => {
  it("all green: backend, Ollama, model, ffmpeg and each local engine — and Ready", () => {
    const rows = rowsFor({
      engines: [makeEngine(), makeEngine({ engine_id: "elevenlabs", paid: true, uses_gpu: false, weights_cached: null })],
    });
    expect(rows.map((r) => r.id)).toEqual(["backend", "ollama", "model", "ffmpeg", "engine:kokoro"]); // cloud engine omitted
    expect(rows.every((r) => r.state === "ok")).toBe(true);
    expect(overallHealth(rows, null, true)).toBe("ready");
  });

  it("backend down: only the backend row, carrying the start command — Backend offline", () => {
    const rows = rowsFor({ probe: undefined, probeError: down, engines: undefined });
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ id: "backend", state: "missing", status: "Not running", command: BACKEND_START_COMMAND });
    expect(overallHealth(rows, down, false)).toBe("offline");
  });

  it("still loading: Checking…", () => {
    const rows = rowsFor({ probe: undefined, engines: undefined });
    expect(rows[0].state).toBe("checking");
    expect(overallHealth(rows, null, false)).toBe("checking");
  });

  it("Ollama unreachable -> ollama serve fix, model waits on it — Needs setup", () => {
    const rows = rowsFor({
      probe: makeSystemProbe({ ollama: { base_url: "x", reachable: false, model_installed: null } }),
    });
    expect(byId(rows, "ollama")).toMatchObject({ state: "missing", required: true, command: "ollama serve" });
    expect(byId(rows, "model")).toMatchObject({ state: "unknown", status: "Waiting for Ollama" });
    expect(overallHealth(rows, null, true)).toBe("setup");
  });

  it("model not pulled -> ollama pull <configured model>", () => {
    const rows = rowsFor({
      probe: makeSystemProbe({ ollama: { base_url: "x", reachable: true, model_installed: false } }),
    });
    expect(byId(rows, "model")).toMatchObject({ state: "missing", command: "ollama pull qwen2.5:7b" });
    expect(overallHealth(rows, null, true)).toBe("setup");
  });

  it("ffmpeg missing is required", () => {
    const rows = rowsFor({ probe: makeSystemProbe({ ffmpeg_available: false }) });
    expect(byId(rows, "ffmpeg")).toMatchObject({ state: "missing", required: true });
    expect(overallHealth(rows, null, true)).toBe("setup");
  });

  it("Claude attribution: Ollama is optional and the local-model row is dropped", () => {
    const base = makeSystemProbe();
    const rows = rowsFor({
      probe: makeSystemProbe({
        attribution: { ...base.attribution, provider: "anthropic", model: "claude-test" },
        ollama: { base_url: "x", reachable: false, model_installed: null },
      }),
    });
    expect(byId(rows, "ollama").required).toBe(false);
    expect(rows.find((r) => r.id === "model")).toBeUndefined();
    expect(overallHealth(rows, null, true)).toBe("ready");
  });

  it("uncached engine weights are optional: Download now offered, still Ready", () => {
    const rows = rowsFor({ engines: [makeEngine({ engine_id: "chatterbox", weights_cached: false })] });
    expect(byId(rows, "engine:chatterbox")).toMatchObject({
      state: "missing",
      required: false,
      canDownload: true,
      label: "Chatterbox voice model",
    });
    expect(overallHealth(rows, null, true)).toBe("ready");
  });

  it("IndexTTS-2 needs a manual install: no Download now", () => {
    const rows = rowsFor({ engines: [makeEngine({ engine_id: "indextts2", weights_cached: false })] });
    const row = byId(rows, "engine:indextts2");
    expect(row.canDownload).toBeUndefined();
    expect(row.fix).toContain("INDEXTTS2_CHECKPOINTS_DIR");
  });

  it("a live warm-up job shows queued/running from the jobs poll", () => {
    const engines = [makeEngine({ engine_id: "chatterbox", weights_cached: false })];
    const queued = makeJob({ kind: "warmup", book_id: "engine:chatterbox", state: "queued" });
    expect(byId(rowsFor({ engines, liveJobs: [queued] }), "engine:chatterbox")).toMatchObject({
      state: "busy",
      status: "Queued",
    });
    const running = { ...queued, state: "running" as const };
    expect(byId(rowsFor({ engines, liveJobs: [running] }), "engine:chatterbox").status).toBe(
      "Downloading and loading…",
    );
    // another engine's warm-up (or a book job) does not count
    expect(warmupJobFor("kokoro", [queued, makeJob()])).toBeUndefined();
  });

  it("a slim payload (older backend) degrades to 'not checked' instead of crashing", () => {
    const slim = { ...makeSystemProbe() } as Partial<ReturnType<typeof makeSystemProbe>>;
    delete slim.ollama;
    delete slim.ffmpeg_available;
    const rows = rowsFor({ probe: slim as ReturnType<typeof makeSystemProbe> });
    expect(byId(rows, "ollama").state).toBe("unknown");
    expect(byId(rows, "ffmpeg").state).toBe("unknown");
  });
});

describe("staleness: the latest answer wins over cached data", () => {
  it("up, then down: a cached probe can't keep the pill on Ready or the backend row Running", () => {
    // react-query v5 keeps the last good data AND sets error after a failed refetch
    const rows = rowsFor({ probe: makeSystemProbe(), probeError: down });
    expect(rows).toHaveLength(1); // no stale greens beside a red backend row
    expect(rows[0]).toMatchObject({ id: "backend", status: "Not running", command: BACKEND_START_COMMAND });
    expect(overallHealth(rows, down, true)).toBe("offline");
  });

  it("a running backend that answers with an error is Needs setup, not Backend offline", () => {
    const locked = new ApiError(503, "lock_timeout", "database is locked");
    for (const probe of [undefined, makeSystemProbe()]) {
      const rows = rowsFor({ probe, probeError: locked });
      expect(rows).toHaveLength(1);
      expect(rows[0]).toMatchObject({ state: "missing", status: "Not answering properly" });
      expect(rows[0].command).toBeUndefined(); // it IS running: no "start it" command
      expect(overallHealth(rows, locked, probe !== undefined)).toBe("setup");
    }
  });

  it("a failed engine catalog says so instead of silently listing nothing", () => {
    const rows = rowsFor({ engines: null });
    expect(byId(rows, "engines")).toMatchObject({ state: "unknown", status: "Couldn't check", required: false });
    expect(overallHealth(rows, null, true)).toBe("ready");
  });
});

describe("warmupErrorText", () => {
  it("maps the warm-up route's real refusals (the enqueue ones) to plain words", () => {
    expect(
      warmupErrorText(new ApiError(409, "duplicate_job", "a warmup job for 'engine:chatterbox' is already queued")),
    ).toBe("Already queued, it will start on its own.");
    expect(warmupErrorText(new ApiError(503, "not_ready", "runner not started"))).toBe(
      "The backend is still starting, try again in a moment.",
    );
    expect(warmupErrorText(new ApiError(409, "nothing_to_warm", "cloud engine"))).toBe("cloud engine");
  });
});
