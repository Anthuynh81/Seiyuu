import { describe, expect, it } from "vitest";

import { makeJob } from "../test/fixtures";
import { explainJobError, type JobErrorId, unresolvedFailure } from "./jobErrors";

describe("explainJobError", () => {
  // the exact strings the backend raises (job errors are "<ExceptionType>: <message>")
  const table: [string, string | { code: string; message: string }, JobErrorId][] = [
    ["ollama_unreachable code", { code: "ollama_unreachable", message: "whatever" }, "ollama_unreachable"],
    [
      "preflight unreachable job error",
      "OllamaUnreachableError: Ollama isn't reachable at http://localhost:11434 - start it with 'ollama serve' (or fix OLLAMA_BASE_URL) ([WinError 10061])",
      "ollama_unreachable",
    ],
    [
      "provider 'cannot reach Ollama' text",
      "AttributionError: cannot reach Ollama at http://localhost:11434/api/chat: is `ollama serve` running and is model 'qwen2.5:7b' pulled (`ollama pull qwen2.5:7b`)? (<urlopen error>)",
      "ollama_unreachable",
    ],
    ["model_missing code", { code: "model_missing", message: "x" }, "model_missing"],
    [
      "preflight model missing job error",
      "OllamaModelMissingError: model qwen2.5:7b isn't pulled in Ollama at http://localhost:11434 - run 'ollama pull qwen2.5:7b'",
      "model_missing",
    ],
    ["ffmpeg missing", "AssembleError: ffmpeg not found on PATH; install it and retry", "ffmpeg_missing"],
    ["render_active code", { code: "render_active", message: "a render job for 'b1' is running" }, "render_active"],
    [
      "render_active text",
      "a render job for 'b1' is queued; cancel it first — this write would invalidate its cost quote mid-run",
      "render_active",
    ],
    ["gpu_busy code", { code: "gpu_busy", message: "busy" }, "gpu_busy"],
    ["gpu_busy_retry code", { code: "gpu_busy_retry", message: "busy" }, "gpu_busy"],
    [
      "GpuBusyError job error (another process holds gpu.lock)",
      "GpuBusyError: cannot load kokoro: another seiyuu process holds the GPU (lock: data/gpu.lock).",
      "gpu_other_process",
    ],
    ["restart reconciliation", "interrupted: server stopped mid-job", "interrupted"],
    ["engine_cold code", { code: "engine_cold", message: "cold" }, "engine_cold"],
    ["audition_in_flight code", { code: "audition_in_flight", message: "busy" }, "audition_in_flight"],
    ["quote_expired code", { code: "quote_expired", message: "x" }, "quote_expired"],
    ["quote expired job error", "CostGateError: cost token expired; re-run estimate-cost", "quote_expired"],
    [
      "single-voice merge identity",
      "RenderError: b1: refusing single-voice chapter-subset render: voice differ(s) from the existing manifest (...); render the full book to change the voice",
      "merge_identity",
    ],
    [
      "multivoice merge identity",
      "RenderError: b1: refusing multivoice chapter-subset render: the voice assignment (assignments) differs from the existing manifest's assignment snapshot; ...",
      "merge_identity",
    ],
  ];

  it.each(table)("%s -> %s", (_label, input, id) => {
    const out = explainJobError(input);
    expect(out.id).toBe(id);
    expect(out.known).toBe(true);
    expect(out.summary).not.toMatch(/Error:|_/); // plain words, never the raw code
    expect(out.fix).toBeTruthy();
  });

  it("keeps the raw message as details for the disclosure", () => {
    const raw = "AssembleError: ffmpeg not found on PATH; install it and retry";
    expect(explainJobError(raw).details).toBe(raw);
  });

  it("names the missing model in the pull command", () => {
    const out = explainJobError(
      "OllamaModelMissingError: model qwen3.5:9b isn't pulled in Ollama at http://localhost:11434 - run 'ollama pull qwen3.5:9b'",
    );
    expect(out.command).toBe("ollama pull qwen3.5:9b");
    expect(explainJobError({ code: "ollama_unreachable", message: "" }).command).toBe("ollama serve");
  });

  it("a GpuBusyError job failure never claims a recording is running here", () => {
    const out = explainJobError("GpuBusyError: cannot load kokoro: another seiyuu process holds the GPU (lock: x).");
    expect(out.summary).toMatch(/Another Seiyuu program/);
    expect(out.summary).not.toMatch(/Recording in progress/);
    expect(explainJobError({ code: "gpu_busy", message: "" }).summary).toBe("Recording in progress, try after it finishes.");
  });

  it("the provider's 'cannot reach Ollama' asks for both checks and names the model", () => {
    const out = explainJobError(
      "AttributionError: cannot reach Ollama at http://localhost:11434/api/chat: is `ollama serve` running and is model 'qwen2.5:7b' pulled (`ollama pull qwen2.5:7b`)? (HTTP Error 500)",
    );
    expect(out.id).toBe("ollama_unreachable");
    expect(out.fix).toMatch(/running/);
    expect(out.fix).toMatch(/downloaded/);
    expect(out.command).toBe("ollama pull qwen2.5:7b");
    // the pre-flight text means it is truly down: start it
    expect(explainJobError("OllamaUnreachableError: Ollama isn't reachable at http://localhost:11434").command).toBe(
      "ollama serve",
    );
  });

  it("a known code wins over the message text", () => {
    expect(explainJobError({ code: "gpu_busy", message: "ffmpeg not found on PATH" }).id).toBe("gpu_busy");
  });

  it("an unknown error falls back to the raw message under details", () => {
    const out = explainJobError("RuntimeError: CUDA out of memory");
    expect(out).toMatchObject({ id: "unknown", known: false, details: "RuntimeError: CUDA out of memory" });
    expect(out.summary).toBe("Something went wrong.");
  });

  it("does not mistake an ffmpeg failure that isn't 'not found' for a missing install", () => {
    expect(explainJobError("AssembleError: ffmpeg failed for ch1.mp3: boom").id).toBe("unknown");
  });

  it("handles a missing message", () => {
    expect(explainJobError(null)).toMatchObject({ id: "unknown", details: null });
    expect(explainJobError(undefined).fix).toBeTruthy();
  });
});

describe("unresolvedFailure", () => {
  const at = (min: number) => `2026-10-22T10:${String(min).padStart(2, "0")}:00Z`;

  it("returns the book's most recent failed job", () => {
    const older = makeJob({ job_id: "f1", book_id: "b1", kind: "render", state: "failed", created_at: at(1) });
    const newer = makeJob({ job_id: "f2", book_id: "b1", kind: "assemble", state: "failed", created_at: at(5) });
    // order from the API is not load-bearing
    expect(unresolvedFailure([older, newer], "b1")?.job_id).toBe("f2");
  });

  it("ignores other books and returns null when nothing failed", () => {
    const other = makeJob({ job_id: "x", book_id: "b2", state: "failed", created_at: at(1) });
    const ok = makeJob({ job_id: "y", book_id: "b1", state: "succeeded", created_at: at(2) });
    expect(unresolvedFailure([other, ok], "b1")).toBeNull();
    expect(unresolvedFailure(undefined, "b1")).toBeNull();
  });

  it("is superseded by a newer job of the same kind, in any state", () => {
    const failed = makeJob({ job_id: "f", book_id: "b1", kind: "render", state: "failed", created_at: at(1) });
    for (const state of ["queued", "running", "succeeded", "canceled"] as const) {
      const retry = makeJob({ job_id: "r", book_id: "b1", kind: "render", state, created_at: at(3) });
      expect(unresolvedFailure([retry, failed], "b1")).toBeNull();
    }
  });

  it("is NOT superseded by a newer job of a different kind, or an older same-kind one", () => {
    const failed = makeJob({ job_id: "f", book_id: "b1", kind: "render", state: "failed", created_at: at(5) });
    const otherKind = makeJob({ job_id: "a", book_id: "b1", kind: "attribute", state: "succeeded", created_at: at(9) });
    const olderSame = makeJob({ job_id: "o", book_id: "b1", kind: "render", state: "succeeded", created_at: at(1) });
    expect(unresolvedFailure([otherKind, failed, olderSame], "b1")?.job_id).toBe("f");
  });
});
