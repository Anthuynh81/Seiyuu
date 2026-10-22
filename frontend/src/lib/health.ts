import { ApiError, BACKEND_START_COMMAND, isBackendDown } from "../api/client";
import type { EngineInfo, JobOut, SystemProbeOut } from "../api/types";

/** Pure derivation of the setup checklist and the NavRail health pill from server data
    (the probe, the engine catalog, the live jobs poll). The UI holds no truth: every row is
    recomputed from the latest fetch, nothing is remembered locally. */

export type CheckState = "ok" | "missing" | "busy" | "checking" | "unknown";

export interface CheckRow {
  id: string; // "backend" | "ollama" | "model" | "ffmpeg" | `engine:<id>`
  label: string;
  state: CheckState;
  /** Required rows decide Ready vs Needs setup; optional rows only inform. */
  required: boolean;
  status: string; // short plain-words status ("Running", "Not downloaded yet")
  fix?: string; // what to do, in a sentence
  command?: string; // the exact command to run, shown copyable
  engineId?: string; // set on engine rows
  /** The existing warm-up job can fetch this engine's weights ("Download now"). */
  canDownload?: boolean;
}

export type HealthLevel = "checking" | "ready" | "setup" | "offline";

export const HEALTH_LABEL: Record<HealthLevel, string> = {
  checking: "Checking…",
  ready: "Ready",
  setup: "Needs setup",
  offline: "Backend offline",
};

/** LED grammar from index.css: solid ok/warn/err, hollow = waiting. */
export const HEALTH_LED: Record<HealthLevel, string> = {
  checking: "q",
  ready: "ok",
  setup: "warn",
  offline: "err",
};

export const ENGINE_NAMES: Record<string, string> = {
  kokoro: "Kokoro",
  chatterbox: "Chatterbox",
  indextts2: "IndexTTS-2",
  elevenlabs: "ElevenLabs",
};

export const engineName = (id: string) => ENGINE_NAMES[id] ?? id;

/** Engines whose weights the warm-up job can NOT fetch: IndexTTS-2's checkpoints are a
    separate manual download (the worker refuses to start without them). */
const MANUAL_INSTALL: Record<string, string> = {
  indextts2:
    "Needs a manual install: download the IndexTTS-2 checkpoints and set INDEXTTS2_CHECKPOINTS_DIR in .env, then restart the backend.",
};

/** The manual-install instructions for an engine the warm-up job can't fetch, if any. */
export const manualInstallFor = (engineId: string): string | undefined => MANUAL_INSTALL[engineId];

/** The live warm-up job for an engine, if one is queued or running. Warm-up jobs use the
    documented `engine:<id>` subject in the book column. */
export function warmupJobFor(engineId: string, liveJobs: JobOut[] | undefined): JobOut | undefined {
  return liveJobs?.find((j) => j.kind === "warmup" && j.book_id === `engine:${engineId}` && !j.is_terminal);
}

function backendRow(probe: SystemProbeOut | undefined, probeError: unknown): CheckRow {
  const base = { id: "backend", label: "Seiyuu backend", required: true };
  // Error first: after a failed refetch react-query keeps the last good `data` AND sets
  // `error`. The latest answer wins — a backend that was up and then stopped is down now.
  if (isBackendDown(probeError)) {
    return {
      ...base,
      state: "missing",
      status: "Not running",
      fix: "Start it from the project folder, then come back to this page.",
      command: BACKEND_START_COMMAND,
    };
  }
  if (probeError) {
    const msg = probeError instanceof Error ? probeError.message : String(probeError);
    return {
      ...base,
      state: "missing",
      status: "Not answering properly",
      fix: `It's running but answered with an error (${msg}). Try "check again" in a moment.`,
    };
  }
  if (probe) return { ...base, state: "ok", status: "Running" };
  return { ...base, state: "checking", status: "Checking…" };
}

function engineRow(e: EngineInfo, liveJobs: JobOut[] | undefined): CheckRow {
  const base = {
    id: `engine:${e.engine_id}`,
    label: `${engineName(e.engine_id)} voice model`,
    required: false,
    engineId: e.engine_id,
  };
  const job = warmupJobFor(e.engine_id, liveJobs);
  if (job) {
    return {
      ...base,
      state: "busy",
      status: job.state === "queued" ? "Queued" : "Downloading and loading…",
      fix:
        job.state === "queued"
          ? "Waiting for the job that's running now to finish — it starts on its own."
          : "This can take several minutes the first time.",
    };
  }
  if (e.weights_cached === true) {
    return { ...base, state: "ok", status: e.resident ? "Downloaded · loaded" : "Downloaded" };
  }
  if (e.weights_cached === null) {
    return { ...base, state: "unknown", status: "Couldn't check" };
  }
  const manual = MANUAL_INSTALL[e.engine_id];
  if (manual) return { ...base, state: "missing", status: "Not installed", fix: manual };
  return {
    ...base,
    state: "missing",
    status: "Not downloaded yet",
    fix: "It downloads by itself the first time you use it (several GB), or you can download it now.",
    canDownload: e.uses_gpu,
  };
}

export function buildChecklist({
  probe,
  probeError,
  engines,
  liveJobs,
}: {
  probe: SystemProbeOut | undefined;
  probeError: unknown;
  /** undefined = still loading; null = the catalog request failed. */
  engines: EngineInfo[] | null | undefined;
  liveJobs: JobOut[] | undefined;
}): CheckRow[] {
  const rows: CheckRow[] = [backendRow(probe, probeError)];
  // nothing below is knowable without a backend answering now — a cached probe from before
  // the failure would show stale greens next to a red backend row
  if (!probe || probeError) return rows;

  // Ollama only matters when speaker attribution runs locally (the default).
  const local = probe.attribution.provider === "local";
  // tolerate an older backend's slimmer payload: the pill lives in the shell, and a missing
  // field must read as "not checked", never crash every screen
  const ollama = probe.ollama ?? { reachable: null, model_installed: null };
  const reachable = ollama.reachable;
  rows.push(
    reachable === true
      ? { id: "ollama", label: "Ollama", required: local, state: "ok", status: "Running" }
      : reachable === false
        ? local
          ? {
              id: "ollama",
              label: "Ollama",
              required: true,
              state: "missing",
              status: "Not running",
              fix: "Ollama finds who is speaking in each line. Start it with:",
              command: "ollama serve",
            }
          : {
              id: "ollama",
              label: "Ollama",
              required: false,
              state: "unknown",
              status: "Not running — not needed while speaker detection uses Claude",
            }
        : { id: "ollama", label: "Ollama", required: false, state: "unknown", status: "Not checked" },
  );

  if (local) {
    const model = probe.attribution.model;
    const installed = ollama.model_installed;
    const label = "Speaker-detection model";
    rows.push(
      installed === true
        ? { id: "model", label, required: true, state: "ok", status: `${model} downloaded` }
        : installed === false
          ? {
              id: "model",
              label,
              required: true,
              state: "missing",
              status: `${model} not downloaded`,
              fix: "Download it once (a few GB) with:",
              command: `ollama pull ${model}`,
            }
          : {
              id: "model",
              label,
              required: false,
              state: "unknown",
              status: reachable === false ? "Waiting for Ollama" : "Couldn't check",
            },
    );
  }

  rows.push(
    probe.ffmpeg_available === true
      ? { id: "ffmpeg", label: "ffmpeg", required: true, state: "ok", status: "Found" }
      : probe.ffmpeg_available !== false
        ? { id: "ffmpeg", label: "ffmpeg", required: false, state: "unknown", status: "Not checked" }
        : {
            id: "ffmpeg",
            label: "ffmpeg",
            required: true,
            state: "missing",
            status: "Not found",
            fix: "ffmpeg builds the final audiobook file. Install it, make sure it's on your PATH, then restart the backend.",
          },
  );

  if (engines === undefined) {
    rows.push({ id: "engines", label: "Voice models", required: false, state: "checking", status: "Checking…" });
  } else if (engines === null) {
    rows.push({
      id: "engines",
      label: "Voice models",
      required: false,
      state: "unknown",
      status: "Couldn't check",
      fix: 'Try "check again" in a moment.',
    });
  } else {
    for (const e of engines) if (!e.paid) rows.push(engineRow(e, liveJobs));
  }
  return rows;
}

export function overallHealth(rows: CheckRow[], probeError: unknown, hasProbe: boolean): HealthLevel {
  // "Backend offline" means exactly that (not running / not reachable) — even with a cached
  // probe from before it stopped. Any other probe error is a running backend that answered
  // badly: its backend row is a required "missing" row, so it reads as Needs setup.
  if (isBackendDown(probeError)) return "offline";
  if (!hasProbe && !probeError) return "checking";
  return rows.some((r) => r.required && r.state === "missing") ? "setup" : "ready";
}

/** Plain words for a refused Download now. POST /engines/{id}/warmup only enqueues (the job
    waits behind any GPU work), so its refusals are the enqueue ones; a refusal is shown once,
    never retried. */
export function warmupErrorText(err: unknown): string {
  if (err instanceof ApiError) {
    switch (err.code) {
      case "duplicate_job":
        return "Already queued, it will start on its own.";
      case "not_ready":
        return "The backend is still starting, try again in a moment.";
      case "backend_down":
        return "The Seiyuu backend isn't running — start it, then try again.";
      default:
        return err.message;
    }
  }
  return err instanceof Error ? err.message : String(err);
}
