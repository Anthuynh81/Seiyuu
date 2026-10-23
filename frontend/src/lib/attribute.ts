import type { SystemProbeOut } from "../api/types";

/** Pure helpers for the "find the speakers" panel: the one reason it can't start right now,
    the chapter list a re-run reads, and whether a re-run would just replay the cache. All
    derived from server data; nothing is remembered locally. */

export type AttributeProvider = "local" | "anthropic";

export interface Reader {
  provider: AttributeProvider;
  model: string;
}

export interface Blocker {
  /** One plain sentence: why the button is off. */
  reason: string;
  /** The exact command that fixes it, shown copyable. */
  command: string;
}

/** Why a run can't start, from the live system probe — or null when nothing is known to be
    wrong. Only a LOCAL reader needs Ollama. The probe's `model_installed` is about ONE model:
    the server's configured LOCAL model (settings.attribution_model) — even when the server's
    default reader is Claude, in which case `probe.attribution.model` is the Claude id, not the
    one checked. So the model check only applies when we know which local model was checked
    (`probe.attribution` when its provider is local, else `localModel` from /api/settings) and
    the reader uses exactly that one; any other local model is left to the server's own
    pre-flight (it refuses with the fix). A probe that failed or hasn't answered blocks
    nothing: the backend row of the health pill already says so, and the server refuses for
    real if it must. */
export function attributeBlocker({
  probe,
  probeError,
  reader,
  localModel,
}: {
  probe: SystemProbeOut | undefined;
  probeError?: unknown;
  reader: Reader | null;
  /** The server's local model when its default reader is Claude (from /api/settings). */
  localModel?: string | null;
}): Blocker | null {
  if (!probe || probeError || !reader || reader.provider !== "local") return null;
  // an older backend's slimmer payload has no ollama block: not checked, not blocked
  const ollama = probe.ollama;
  if (!ollama) return null;
  if (ollama.reachable === false) {
    return {
      reason: "Ollama isn't running, and it's what finds the speakers on this PC. Start it, then come back:",
      command: "ollama serve",
    };
  }
  const configured = probedLocalModel(probe, localModel);
  if (ollama.model_installed === false && configured !== null && sameModel(reader.model, configured)) {
    return {
      reason: `The speaker-finding model (${configured}) isn't downloaded yet. Download it once (a few GB):`,
      command: `ollama pull ${configured}`,
    };
  }
  return null;
}

/** The local model the probe's `model_installed` is about, or null when we can't tell. */
export function probedLocalModel(probe: SystemProbeOut | undefined, localModel?: string | null): string | null {
  if (!probe) return null;
  if (probe.attribution.provider === "local") return probe.attribution.model;
  return localModel?.trim() ? localModel : null;
}

function sameModel(a: string, b: string): boolean {
  return a.trim().toLowerCase() === b.trim().toLowerCase();
}

/** "3, 5-7" → [3, 5, 6, 7] (sorted, no repeats), or a plain error. */
export function parseChapterList(
  text: string,
  chapterCount: number,
): { chapters: number[]; error?: undefined } | { chapters?: undefined; error: string } {
  // "5 - 7" / "5 -7" are ranges too: close up the spaces around a dash before splitting
  const parts = text
    .replace(/\s*[-–]\s*/g, "-")
    .split(/[,\s]+/)
    .map((p) => p.trim())
    .filter(Boolean);
  if (parts.length === 0) return { error: "List the chapters to re-read, for example 3, 5-7." };
  const out = new Set<number>();
  for (const part of parts) {
    const m = /^(\d+)(?:-(\d+))?$/.exec(part);
    if (!m) return { error: `"${part}" isn't a chapter number. Use numbers and ranges, for example 3, 5-7.` };
    const from = Number(m[1]);
    const to = m[2] === undefined ? from : Number(m[2]);
    if (to < from) return { error: `"${part}" runs backwards. Write the lower chapter first.` };
    if (from < 1 || to > chapterCount) {
      return {
        error: `This book has chapters 1 to ${chapterCount}, so "${part}" is out of range.`,
      };
    }
    for (let i = from; i <= to; i++) out.add(i);
  }
  return { chapters: [...out].sort((a, b) => a - b) };
}

/** The reader + prompt that produced the current speakers (the report's provenance: the LAST
    run's; chapter re-runs overwrite it). */
export interface Provenance {
  provider_id: string;
  model_id: string;
  prompt_version: string;
}

/** Has Seiyuu's speaker-finding prompt changed since these speakers were found? `promptVersion`
    is the server's configured one (/api/system attribution.prompt_version). Unknown → false. */
export function promptChanged(current: Provenance | null | undefined, promptVersion: string | undefined): boolean {
  return !!current && !!promptVersion && current.prompt_version !== promptVersion;
}

/** A re-run with the SAME provider, model AND prompt that produced the current speakers
    replays the attribution cache (its key is provider + model + prompt version + chunk) and
    changes nothing. Returns the plain reason it's refused, or null when anything differs —
    a prompt upgrade makes even the same model a real re-read. `promptVersion` is the prompt
    a run would use now (the server's configured one). */
export function sameReaderReason(
  current: Provenance | null | undefined,
  chosen: Reader,
  promptVersion: string | undefined,
): string | null {
  if (!chosen.model.trim()) return "Type the model to read with.";
  if (!current) return null;
  if (promptChanged(current, promptVersion)) return null;
  if (current.provider_id === chosen.provider && sameModel(current.model_id, chosen.model)) {
    return `These speakers were found with ${current.model_id}. The same model gives the same answer (Seiyuu reuses its earlier result), so choose a different model or provider.`;
  }
  return null;
}

export function chapterSpan(chapters: number[]): string {
  if (chapters.length === 0) return "the whole book";
  if (chapters.length === 1) return `chapter ${chapters[0]}`;
  return `${chapters.length} chapters`;
}
