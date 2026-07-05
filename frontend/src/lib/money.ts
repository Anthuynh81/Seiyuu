import { ApiError } from "../api/client";

/** What the money flow does when POST /render refuses — the state machine the quote
    ticket lives by. Every branch mirrors a backend gate code (see render/gate.py):
    expiry re-mints silently (the design contract), drift/mismatch/used stamp the dead
    ticket so the user sees WHY the price must be re-shown, anything else surfaces. */
export type RenderFailureAction =
  | { kind: "confirm-full"; detail: { speakable_blocks: number; runtime_estimate_seconds: number } }
  | { kind: "remint" }
  | { kind: "stamp"; stamp: "USED" | "DRIFT"; message: string }
  | { kind: "error"; message: string };

export function classifyRenderFailure(e: unknown, hasLiveTicket: boolean): RenderFailureAction {
  if (!(e instanceof ApiError)) return { kind: "error", message: String(e) };
  if (e.code === "full_render_confirmation_required") {
    return {
      kind: "confirm-full",
      detail: e.detail as { speakable_blocks: number; runtime_estimate_seconds: number },
    };
  }
  if (e.code === "quote_expired" && hasLiveTicket) return { kind: "remint" };
  if ((e.code === "cost_drift" || e.code === "quote_mismatch" || e.code === "quote_used") && hasLiveTicket) {
    return { kind: "stamp", stamp: e.code === "quote_used" ? "USED" : "DRIFT", message: e.message };
  }
  return { kind: "error", message: e.message };
}
