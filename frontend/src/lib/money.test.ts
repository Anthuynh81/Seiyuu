import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import { classifyRenderFailure } from "./money";

const err = (code: string, message = "msg", detail: unknown = null) => new ApiError(402, code, message, detail);

describe("classifyRenderFailure — the quote ticket's state machine", () => {
  it("quote_expired with a live ticket re-mints silently (the design contract)", () => {
    expect(classifyRenderFailure(err("quote_expired"), true)).toEqual({ kind: "remint" });
  });

  it("quote_expired WITHOUT a live ticket is a plain error — nothing to re-mint from", () => {
    expect(classifyRenderFailure(err("quote_expired"), false)).toEqual({ kind: "error", message: "msg" });
  });

  it("cost_drift stamps DRIFT — the user must see the new price, not a silent retry", () => {
    expect(classifyRenderFailure(err("cost_drift", "price changed"), true)).toEqual({
      kind: "stamp",
      stamp: "DRIFT",
      message: "price changed",
    });
  });

  it("quote_mismatch stamps DRIFT (the selection changed under the quote)", () => {
    expect(classifyRenderFailure(err("quote_mismatch"), true)).toMatchObject({ kind: "stamp", stamp: "DRIFT" });
  });

  it("quote_used stamps USED — single-use means single-use", () => {
    expect(classifyRenderFailure(err("quote_used"), true)).toMatchObject({ kind: "stamp", stamp: "USED" });
  });

  it("full-render confirmation carries the detail payload through", () => {
    const detail = { speakable_blocks: 2139, runtime_estimate_seconds: 3600 };
    expect(classifyRenderFailure(err("full_render_confirmation_required", "confirm", detail), false)).toEqual({
      kind: "confirm-full",
      detail,
    });
  });

  it("ceiling_exceeded (and any other gate code) surfaces as an error, never a retry", () => {
    expect(classifyRenderFailure(err("ceiling_exceeded", "over the cap"), true)).toEqual({
      kind: "error",
      message: "over the cap",
    });
  });

  it("non-ApiError failures degrade to a string error", () => {
    expect(classifyRenderFailure(new TypeError("network down"), true)).toEqual({
      kind: "error",
      message: "TypeError: network down",
    });
  });
});
