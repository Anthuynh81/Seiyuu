import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import { AUDITION_MAX_CHARS, auditionLine, explainAuditionRefusal, formatSampleCost, RETRIABLE } from "./audition";

describe("auditionLine", () => {
  it("passes a short line through, whitespace-collapsed", () => {
    expect(auditionLine('  "Hello   there,"\n she said. ')).toBe('"Hello there," she said.');
  });

  it("is undefined for nothing usable, so the server's default sentence plays", () => {
    expect(auditionLine(undefined)).toBeUndefined();
    expect(auditionLine(null)).toBeUndefined();
    expect(auditionLine("   ")).toBeUndefined();
  });

  it("trims a long line to the endpoint's limit at a word boundary", () => {
    const long = Array.from({ length: 200 }, (_, i) => `word${i}`).join(" ");
    const out = auditionLine(long)!;
    expect(Array.from(out).length).toBeLessThanOrEqual(AUDITION_MAX_CHARS);
    expect(out.endsWith("…")).toBe(true);
    expect(out.slice(0, -1)).toMatch(/word\d+$/); // never half a word
  });

  it("counts code points and never splits a surrogate pair", () => {
    const out = auditionLine("😀".repeat(600))!;
    expect(Array.from(out).length).toBeLessThanOrEqual(AUDITION_MAX_CHARS);
    expect(out).not.toMatch(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/);
  });
});

describe("explainAuditionRefusal", () => {
  const err = (code: string, detail: unknown = null) => new ApiError(409, code, `raw ${code} message`, detail);

  it("engine_cold names the engine and offers the download, without guessing its size", () => {
    const r = explainAuditionRefusal(err("engine_cold"), "chatterbox");
    expect(r.kind).toBe("engine_cold");
    expect(r.text).toBe("Chatterbox hasn't been downloaded yet — it's a one-time download.");
    expect(explainAuditionRefusal(err("engine_cold"), "kokoro").text).not.toMatch(/GB/);
  });

  it("engine_cold on an engine the download job can't fetch gives the manual-install steps instead", () => {
    const r = explainAuditionRefusal(err("engine_cold"), "indextts2");
    expect(r.kind).toBe("not_installed");
    expect(r.text).toMatch(/manual install/);
    expect(RETRIABLE.has(r.kind)).toBe(false);
  });

  it("gpu_busy says what the GPU is busy with, defaulting to recording", () => {
    expect(explainAuditionRefusal(err("gpu_busy", { kind: "render" }), "kokoro").text).toBe(
      "Busy recording — try after it finishes.",
    );
    expect(explainAuditionRefusal(err("gpu_busy", { kind: "attribute" }), "kokoro").text).toBe(
      "Busy finding the speakers — try after it finishes.",
    );
    expect(explainAuditionRefusal(err("gpu_busy"), "kokoro").text).toBe("Busy recording — try after it finishes.");
  });

  it("audition_in_flight, cloud_busy and an exhausted borrow each ask to wait", () => {
    expect(explainAuditionRefusal(err("audition_in_flight"), "kokoro").kind).toBe("in_flight");
    expect(explainAuditionRefusal(err("cloud_busy"), "elevenlabs").kind).toBe("cloud_busy");
    expect(explainAuditionRefusal(err("gpu_busy_retry"), "kokoro").kind).toBe("borrow_exhausted");
  });

  it("a paid sample carries its estimate", () => {
    const r = explainAuditionRefusal(err("payment_confirmation_required", { estimated_usd: 0.0123 }), "elevenlabs");
    expect(r.kind).toBe("paid");
    expect(r.estimatedUsd).toBe(0.0123);
    expect(r.text).toContain("$0.0123");
  });

  it("the server's own message and code are kept for a tooltip, never the main text", () => {
    for (const code of ["upstream", "not_ready", "not_found", "consent_invalid", "invalid", "backend_down", "weird"]) {
      const r = explainAuditionRefusal(err(code), "elevenlabs");
      expect(r.text).not.toContain("raw");
      expect(r).toMatchObject({ code, detail: `raw ${code} message` });
    }
  });

  it("each known fault reads plainly, and only the transient ones offer try again", () => {
    const up = explainAuditionRefusal(err("upstream"), "elevenlabs");
    expect(up.text).toBe("ElevenLabs couldn't make the sample this time.");
    expect(RETRIABLE.has(up.kind)).toBe(true);
    const key = explainAuditionRefusal(err("not_ready"), "elevenlabs");
    expect(key.text).toMatch(/Cloud voices need an ElevenLabs key/);
    expect(key.text).not.toMatch(/ELEVENLABS_API_KEY/);
    expect(RETRIABLE.has(key.kind)).toBe(false);
    expect(explainAuditionRefusal(err("not_found"), "kokoro").text).toBe("This voice no longer exists — pick another one.");
    expect(explainAuditionRefusal(err("consent_invalid"), "chatterbox").text).toMatch(/re-create the voice/);
    expect(RETRIABLE.has(explainAuditionRefusal(err("backend_down"), "kokoro").kind)).toBe(true);
    expect(explainAuditionRefusal(err("weird"), "kokoro").text).toBe("The sample couldn't be made.");
  });

  it("busy refusals offer try again; paid and cold ones have their own recourse", () => {
    for (const code of ["gpu_busy", "cloud_busy", "audition_in_flight", "gpu_busy_retry"]) {
      expect(RETRIABLE.has(explainAuditionRefusal(err(code), "kokoro").kind)).toBe(true);
    }
    expect(RETRIABLE.has(explainAuditionRefusal(err("payment_confirmation_required"), "elevenlabs").kind)).toBe(false);
    expect(RETRIABLE.has(explainAuditionRefusal(err("engine_cold"), "kokoro").kind)).toBe(false);
  });

  it("formats a fraction of a cent so it never reads as free", () => {
    expect(formatSampleCost(0.0004)).toBe("$0.0004");
    expect(formatSampleCost(Number.NaN)).toBe("$0.0000");
  });
});
