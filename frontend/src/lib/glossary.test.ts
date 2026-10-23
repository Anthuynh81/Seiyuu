import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import {
  cliHint,
  explainFailure,
  failureLine,
  failureOf,
  JOB_CANCEL,
  JOB_DOING,
  MODE_NAMES,
  plural,
  SCREENS,
  term,
  TERMS,
  type JobErrorId,
} from "./glossary";
import { MODE_LABEL } from "./render";

describe("glossary terms", () => {
  // the README / plan table: UI word ↔ CLI term
  it.each([
    ["ingest", "import", "ingest"],
    ["attribute", "find the speakers", "attribute"],
    ["assign", "cast the voices", "assign"],
    ["render", "record", "render"],
    ["master", "build the audiobook", "master"],
    ["segment", "line", "segment"],
    ["block", "paragraph", "block"],
    ["lowConfidence", "check this", "low-confidence"],
    ["quote", "get the price", "mint quote"],
    ["validation", "audio check", "whisper validation"],
    ["multivoice", "Full cast", "multivoice"],
    ["caseSensitive", "match capitals exactly", "case_sensitive"],
  ] as const)("%s → %s (CLI %s)", (key, ui, cli) => {
    expect(term(key)).toBe(ui);
    expect(TERMS[key].cli).toBe(cli);
    expect(cliHint(key)).toBe(`(CLI: ${cli})`);
  });

  it("pluralises by count", () => {
    expect(plural("segment", 1)).toBe("line");
    expect(plural("segment", 3)).toBe("lines");
    expect(plural("block", 0)).toBe("paragraphs");
  });

  it("is the one source for mode names, screen names and job words", () => {
    expect(MODE_LABEL).toBe(MODE_NAMES);
    expect(MODE_NAMES).toEqual({ multivoice: "Full cast", single: "One narrator" });
    expect(SCREENS.render).toBe("Produce");
    expect(JOB_DOING.attribute).toBe("Finding the speakers");
    expect(JOB_CANCEL.render).toBe("Cancel recording");
  });

  it("no UI word is a raw code", () => {
    for (const t of Object.values(TERMS)) expect(t.ui).not.toMatch(/_/);
  });
});

describe("explainFailure: raw codes → a sentence + a recourse", () => {
  const job = { job_id: "job-7", kind: "render", state: "running" };

  // Every code the API raises today maps to plain words. Generated from the backend with
  //   grep -rhozP '(ApiError|envelope)\(\s*\d{3},\s*"[a-z_]+"' src/seiyuu   (+ money.py's
  // gate codes, which are computed): re-run it when a route gains a code.
  const BACKEND_CODES = [
    "alignment_failed", "audition_in_flight", "cloud_busy", "conflict", "conflicting_job",
    "consent_invalid", "corrupt_artifact", "duplicate_job", "edit_conflict", "engine_cold",
    "full_render_confirmation_required", "gpu_busy", "gpu_busy_retry", "illegal_transition",
    "internal", "invalid", "lock_timeout", "not_found", "not_ready", "nothing_to_quote",
    "nothing_to_warm", "partial_delete", "payload_too_large", "payment_confirmation_required",
    "reclone_blocked", "render_active", "stage_prerequisite", "token_required",
    "unsupported_media_type", "upstream", "upstream_error", "voice_exists", "voice_referenced",
    // money.py gate_code()
    "quote_mismatch", "quote_used", "quote_expired", "quote_signature_invalid", "ceiling_exceeded",
    "cost_drift", "quote_refused",
  ];

  it.each(BACKEND_CODES)("%s is a known code", (code) => {
    const out = explainFailure({ code, message: "x" });
    expect(out.known).toBe(true);
    expect(out.id).not.toBe("unknown");
  });

  // the real backend messages where a code is split by its message
  const table: [string, string, JobErrorId][] = [
    ["render_active", "a render job for 'b1' is running", "render_active"],
    ["conflicting_job", "a master job for 'b1' is running", "conflicting_job"],
    ["gpu_busy", "busy", "gpu_busy"],
    ["engine_cold", "cold", "engine_cold"],
    ["audition_in_flight", "x", "audition_in_flight"],
    ["cloud_busy", "x", "cloud_busy"],
    ["quote_expired", "x", "quote_expired"],
    ["quote_mismatch", "x", "quote_stale"],
    ["quote_used", "x", "quote_stale"],
    ["cost_drift", "x", "quote_stale"],
    ["token_required", "x", "quote_needed"],
    ["ceiling_exceeded", "x", "ceiling_exceeded"],
    ["payment_confirmation_required", "x", "paid_confirm"],
    ["stage_prerequisite", "book 'b1' is not ingested; run ingest first", "not_imported"],
    ["stage_prerequisite", "book 'b1' has no render; run render first", "not_recorded"],
    ["not_found", "book 'b1' has no attribution; run attribute first", "no_speakers_yet"],
    ["stage_prerequisite", "no assignment at x; run `seiyuu assign b1` first", "not_cast"],
    ["stage_prerequisite", "something else", "prerequisite"],
    ["edit_conflict", "x", "edit_conflict"],
    ["voice_referenced", "x", "voice_in_use"],
    ["voice_exists", "x", "voice_exists"],
    ["consent_invalid", "x", "consent_invalid"],
    ["upstream", "x", "upstream"],
    ["upstream_error", "x", "upstream"],
    ["corrupt_artifact", "x", "damaged_file"],
    ["backend_down", "x", "backend_down"],
    ["not_ready", "job runner is not running; start() it before enqueueing", "starting"],
    ["not_ready", "ANTHROPIC_API_KEY not set; required for anthropic/hybrid", "key_missing"],
    ["not_ready", "ANTHROPIC_API_KEY not set; required for the anthropic suggester", "key_missing"],
    ["not_ready", "ELEVENLABS_API_KEY not set; required for paid auditions", "key_missing"],
    ["duplicate_job", "a master job for 'b1' is already queued", "duplicate_job"],
    ["lock_timeout", "database is locked", "busy_saving"],
    ["illegal_transition", "job job-7: running -> queued", "job_already_ended"],
    ["internal", "unhandled KeyError", "internal"],
    ["conflict", "series 'stormlight' already exists", "series_exists"],
    ["nothing_to_quote", "every segment in scope is free or cached — render without a token", "nothing_to_quote"],
    ["nothing_to_warm", "elevenlabs is a cloud engine; it holds no local weights", "nothing_to_warm"],
    ["alignment_failed", "forced alignment failed for 'ch001_b0001'[0]: boom", "alignment_failed"],
    ["partial_delete", "book 'b1' was only partially deleted; 2 path(s) could not be removed", "partial_delete"],
    ["reclone_blocked", "voice 'v1' already exists; re-clone replaces …", "reclone_blocked"],
    ["full_render_confirmation_required", "full-book render: 900 speakable segments", "whole_book_confirm"],
    ["stage_prerequisite", "book 'b1': run assign first", "not_cast"],
    ["not_found", "book 'b1': run assign first", "not_cast"],
    ["stage_prerequisite", "book 'b1': run attribute first", "no_speakers_yet"],
    ["not_found", "book 'b1' has no mastered m4b; run master first", "not_built"],
    ["not_found", "no assembled chapter 3; run assemble first", "not_built"],
    ["not_found", "book 'b1' has no manual edits to undo", "nothing_to_undo"],
    ["quote_signature_invalid", "x", "quote_stale"],
    ["quote_refused", "x", "quote_stale"],
    ["gpu_busy_retry", "x", "gpu_busy"],
    ["payload_too_large", "x", "too_large"],
    ["unsupported_media_type", "x", "wrong_file_type"],
    ["not_found", "voice 'v9' not found", "not_found"],
  ];

  it.each(table)("%s (%s) → %s", (code, message, id) => {
    const out = explainFailure({ code, message, detail: job });
    expect(out.id).toBe(id);
    expect(out.known).toBe(true);
    // plain words: never the raw code, never a quoted id, never "Error:"
    expect(out.summary).not.toMatch(/_|Error:|'b1'|job-7/);
    expect(out.summary).not.toContain(code);
    expect(out.fix).toBeTruthy();
  });

  it("render_active offers [Cancel recording] for the blocking job", () => {
    const out = explainFailure({ code: "render_active", message: "a render job for 'b1' is running", detail: job });
    expect(out.summary).toBe("A recording is running for this book; changes are paused until it finishes.");
    expect(out.recourse).toEqual({ kind: "cancel_job", jobId: "job-7", label: "Cancel recording" });
  });

  it("conflicting_job names what is running, and its stop button fits the kind", () => {
    const out = explainFailure({
      code: "conflicting_job",
      message: "x",
      detail: { job_id: "j2", kind: "attribute", state: "queued" },
    });
    expect(out.summary).toBe("Finding the speakers is still running for this book, so this has to wait.");
    expect(out.recourse?.label).toBe("Stop finding the speakers");
  });

  it("a missing Anthropic key says so, never 'still starting'", () => {
    const out = explainFailure({ code: "not_ready", message: "ANTHROPIC_API_KEY not set; required for the anthropic LLM caster" });
    expect(out.summary).toBe("Using Claude online needs an Anthropic key, and none is set.");
    expect(out.fix).toMatch(/ANTHROPIC_API_KEY.*or use the local model/);
    expect(out.summary).not.toMatch(/starting/);
  });

  it("duplicate_job names the job already under way and offers to stop it", () => {
    const out = explainFailure({
      code: "duplicate_job",
      message: "a master job for 'b1' is already queued",
      detail: { job_id: "j9", kind: "master", state: "queued" },
    });
    expect(out.summary).toBe("Building the .m4b audiobook is already waiting to start for this book.");
    expect(out.recourse).toEqual({ kind: "cancel_job", jobId: "j9", label: "Stop building the audiobook" });
  });

  it("a global render_active (the Voices guard) names the job and doesn't claim 'this book'", () => {
    const out = explainFailure({
      code: "render_active",
      message: "a master job is running; re-cloning this voice would corrupt the cached segments",
      detail: { job_id: "j3", kind: "master", state: "running" },
    });
    expect(out.summary).toBe("Building the .m4b audiobook is under way; this has to wait until it finishes.");
    expect(out.recourse?.label).toBe("Stop building the audiobook");
    const rec = explainFailure({
      code: "render_active",
      message: "a render job is queued; deleting this voice would corrupt …",
      detail: { job_id: "j4", kind: "render", state: "queued" },
    });
    expect(rec.summary).toBe("A recording is running; this has to wait until it finishes.");
  });

  it("render_active keeps where to cancel in the fix, for spots without a button", () => {
    expect(explainFailure({ code: "render_active", message: "a render job for 'b1' is running" }).fix).toMatch(
      /cancel it in the bar at the bottom/,
    );
  });

  it("reclone_blocked counts the recorded books that would need re-recording", () => {
    expect(explainFailure({ code: "reclone_blocked", message: "x", detail: { stale_books: ["b1", "b2"] } }).fix).toMatch(
      /^2 recorded books would need re-recording\. .*billed again/,
    );
    expect(explainFailure({ code: "reclone_blocked", message: "x", detail: { stale_books: [] } }).fix).toMatch(
      /^No recorded book uses it yet\./,
    );
  });

  it("no recourse without a blocking job in the detail", () => {
    expect(explainFailure({ code: "render_active", message: "x" }).recourse).toBeUndefined();
  });

  it("an invalid request keeps the server's reason as the fix, without a duplicate Details copy", () => {
    const out = explainFailure({ code: "invalid", message: "chapter 99 is out of range" });
    expect(out.summary).toBe("Seiyuu couldn't accept that.");
    expect(out.fix).toBe("chapter 99 is out of range");
    expect(out.details).toBeNull();
  });

  it("failureOf reads ApiErrors, Errors and strings", () => {
    const api = new ApiError(409, "render_active", "msg", job);
    expect(failureOf(api)).toEqual({ code: "render_active", message: "msg", detail: job });
    expect(failureOf(new Error("boom"))).toEqual({ message: "boom" });
    expect(failureOf("plain")).toEqual({ message: "plain" });
    expect(failureOf(null)).toEqual({});
  });

  it("failureLine is one sentence for known failures and keeps the server's words for unknown ones", () => {
    expect(failureLine(new ApiError(409, "edit_conflict", "anchor moved"))).toBe(
      "That line changed since this page loaded, so the change wasn't saved. Reload the chapter and make the change again.",
    );
    expect(failureLine(new ApiError(500, "weird", "kaboom"), "Couldn't save")).toBe("Couldn't save: kaboom");
  });
});
