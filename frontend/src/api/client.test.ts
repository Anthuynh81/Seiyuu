import { describe, expect, it } from "vitest";

import { jsonResponse, mockApi } from "../test/utils";
import { api, ApiError, postForm, postJson } from "./client";

describe("api()", () => {
  it("returns the parsed JSON body on 2xx", async () => {
    mockApi().get("/api/ping", { pong: true, n: 3 });
    await expect(api<{ pong: boolean; n: number }>("/api/ping")).resolves.toEqual({
      pong: true,
      n: 3,
    });
  });

  it("throws ApiError carrying status/code/message/detail from the error envelope", async () => {
    mockApi().error("GET", "/api/books/b1", 409, "conflicting_job", "a job is still running", {
      job_id: "j7",
    });
    const err = await api("/api/books/b1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const apiErr = err as ApiError;
    expect(apiErr.status).toBe(409);
    expect(apiErr.code).toBe("conflicting_job");
    expect(apiErr.message).toBe("a job is still running");
    expect(apiErr.detail).toEqual({ job_id: "j7" });
  });

  it("falls back to code http_error with status + statusText when the error body is not JSON", async () => {
    mockApi().on(
      "GET",
      "/api/broken",
      () => new Response("<html>gateway melted</html>", { status: 502, statusText: "Bad Gateway" }),
    );
    const err = await api("/api/broken").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const apiErr = err as ApiError;
    expect(apiErr.status).toBe(502);
    expect(apiErr.code).toBe("http_error");
    expect(apiErr.message).toBe("502 Bad Gateway");
    expect(apiErr.detail).toBeNull();
  });

  it("resolves undefined on 204 No Content instead of trying to parse a body", async () => {
    mockApi().on("DELETE", "/api/things/t1", () => jsonResponse(null, 204));
    await expect(api<undefined>("/api/things/t1", { method: "DELETE" })).resolves.toBeUndefined();
  });
});

describe("postJson", () => {
  it("sends Content-Type application/json with the serialized body", async () => {
    const server = mockApi();
    let captured: RequestInit | undefined;
    server.on("POST", "/api/echo", (_url, init) => {
      captured = init;
      return jsonResponse({ ok: true });
    });

    await expect(postJson<{ ok: boolean }>("/api/echo", { a: 1, b: ["x"] })).resolves.toEqual({
      ok: true,
    });
    expect(new Headers(captured?.headers).get("content-type")).toBe("application/json");
    expect(server.jsonBodyOf("POST", "/api/echo")).toEqual({ a: 1, b: ["x"] });
  });
});

describe("postForm", () => {
  it("passes the FormData through untouched and sets no Content-Type header", async () => {
    const server = mockApi();
    let captured: RequestInit | undefined;
    server.on("POST", "/api/voices/clone", (_url, init) => {
      captured = init;
      return jsonResponse({ voice_id: "v1" });
    });

    const form = new FormData();
    form.append("name", "Narrator");
    form.append("file", new File(["wav bytes"], "ref.wav", { type: "audio/wav" }));
    await postForm("/api/voices/clone", form);

    expect(captured?.body).toBe(form); // the exact instance — no re-wrapping or serialization
    // no manual Content-Type: the browser must set the multipart boundary itself
    expect(captured?.headers).toBeUndefined();
    const recorded = server.formBodyOf("POST", "/api/voices/clone");
    expect(recorded.get("name")).toBe("Narrator");
    expect((recorded.get("file") as File).name).toBe("ref.wav");
  });
});
