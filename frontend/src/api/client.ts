import type { ErrorEnvelope } from "./types";

/** Every non-2xx from the API carries the uniform envelope; the `code` is the stable
    machine string screens branch on (quote_expired -> re-mint silently, render_active ->
    link the blocking job from `detail`, ...). */
export class ApiError extends Error {
  status: number;
  code: string;
  detail: unknown;

  constructor(status: number, code: string, message: string, detail: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

/** The one command that brings the API up (run from the repo root). */
export const BACKEND_START_COMMAND = "uv run uvicorn seiyuu.api.main:app";

/** Code for "nothing answered": the fetch itself failed, or the dev proxy answered for a
    backend that isn't there. Status 0 = no HTTP response at all. */
export const BACKEND_DOWN = "backend_down";

export const BACKEND_DOWN_MESSAGE = `The Seiyuu backend isn't running. Start it from the project folder with: ${BACKEND_START_COMMAND}`;

export function isBackendDown(err: unknown): boolean {
  return err instanceof ApiError && err.code === BACKEND_DOWN;
}

function backendDown(status: number, cause: string): ApiError {
  return new ApiError(status, BACKEND_DOWN, BACKEND_DOWN_MESSAGE, { cause });
}

// The backend's catch-all handler answers EVERY failure with the JSON envelope, so a
// 500/502/504 whose body is empty or not JSON came from the dev proxy (Vite answers 500
// with an empty body when uvicorn is down), not from Seiyuu.
const PROXY_DOWN_STATUSES = new Set([500, 502, 504]);

function isEnvelope(body: unknown): body is ErrorEnvelope {
  if (typeof body !== "object" || body === null || !("error" in body)) return false;
  const error = (body as { error: unknown }).error;
  return typeof error === "object" && error !== null && typeof (error as { code?: unknown }).code === "string";
}

async function toApiError(res: Response): Promise<ApiError> {
  let text = "";
  try {
    text = await res.text();
  } catch {
    // unreadable body: treat as empty
  }
  let parsed = false;
  let body: unknown = null;
  if (text.trim() !== "") {
    try {
      body = JSON.parse(text);
      parsed = true;
    } catch {
      // not JSON
    }
  }
  if (parsed && isEnvelope(body)) {
    return new ApiError(res.status, body.error.code, body.error.message, body.error.detail);
  }
  if (!parsed && PROXY_DOWN_STATUSES.has(res.status)) {
    return backendDown(res.status, `${res.status} ${res.statusText}`.trim());
  }
  return new ApiError(res.status, "http_error", `${res.status} ${res.statusText}`);
}

function isAbort(err: unknown, init?: RequestInit): boolean {
  return init?.signal?.aborted === true || (err instanceof Error && err.name === "AbortError");
}

/** Like api(), but also hands back the HTTP status for the few callers whose meaning
    depends on it (ingest: 201 = newly added, 200 = these exact bytes were already here). */
export async function apiWithStatus<T>(path: string, init?: RequestInit): Promise<{ data: T; status: number }> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (err) {
    // a superseded query's abort is not an outage — let react-query see the real abort
    if (isAbort(err, init)) throw err;
    throw backendDown(0, err instanceof Error ? err.message : String(err));
  }
  if (!res.ok) throw await toApiError(res);
  if (res.status === 204) return { data: undefined as T, status: res.status };
  return { data: (await res.json()) as T, status: res.status };
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  return (await apiWithStatus<T>(path, init)).data;
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function postForm<T>(path: string, form: FormData): Promise<T> {
  return api<T>(path, { method: "POST", body: form });
}

export function postFormWithStatus<T>(path: string, form: FormData): Promise<{ data: T; status: number }> {
  return apiWithStatus<T>(path, { method: "POST", body: form });
}
