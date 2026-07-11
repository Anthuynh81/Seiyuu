import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { PlayerProvider } from "../app/player";

/** Fresh client per test: no retries (an ApiError should surface immediately, matching the
    app's "typed and actionable" stance) and no window-focus refetches (jsdom fires focus). */
export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
}

/** Render under the real provider stack (react-query + router + player) the way main.tsx
    composes it. Screens are routed components; pass `route` when one reads the location. */
export function renderWithProviders(
  ui: ReactElement,
  { route = "/", queryClient = makeQueryClient() }: { route?: string; queryClient?: QueryClient } = {},
) {
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[route]}>
          <PlayerProvider>{children}</PlayerProvider>
        </MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { queryClient, ...render(ui, { wrapper: Wrapper }) };
}

// -- fetch route mock ------------------------------------------------------------------------

export interface RecordedCall {
  method: string;
  url: string; // as the app called fetch: path + query
  body: string | FormData | null;
}

/** String without "?" matches the pathname alone (any query); with "?" it must equal
    pathname+search exactly; a RegExp tests pathname+search. */
export type RouteMatch = string | RegExp;

type HandlerFn = (url: URL, init: RequestInit | undefined) => Response | unknown;

interface RouteEntry {
  method: string;
  match: RouteMatch;
  handler: HandlerFn;
}

export function jsonResponse(body: unknown, status = 200): Response {
  if (status === 204) return new Response(null, { status });
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** The uniform non-2xx envelope client.ts turns into ApiError. */
export function errorResponse(status: number, code: string, message: string, detail: unknown = null): Response {
  return jsonResponse({ error: { code, message, detail } }, status);
}

function matches(entry: RouteEntry, method: string, url: URL): boolean {
  if (entry.method !== method) return false;
  const full = url.pathname + url.search;
  if (entry.match instanceof RegExp) return entry.match.test(full);
  return entry.match.includes("?") ? entry.match === full : entry.match === url.pathname;
}

export interface MockApi {
  /** Every request the app made, in order — assert on method/url/body. */
  calls: RecordedCall[];
  /** Requests no route matched (each also rejects loudly); assert empty when it matters. */
  unmatched: RecordedCall[];
  /** Register a handler; the LATEST matching registration wins, so re-register to override
      (e.g. the refetch after a save returns the new payload). Return a Response for full
      control; any other value becomes 200 JSON. */
  on(method: string, match: RouteMatch, handler: HandlerFn): MockApi;
  get(match: RouteMatch, body: unknown, status?: number): MockApi;
  post(match: RouteMatch, body: unknown, status?: number): MockApi;
  put(match: RouteMatch, body: unknown, status?: number): MockApi;
  delete(match: RouteMatch, body: unknown, status?: number): MockApi;
  /** Register a non-2xx reply in the ApiError envelope. */
  error(method: string, match: RouteMatch, status: number, code: string, message: string, detail?: unknown): MockApi;
  /** Parsed JSON body of the most recent call matching method + url substring. */
  jsonBodyOf(method: string, urlIncludes: string): unknown;
  /** FormData body of the most recent call matching method + url substring. */
  formBodyOf(method: string, urlIncludes: string): FormData;
  lastCall(method: string, urlIncludes: string): RecordedCall | undefined;
}

/** Install a route-table fetch mock (via vi.stubGlobal, so vi.unstubAllGlobals or the next
    mockApi() replaces it cleanly). GET /api/jobs (any query) is pre-registered to `{jobs: []}`
    because nearly every screen mounts the live-jobs poll; override it when a test needs jobs. */
export function mockApi(): MockApi {
  const routes: RouteEntry[] = [];
  const calls: RecordedCall[] = [];
  const unmatched: RecordedCall[] = [];

  const server: MockApi = {
    calls,
    unmatched,
    on(method, match, handler) {
      routes.push({ method: method.toUpperCase(), match, handler });
      return server;
    },
    get: (match, body, status = 200) => server.on("GET", match, () => jsonResponse(body, status)),
    post: (match, body, status = 200) => server.on("POST", match, () => jsonResponse(body, status)),
    put: (match, body, status = 200) => server.on("PUT", match, () => jsonResponse(body, status)),
    delete: (match, body, status = 200) => server.on("DELETE", match, () => jsonResponse(body, status)),
    error: (method, match, status, code, message, detail = null) =>
      server.on(method, match, () => errorResponse(status, code, message, detail)),
    lastCall(method, urlIncludes) {
      const m = method.toUpperCase();
      return [...calls].reverse().find((c) => c.method === m && c.url.includes(urlIncludes));
    },
    jsonBodyOf(method, urlIncludes) {
      const call = server.lastCall(method, urlIncludes);
      if (!call || typeof call.body !== "string") {
        throw new Error(`no recorded ${method} ${urlIncludes} call with a JSON body`);
      }
      return JSON.parse(call.body);
    },
    formBodyOf(method, urlIncludes) {
      const call = server.lastCall(method, urlIncludes);
      if (!call || !(call.body instanceof FormData)) {
        throw new Error(`no recorded ${method} ${urlIncludes} call with a FormData body`);
      }
      return call.body;
    },
  };

  server.get("/api/jobs", { jobs: [] });

  const fetchMock = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const rawUrl = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const url = new URL(rawUrl, "http://localhost");
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body instanceof FormData ? init.body : typeof init?.body === "string" ? init.body : null;
    const call: RecordedCall = { method, url: url.pathname + url.search, body };
    calls.push(call);

    for (let i = routes.length - 1; i >= 0; i--) {
      if (matches(routes[i], method, url)) {
        const out = await routes[i].handler(url, init);
        return out instanceof Response ? out : jsonResponse(out);
      }
    }
    unmatched.push(call);
    throw new Error(
      `mockApi: no route for ${method} ${call.url} — registered: ${routes
        .map((r) => `${r.method} ${String(r.match)}`)
        .join(", ")}`,
    );
  };

  vi.stubGlobal("fetch", fetchMock);
  return server;
}
